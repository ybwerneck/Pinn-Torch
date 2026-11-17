"""Ensembles of independent models, trained by ``Trainer`` as one model.

``EnsembleNet`` holds M members of one architecture. Their parameters are
stacked along a leading member axis, shape ``(M, ...)``; every buffer and
every non-tensor attribute comes from one template member and is shared.
Members never exchange information: each member's loss depends only on its
own slice, so its gradient is the one it would get trained alone.

Single-model code is reused unchanged. Any function of a member,
``eval_func(member, *eval_args)``, is evaluated for all members by
``eval_members``: the template is called with each member's parameter slice
swapped in (``torch.func.functional_call``), either under ``torch.func.vmap``
(``vectorize=True``) or one member at a time (``vectorize=False``, the
reference). Under ``vmap`` the member code must be vmap-compatible: no
``.item()``, no Python ``float()`` of tensors, no data-dependent control flow,
and custom autograd Functions need a vmap rule.

``EnsembleLoss(loss)`` and ``EnsembleValidator(validator)`` take any object
meeting Trainer's contract for a single model (``forward(model)`` and
``name``; ``val(model)``) and meet the same contract for an ensemble, by
running the single version on every member; the single version stays
reachable as ``.single``. They belong to no particular ensemble, and their
steps (``member_losses`` / ``gather``; ``val_member`` / ``gather``) are
methods to override in a subclass::

    members = []
    for s in seeds:
        torch.manual_seed(s)
        members.append(make_model())
    model = EnsembleNet(members)
    trainer = Trainer(n_epochs, model, adaptive=False, optimizer=opt)
    trainer.add_loss(EnsembleLoss(loss), weigth=1)
    trainer.add_validator(EnsembleValidator(validator), freq=100)

The ensemble loss is the SUM of the member losses, so every member's gradient
equals its single-model gradient; Adam and other element-wise optimizers on
the stacked tensors are then M independent optimizers. Anything that mixes
members breaks this equivalence: ``Trainer(adaptive=True)`` (one weight per
loss term, learned from the summed loss), gradient clipping by a global norm,
line-search optimizers.
"""

import torch
from torch import nn
from torch.func import functional_call, vmap


def _stack(outs):
    """Stack a list of per-member outputs (tensor, tuple, list or dict)."""
    first = outs[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(outs)
    if isinstance(first, (tuple, list)):
        return type(first)(_stack([o[k] for o in outs]) for k in range(len(first)))
    if isinstance(first, dict):
        return {k: _stack([o[k] for o in outs]) for k in first}
    if isinstance(first, (int, float, bool)):
        return torch.tensor(outs)
    raise TypeError(f"cannot stack member outputs of type {type(first).__name__}")


class _Bound(nn.Module):
    """Holds the template; ``forward(fn, *args)`` returns ``fn(template, *args)``."""

    def __init__(self, template):
        super().__init__()
        self.template = template

    def forward(self, eval_func, *eval_args):
        return eval_func(self.template, *eval_args)


class EnsembleNet(nn.Module):
    """M independent members with stacked parameters.

    Parameters
    ----------
    members : sequence of nn.Module
        At least one model, all of the same class, with the same parameter
        names and shapes and equal buffers. Seed each one before building it;
        member i starts exactly where its single-model run would start.
    vectorize : bool
        True evaluates all members in one ``vmap`` call; False loops over
        them. Both give the same values.

    Attributes
    ----------
    params : nn.ParameterDict
        The stacked parameters, ``(M, ...)``; keys are the member's parameter
        names with "." replaced by "-".
    template : nn.Module
        ``members[0]``. Supplies buffers and non-tensor state to every member;
        its own parameter values are never used. Not a registered submodule.
    """

    def __init__(self, members, vectorize=True):
        super().__init__()
        members = list(members)
        if not members:
            raise ValueError("EnsembleNet needs at least one member")
        template = members[0]
        names = [n for n, _ in template.named_parameters()]
        shapes = {n: p.shape for n, p in template.named_parameters()}
        bufs = dict(template.named_buffers())
        for i, m in enumerate(members[1:], 1):
            if type(m) is not type(template):
                raise ValueError(
                    f"member {i} is a {type(m).__name__}, member 0 a "
                    f"{type(template).__name__}")
            mp = dict(m.named_parameters())
            if list(mp) != names or any(mp[n].shape != shapes[n] for n in names):
                raise ValueError(f"member {i} has different parameters than member 0")
            mb = dict(m.named_buffers())
            if list(mb) != list(bufs) or any(
                    mb[n].shape != bufs[n].shape or not torch.equal(mb[n], bufs[n])
                    for n in bufs):
                raise ValueError(
                    f"member {i} has different buffers than member 0; buffers "
                    "are shared, so only parameters may differ")

        self.n_members = len(members)
        self.vectorize = bool(vectorize)
        self._names = names
        self.params = nn.ParameterDict()
        for n in names:
            ps = [dict(m.named_parameters())[n] for m in members]
            self.params[n.replace(".", "-")] = nn.Parameter(
                torch.stack([p.detach().clone() for p in ps]),
                requires_grad=ps[0].requires_grad)
        # Not registered: its parameters must not reach parameters(),
        # state_dict() or the optimizer. _apply below moves it with the ensemble.
        object.__setattr__(self, "_bound", _Bound(template))

    @property
    def template(self):
        return self._bound.template

    def _apply(self, fn, *args, **kwargs):
        """Move/cast the template together with the stacked parameters."""
        self._bound._apply(fn, *args, **kwargs)
        return super()._apply(fn, *args, **kwargs)

    def train(self, mode=True):
        self._bound.train(mode)
        return super().train(mode)

    def __len__(self):
        return self.n_members

    # -- evaluating members ---------------------------------------------------
    def _slices(self, i=None):
        """Parameter dict for functional_call: stacked, or member i's slice."""
        return {"template." + n: (self.params[n.replace(".", "-")] if i is None
                                  else self.params[n.replace(".", "-")][i])
                for n in self._names}

    def eval_member(self, i, eval_func, *eval_args):
        """``eval_func(member_i, *eval_args)``, with gradients to member i's slice."""
        return functional_call(self._bound, self._slices(i), (eval_func,) + eval_args)

    def eval_members(self, eval_func, *eval_args):
        """``eval_func(member, *eval_args)`` for every member, stacked on axis 0.

        ``eval_args`` are shared by all members. Outputs may be tensors or
        tuples, lists and dicts of tensors.
        """
        if not self.vectorize:
            return _stack([self.eval_member(i, eval_func, *eval_args)
                           for i in range(self.n_members)])
        call = lambda p: functional_call(self._bound, p, (eval_func,) + eval_args)
        return vmap(call, randomness="error")(self._slices())

    def forward(self, *args):
        """Every member's output, ``(M, ...)``."""
        return self.eval_members(lambda m, *a: m(*a), *args)

    def __getitem__(self, i):
        """Member i, usable wherever the single model is (see ``Member``)."""
        if not -self.n_members <= i < self.n_members:
            raise IndexError(f"member {i} out of range for {self.n_members} members")
        return Member(self, i % self.n_members)

    def member_state_dict(self, i):
        """Member i's ``state_dict``, loadable by the single model."""
        sd = self.template.state_dict()
        for n in self._names:
            sd[n] = self.params[n.replace(".", "-")][i].detach().clone()
        return sd


_NAMED = {}


def _named(cls, single):
    """``cls`` renamed after the wrapped object: ``EnsembleLoss<HeatLoss>``.

    A cached subclass of ``cls`` (so a user subclass ``MyLoss`` gives
    ``MyLoss<HeatLoss>``, still an instance of ``MyLoss``).
    """
    base = getattr(cls, "_named_base", cls)
    key = (base, type(single))
    if key not in _NAMED:
        name = f"{base.__name__}<{type(single).__name__}>"
        _NAMED[key] = type(name, (base,), {"_named_base": base,
                                           "__module__": base.__module__,
                                           "__qualname__": name})
    return _NAMED[key]


class EnsembleLoss:
    """A single-model loss, evaluated for every member of an ``EnsembleNet``.

    Meets Trainer's loss contract (``name``, ``forward(model) -> scalar``)
    and belongs to no particular ensemble: ``forward`` works on whichever
    ``EnsembleNet`` it is given (vmap or loop, per its ``vectorize``).

    Parameters
    ----------
    loss : any object meeting Trainer's loss contract for a single model,
        ``forward(model) -> scalar`` and ``name`` (``LOSS``, ``GridLoss``
        subclasses, ...). Not modified; reachable as ``.single``.
    gather : optional ``gather(values) -> scalar`` on the (M,) member
        losses, still in the graph. Default: the SUM, the only choice that
        gives every member exactly its single-model gradient (a mean scales
        it by 1/M; anything nonlinear couples the members).

    Steps, each overridable in a subclass::

        forward(model)  = gather(member_losses(model))

    Under ``vmap`` ``loss.forward`` runs ONCE in total, in loop mode once per
    member, so side effects in it (batch counters, stored attributes) differ
    between the modes.

    The class is renamed after the wrapped loss: ``EnsembleLoss<HeatLoss>``.

    Attributes
    ----------
    single : the wrapped single-model loss.
    name : ``loss.name`` (the key in Trainer's loss_dict).
    values : (M,) detached member losses of the latest call.
    """

    def __new__(cls, *args, **kwargs):
        if not args:                       # copy / unpickling
            return object.__new__(cls)
        return object.__new__(_named(cls, args[0]))

    def __init__(self, loss, gather=None):
        self.single = loss
        self.name = getattr(loss, "name", "Loss")
        self.gather_func = gather
        self.values = None

    def member_losses(self, model, *loss_args):
        """The (M,) member losses, in the graph."""
        single = self.single
        vals = model.eval_members(lambda m: single.forward(m, *loss_args))
        return vals.reshape(len(model))

    def gather(self, values):
        """Consolidate the (M,) member losses into the trained scalar."""
        if self.gather_func is None:
            return values.sum()
        return self.gather_func(values)

    def forward(self, model, *loss_args):
        values = self.member_losses(model, *loss_args)
        self.values = values.detach()
        return self.gather(values)

    __call__ = forward

    def __repr__(self):
        return f"{type(self).__name__}()"


class EnsembleValidator:
    """A single-model validator, run on every member of an ``EnsembleNet``.

    Meets Trainer's validator contract (``val(model)``) and belongs to no
    particular ensemble: ``val`` works on whichever ``EnsembleNet`` it is
    given.

    Parameters
    ----------
    validator : any object meeting Trainer's validator contract for a single
        model, ``val(model)``. Not modified; reachable as ``.single``.
    gather : optional ``gather(outs) -> result`` on the stacked member
        results, e.g. statistics over the members written to a file.
        Default: return ``outs``.

    Steps, each overridable in a subclass::

        val(model) = gather(stack([val_member(model[i]) for i in range(M)]))

    ``val_member`` calls ``validator.val(member)``. The results are stacked
    on a leading member axis when they are tensors,
    numbers, or tuples/lists/dicts of them, and left a list of M otherwise.
    The same single validator is called for every member, so state it
    records in itself (a history, files in one folder) must be keyed by
    member: ``model[i].member_index`` is i.

    The class is renamed after the wrapped validator:
    ``EnsembleValidator<MapValidator>``.

    Attributes
    ----------
    single : the wrapped single-model validator.
    name : ``validator.name`` if it has one.
    """

    def __new__(cls, *args, **kwargs):
        if not args:                       # copy / unpickling
            return object.__new__(cls)
        return object.__new__(_named(cls, args[0]))

    def __init__(self, validator, gather=None):
        self.single = validator
        self.name = getattr(validator, "name", "val")
        self.gather_func = gather

    def val_member(self, member, *args, **kwargs):
        """Validate one member: ``single.val(member)``; ``member`` stands in
        for the single model."""
        return self.single.val(member, *args, **kwargs)

    def gather(self, outs):
        """Combine the stacked member results."""
        if self.gather_func is None:
            return outs
        return self.gather_func(outs)

    def val(self, model, *args, **kwargs):
        outs = [self.val_member(model[i], *args, **kwargs)
                for i in range(len(model))]
        try:
            outs = _stack(outs)
        except (TypeError, RuntimeError):
            pass                          # not stackable (None, mixed shapes...)
        return self.gather(outs)

    def __repr__(self):
        return f"{type(self).__name__}()"


class Member:
    """Member i of an ``EnsembleNet``, standing in for the single model.

    Calling it runs the template's ``forward`` with member i's parameters.
    Any other attribute is read inside that same parameter swap: a property
    (e.g. a learned constant computed from a parameter) or a value is
    evaluated there and returned; a method is wrapped so that it runs there
    when called. Gradients reach member i's slice of the stacked parameters.
    Submodule attributes are refused: used later, outside the swap, they
    would silently carry the template's own parameters.

    ``member_index`` is i.
    """

    def __init__(self, ensemble, i):
        object.__setattr__(self, "_ens", ensemble)
        object.__setattr__(self, "member_index", i)

    def __call__(self, *args, **kwargs):
        return self._ens.eval_member(self.member_index, lambda m: m(*args, **kwargs))

    def __getattr__(self, name):
        ens, i = self._ens, self.member_index
        value = ens.eval_member(i, lambda m: getattr(m, name))
        if isinstance(value, nn.Module):
            raise AttributeError(
                f"submodule '{name}' of a member is not available outside the "
                "member's parameter swap; evaluate what you need from it with "
                "EnsembleNet.eval_member")
        if callable(value):
            return lambda *a, **k: ens.eval_member(
                i, lambda m: getattr(m, name)(*a, **k))
        return value

    def __repr__(self):
        return f"Member({self.member_index} of {self._ens.n_members})"

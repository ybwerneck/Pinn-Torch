"""Deterministic CPU checks for EnsembleNet, EnsembleLoss and EnsembleValidator.

The reference is always the single model: an ensemble of M seeded members
must give the outputs, gradients and Adam trajectories of M separate runs,
in both evaluation modes (vmap and loop).
"""

import copy
import os
import tempfile
import unittest

import torch
from torch import nn

from fisiocomPinn import (EnsembleLoss, EnsembleNet, EnsembleValidator,
                          FullyConnectedNetwork, LOSS, Trainer)

MODES = (True, False)


class Toy(nn.Module):
    """Single model with a shared buffer and a learned bounded constant.

    forward() -> (N,) on the stored points; ``c`` is a property computed
    from a parameter, like a learned speed.
    """

    def __init__(self, x):
        super().__init__()
        self.net = FullyConnectedNetwork(2, 1, [8, 8])
        self.raw = nn.Parameter(torch.randn(1))
        self.register_buffer("x", x)

    @property
    def c(self):
        return 0.5 + 0.1 * torch.sigmoid(self.raw[0])

    def forward(self, x=None):
        x = self.x if x is None else x
        return self.net(x).squeeze(-1) * self.c


class ToyLoss:
    """Single-model loss using forward() and the property ``c``."""

    name = "toy"

    def __init__(self, target):
        self.target = target

    def forward(self, model):
        return ((model() - self.target) ** 2).mean() + (model.c - 0.55) ** 2


def make_members(m=3, n=17):
    gen = torch.Generator().manual_seed(123)
    x = torch.rand(n, 2, generator=gen)
    members = []
    for s in range(m):
        torch.manual_seed(s)
        members.append(Toy(x.clone()))
    target = torch.sin(3 * x[:, 0]) * torch.cos(2 * x[:, 1])
    return members, target


def flat_grads(model):
    return [p.grad.clone() for p in model.parameters()]


class EnsembleNetTests(unittest.TestCase):

    def test_stacks_the_seeded_members(self):
        members, _ = make_members()
        ens = EnsembleNet(members)
        self.assertEqual(len(ens), 3)
        for i, m in enumerate(members):
            for n, p in m.named_parameters():
                self.assertTrue(torch.equal(ens.params[n.replace(".", "-")][i], p))
        n_single = sum(p.numel() for p in members[0].parameters())
        self.assertEqual(sum(p.numel() for p in ens.parameters()), 3 * n_single)

    def test_forward_matches_members(self):
        members, _ = make_members()
        want = torch.stack([m() for m in members])
        x_new = torch.rand(5, 2)
        want_x = torch.stack([m(x_new) for m in members])
        for vec in MODES:
            with self.subTest(vectorize=vec):
                ens = EnsembleNet(members, vectorize=vec)
                torch.testing.assert_close(ens(), want)
                torch.testing.assert_close(ens(x_new), want_x)

    def test_loss_gradients_match_single_models(self):
        members, target = make_members()
        loss = ToyLoss(target)
        solo = [copy.deepcopy(m) for m in members]
        want_vals, want_grads = [], []
        for m in solo:
            v = loss.forward(m)
            v.backward()
            want_vals.append(v.detach())
            want_grads.append({n: p.grad for n, p in m.named_parameters()})
        for vec in MODES:
            with self.subTest(vectorize=vec):
                ens = EnsembleNet(members, vectorize=vec)
                el = EnsembleLoss(loss)
                self.assertIs(el.single, loss)               # points to the single version
                self.assertEqual(type(el).__name__, "EnsembleLoss<ToyLoss>")
                self.assertEqual(el.name, "toy")
                total = el.forward(ens)
                total.backward()
                torch.testing.assert_close(el.values, torch.stack(want_vals))
                torch.testing.assert_close(total.detach(), torch.stack(want_vals).sum())
                for i in range(len(members)):
                    for n, g in want_grads[i].items():
                        torch.testing.assert_close(
                            ens.params[n.replace(".", "-")].grad[i], g)

    def test_trainer_adam_matches_independent_runs(self):
        members, target = make_members()
        loss = ToyLoss(target)

        def train(model, lossobj):
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            tr = Trainer(25, model, adaptive=False, patience=None, optimizer=opt,
                         print_steps=10 ** 9)
            tr.add_loss(lossobj, weigth=1)
            tr.train()

        solo = [copy.deepcopy(m) for m in members]
        for m in solo:
            train(m, loss)
        for vec in MODES:
            with self.subTest(vectorize=vec):
                ens = EnsembleNet(copy.deepcopy(members), vectorize=vec)
                train(ens, EnsembleLoss(loss))
                for i, m in enumerate(solo):
                    for n, p in m.named_parameters():
                        torch.testing.assert_close(
                            ens.params[n.replace(".", "-")][i].detach(), p.detach(),
                            rtol=1e-5, atol=1e-6)

    def test_pinntorch_LOSS_is_reused_unchanged(self):
        torch.manual_seed(0)
        nets = []
        for s in range(2):
            torch.manual_seed(s)
            nets.append(FullyConnectedNetwork(1, 1, [6]))
        x = torch.linspace(0, 1, 9).unsqueeze(-1)
        y = x ** 2
        loss = LOSS(device="cpu", criterium="MSE", name="data", batch_size=9)
        loss.add_data(x, y)
        want = []
        for n in nets:
            n2 = copy.deepcopy(n)
            loss.forward(n2).backward()
            want.append([p.grad for p in n2.parameters()])
        for vec in MODES:
            with self.subTest(vectorize=vec):
                ens = EnsembleNet(nets, vectorize=vec)
                el = EnsembleLoss(loss)
                self.assertIs(el.single, loss)
                self.assertEqual(type(el).__name__, "EnsembleLoss<LOSS>")
                self.assertEqual(el.name, "data")
                el.forward(ens).backward()
                for i in range(2):
                    for (n, _), g in zip(nets[0].named_parameters(), want[i]):
                        torch.testing.assert_close(
                            ens.params[n.replace(".", "-")].grad[i], g)

    def test_member_stands_in_for_the_single_model(self):
        members, target = make_members()
        ens = EnsembleNet(members)
        m1 = ens[1]
        torch.testing.assert_close(m1(), members[1]())
        torch.testing.assert_close(m1.c, members[1].c)
        torch.testing.assert_close(ens[-1](), members[2]())
        self.assertTrue(m1.training)
        m1.eval()
        self.assertFalse(ens.template.training)
        ens.train()
        # gradients through the stand-in reach member 1's slice only
        ToyLoss(target).forward(m1).backward()
        g = ens.params["raw"].grad
        self.assertEqual(float(g[0].abs().sum()), 0.0)
        self.assertEqual(float(g[2].abs().sum()), 0.0)
        self.assertGreater(float(g[1].abs().sum()), 0.0)
        with self.assertRaises(AttributeError):
            m1.net
        with self.assertRaises(IndexError):
            ens[3]

    def test_member_state_dict_loads_into_the_single_model(self):
        members, _ = make_members()
        ens = EnsembleNet(members)
        with torch.no_grad():
            for p in ens.parameters():
                p.add_(0.01)
        fresh = Toy(members[0].x.clone())
        fresh.load_state_dict(ens.member_state_dict(2))
        torch.testing.assert_close(fresh(), ens()[2])

    def test_state_dict_round_trip(self):
        members, _ = make_members()
        a = EnsembleNet(copy.deepcopy(members))
        with torch.no_grad():
            for p in a.parameters():
                p.mul_(1.5)
        b = EnsembleNet(copy.deepcopy(members))
        b.load_state_dict(a.state_dict())
        torch.testing.assert_close(b(), a())
        self.assertTrue(all(k.startswith("params.") for k in a.state_dict()))

    def test_validator_loops_the_members(self):
        class Recording:
            name = "rec"

            def __init__(self):
                self.seen = []

            def val(self, model):
                self.seen.append(model.member_index)
                return model().detach()

        members, _ = make_members()
        want = torch.stack([m().detach() for m in members])
        ens = EnsembleNet(members)
        rec = Recording()
        ev = EnsembleValidator(rec)
        self.assertIs(ev.single, rec)                       # points to the single version
        self.assertEqual(type(ev).__name__, "EnsembleValidator<Recording>")
        self.assertEqual(ev.name, "rec")
        torch.testing.assert_close(ev.val(ens), want)       # stacked (M, N)
        self.assertEqual(rec.seen, [0, 1, 2])               # shared state, keyed by member
        got = {}
        ev2 = EnsembleValidator(rec, gather=lambda outs: got.setdefault("o", outs).mean(0))
        torch.testing.assert_close(ev2.val(ens), want.mean(0))
        self.assertEqual(got["o"].shape, want.shape)

    def test_validator_return_mode_and_gather(self):
        class Stats:
            """return_mode toggles val() between full and measure-only."""

            def __init__(self, return_mode=False):
                self.dumped, self.return_mode = 0, return_mode

            def measure(self, model):
                return {"mean": model().mean().detach(), "c": model.c.detach()}

            def val(self, model):
                row = self.measure(model)
                if self.return_mode:
                    return row
                self.dumped += 1                      # stands for plots and dumps
                return row

        members, _ = make_members()
        ens = EnsembleNet(members)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stats.txt")

            def write_stats(outs):
                with open(path, "w") as fh:
                    fh.write(f"{float(outs['mean'].median())} {float(outs['c'].std())}\n")
                return outs

            full = Stats()
            full.val(members[0])                      # the single model, full mode
            self.assertEqual(full.dumped, 1)
            single = Stats(return_mode=True)
            outs = EnsembleValidator(single, gather=write_stats).val(ens)
            self.assertEqual(single.dumped, 0)        # no plot/dump path per member
            self.assertEqual(outs["mean"].shape, (3,))
            torch.testing.assert_close(
                outs["c"], torch.stack([m.c for m in members]).detach())
            self.assertTrue(os.path.isfile(path))

    def test_unstackable_validator_results_stay_a_list(self):
        class NoReturn:
            def val(self, model):
                model()

        members, _ = make_members()
        ens = EnsembleNet(members)
        self.assertEqual(EnsembleValidator(NoReturn()).val(ens), [None] * 3)

    def test_loss_gather_consolidates_other_than_the_sum(self):
        members, target = make_members()
        loss = ToyLoss(target)
        solo = [copy.deepcopy(m) for m in members]
        want = []
        for m in solo:
            loss.forward(m).backward()
            want.append(dict(m.named_parameters()))
        for vec in MODES:
            with self.subTest(vectorize=vec):
                ens = EnsembleNet(members, vectorize=vec)
                el = EnsembleLoss(loss, gather=lambda v: v.mean())
                total = el.forward(ens)
                torch.testing.assert_close(total.detach(), el.values.mean())
                total.backward()
                for i in range(3):     # a mean scales each member's gradient by 1/M
                    torch.testing.assert_close(ens.params["raw"].grad[i],
                                               want[i]["raw"].grad / 3)

        ens = EnsembleNet(members)
        el = EnsembleLoss(loss, gather=lambda v: v.max())
        torch.testing.assert_close(el.forward(ens).detach(), el.values.max())

    def test_rejects_members_that_differ_beyond_parameters(self):
        members, _ = make_members(2)
        other = Toy(members[0].x + 1.0)
        with self.assertRaises(ValueError):
            EnsembleNet([members[0], other])          # different buffer
        torch.manual_seed(0)
        wide = Toy(members[0].x.clone())
        wide.net = FullyConnectedNetwork(2, 1, [9, 8])
        with self.assertRaises(ValueError):
            EnsembleNet([members[0], wide])           # different shapes
        with self.assertRaises(ValueError):
            EnsembleNet([])

    def test_one_wrapper_serves_any_ensemble(self):
        members, target = make_members(4)
        a, b = EnsembleNet(members[:2]), EnsembleNet(members, vectorize=False)
        el = EnsembleLoss(ToyLoss(target))
        el.forward(a)
        self.assertEqual(el.values.shape, (2,))
        el.forward(b)
        self.assertEqual(el.values.shape, (4,))
        class Out:
            def val(self, model):
                return model().detach()

        ev = EnsembleValidator(Out())
        self.assertEqual(ev.val(a).shape[0], 2)
        self.assertEqual(ev.val(b).shape[0], 4)

    def test_extending_by_subclass(self):
        members, target = make_members()
        ens = EnsembleNet(members)

        class MeanLoss(EnsembleLoss):
            def gather(self, values):
                return values.mean()

        class FirstTwo(EnsembleLoss):
            def member_losses(self, model):
                return super().member_losses(model)[:2]

        ml = MeanLoss(ToyLoss(target))
        self.assertIsInstance(ml, MeanLoss)
        self.assertIsInstance(ml, EnsembleLoss)
        self.assertEqual(type(ml).__name__, "MeanLoss<ToyLoss>")
        torch.testing.assert_close(ml.forward(ens).detach(), ml.values.mean())
        ft = FirstTwo(ToyLoss(target))
        self.assertEqual(ft.forward(ens).shape, ())
        self.assertEqual(ft.values.shape, (2,))

        class Plain:
            name = "plain"

            def val(self, model):
                return model().detach()

        class MaxOnly(EnsembleValidator):
            def val_member(self, member):
                return super().val_member(member).max()

            def gather(self, outs):
                return outs.argmax()

        mo = MaxOnly(Plain())
        self.assertEqual(type(mo).__name__, "MaxOnly<Plain>")
        self.assertEqual(mo.name, "plain")
        want = torch.stack([m().detach().max() for m in members]).argmax()
        self.assertEqual(int(mo.val(ens)), int(want))
        self.assertIs(type(MaxOnly(Plain())), type(mo))      # named class is cached

    def test_casting_moves_the_template(self):
        members, _ = make_members()
        ens = EnsembleNet(members).double()
        self.assertEqual(ens.template.x.dtype, torch.float64)
        self.assertEqual(ens.params["raw"].dtype, torch.float64)
        self.assertEqual(ens().dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()

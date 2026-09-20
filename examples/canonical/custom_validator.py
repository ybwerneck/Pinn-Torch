"""
Writing a validator.

The Trainer contract is a single method::

    trainer.add_validator(obj, freq=100)   # registers (obj, freq)
    obj.val(model)                         # called when it % freq == 0

The return value is ignored, so anything worth keeping is stored on the
object. Several validators may be registered, each with its own frequency.

MeshValidator extends fisiocomPinn.Validator for the target, device, name and
call counter, and overrides val() because the base runs the model as model(x)
while these networks take one fixed input matrix.

Extension points::

    predict(model) -> field           the field the model represents
    measure(field) -> {name: value}   scalars recorded at each check
    arrays(field)  -> {key: array}    what the per-check dump stores
    panels(field)  -> list of panels  default: truth | prediction | difference

Given a grid it also draws itself: 2-D meshes with tripcolor, others as an xy
projection.

Output layout::

    output/<run>/
      {name}_err.h5           (iteration, metrics...), one row per check
      it_000000/{name}.npz    the field
      it_000000/{name}.png    one directory, shared by every validator
"""

import glob
import os

import numpy as np
import torch

from fisiocomPinn import Validator


class MeshValidator(Validator):
    """Pointwise error of a nodal field against a reference.

    Parameters
    ----------
    u_true : (N,) reference field, or None when there is nothing to compare
             against — subclasses measuring the physics pass None.
    grid   : optional Grid / Grid3D, enabling the built-in field plots.
    root   : run directory, defaulting to ``output``. Created if absent
             (``setFolder`` does not create it). Per-check output goes to
             ``root/it_XXXXXX/``. Pass None to write nothing.
    freq   : the frequency this validator is registered with; used to label
             iterations, since the Trainer does not pass ``it``.
    dump_f : write arrays and draw every ``dump_f`` checks.
    """

    def __init__(self, u_true=None, grid=None, name='val', freq=1, root='output',
                 dump_f=1, device='cpu', cmap='viridis', math_labels=None,
                 inputs=None):
        n = 1 if u_true is None else len(np.asarray(u_true).ravel())
        target = (torch.zeros(n, 1) if u_true is None else
                  torch.from_numpy(
                      np.asarray(u_true, dtype=np.float32).ravel()).reshape(-1, 1))
        super().__init__(
            data_in=torch.zeros_like(target),
            target=target,
            name=name,
            device=torch.device(device),
            dump_f=max(int(dump_f), 1),   # the base divides by dump_f; 0 raises
        )
        self.u_true = None if u_true is None else np.asarray(u_true).ravel()
        self.grid = grid
        self.freq = freq
        # nets taking coordinates are called model(inputs), nets carrying
        # their own encoding are called model()
        self.inputs = inputs
        self.plot = grid is not None
        self.cmap = cmap
        # Panel titles. Pass mathtext to state the quantity exactly, e.g.
        # (r'$u^*$', r'$\hat{u}$', r'$|\hat{u}-u^*|$').
        self.math_labels = math_labels or ('truth', 'prediction', 'difference')
        self.history = []          # (iteration, {metric: value})
        self.last_dir = None
        self._clim = None          # fixed on the first dump, shared by all

        self.root = root
        if root is not None:
            os.makedirs(root, exist_ok=True)
            self.setFolder(root)   # -> root/{name}_err.h5

    # -- extension points ----------------------------------------------------

    def predict(self, model):
        """The field the model currently represents, as a (N,) array."""
        with torch.no_grad():
            out = model() if self.inputs is None else model(self.inputs)
        return out.squeeze(-1).cpu().numpy()

    def measure(self, u):
        """Scalars for this check. Override to measure something else."""
        if self.u_true is None:
            return {}
        return {'l2': float(np.sqrt(np.mean((u - self.u_true) ** 2)))}

    def arrays(self, u):
        """What to store in the per-iteration dump."""
        out = {'pred': u}
        if self.u_true is not None:
            out['target'] = self.u_true
        return out

    # -- drawing -------------------------------------------------------------

    def _limits(self, u):
        if self._clim is None:
            ref = self.u_true if self.u_true is not None else u
            self._clim = (float(np.min(ref)), float(np.max(ref)))
        return self._clim

    def _draw(self, ax, u, cmap, vmin, vmax):
        """One panel: tripcolor on a 2-D grid, an xy scatter otherwise."""
        v = self.grid.vertices
        if getattr(self.grid, 'faces', None) is not None and v.shape[1] == 2:
            import matplotlib.tri as mtri
            tri = mtri.Triangulation(v[:, 0], v[:, 1], self.grid.faces)
            h = ax.tripcolor(tri, u, shading='gouraud', cmap=cmap,
                             vmin=vmin, vmax=vmax)
        else:
            # A tet mesh carries no surface triangles; project onto xy.
            h = ax.scatter(v[:, 0], v[:, 1], c=u, s=4, cmap=cmap,
                           vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        return h

    def panels(self, u):
        """The default figure template: truth | prediction | difference.

        Returns (field, label, cmap, vmin, vmax) per panel. With no reference
        to compare against — a validator measuring the physics — it degrades to
        the single field. Override to lay the figure out differently.
        """
        vmin, vmax = self._limits(u)
        if self.u_true is None:
            return [(u, self.math_labels[0], self.cmap, None, None)]
        diff = np.abs(u - self.u_true)
        lt, lp, ld = self.math_labels
        return [
            (self.u_true, lt, self.cmap, vmin, vmax),
            (u, lp, self.cmap, vmin, vmax),
            (diff, ld, 'Reds', 0.0, float(diff.max()) or 1.0),
        ]

    def plot_field(self, u, path, title=None):
        """Draw ``panels(u)`` as one row and save it."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        panels = self.panels(u)
        fig, axes = plt.subplots(1, len(panels), squeeze=False,
                                 figsize=(4.0 * len(panels), 3.5))
        for ax, (field, label, cmap, lo, hi) in zip(axes[0], panels):
            h = self._draw(ax, field, cmap, lo, hi)
            fig.colorbar(h, ax=ax, fraction=0.046)
            ax.set_title(label, fontsize=10)
        fig.suptitle(title or self.name)
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches='tight')
        plt.close(fig)

    def plot_history(self, path, others=(), extra=None):
        """Metric evolution: this validator, any ``others``, and ``extra``.

        ``extra`` maps a label to a per-iteration sequence (e.g. the training
        loss), plotted against its own index."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        for v in (self, *others):
            steps = [it for it, _ in v.history]
            for key in (v.history[0][1] if v.history else {}):
                ax.semilogy(steps, [r[key] for _, r in v.history], 'o-',
                            label=f'{v.name}:{key}')
        for label, series in (extra or {}).items():
            ax.semilogy(np.arange(len(series)), series, '-', lw=1, alpha=0.7,
                        label=label)
        ax.set_xlabel('iteration')
        ax.set_ylabel('value')
        ax.set_title('loss and validator evolution')
        ax.grid(True, which='both', alpha=0.35)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f'[{self.name}] saved {path}')

    # -- Trainer contract ----------------------------------------------------

    def val(self, model):
        was_training = model.training
        model.eval()
        u = np.asarray(self.predict(model)).ravel()
        if was_training:
            model.train()

        row = self.measure(u)
        it = self.count * self.freq
        self.history.append((it, row))
        self.last_out = u

        dump = self.count % self.dump_f == 0
        self.count += 1

        # The base class writes {name}_err.h5 from dump_f_def, which val()
        # does not call -- so append the history here instead of shipping an
        # empty file. One growing (n, k) array, one column per metric.
        if self.root is not None and row:
            import h5py
            with h5py.File(os.path.join(self.root, f'{self.name}_err.h5'), 'a') as hf:
                vals = np.array([[it] + [row[k] for k in row]], dtype=float)
                if 'history' in hf:
                    prev = np.array(hf['history'])
                    del hf['history']
                    vals = np.vstack([prev, vals])
                ds = hf.create_dataset('history', data=vals)
                ds.attrs['columns'] = ['iteration'] + list(row)

        if dump and self.root is not None:
            self.last_dir = os.path.join(self.root, f'it_{it:06d}')
            os.makedirs(self.last_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(self.last_dir, f'{self.name}.npz'), **self.arrays(u))
            if self.plot:
                self.plot_field(u, os.path.join(self.last_dir, f'{self.name}.png'),
                                title=f'{self.name}  it={it}')

        cols = '   '.join(f'{k} = {v:.3e}' for k, v in row.items())
        print(f'  [{self.name} {it:6d}]  {cols}')
        return next(iter(row.values()), 0.0)

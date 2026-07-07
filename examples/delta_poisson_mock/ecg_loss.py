"""
Application-level ECG loss for the Delta-PoIssoNN mock.
Compatible with the FisiocomPINN Trainer: loss_obj.forward(model) -> scalar.
"""

import sys
import os
import numpy as np
import torch
import h5py
import multiprocessing as mp


class ECGLoss:
    """
    Full differentiable pipeline: network -> phi -> ECG -> normalized misfit.

        forward(model):
            p     = model(eig_vecs)              # (N, 2) unit direction field
            q     = p / (||Up|| + eps)           # apply conductivity tensor
            b     = grid.assemble_rhs(q)         # FEM RHS
            phi   = grid.solve_poisson(b)        # activation map
            V_pred = ecg_forward(phi, ...)       # lead voltages
            loss  = mean_l || (V_meas[l] - V_pred[l]) / max|V_meas[l]| ||²

    Parameters
    ----------
    grid       : Grid
    eig_vecs   : (N, Ne) torch tensor — fixed eigenfunction inputs
    grad_Z     : (n_elec, F, 2) numpy — precomputed lead field gradients
    t_grid     : (Nt,) numpy
    V_meas     : (n_elec, Nt) numpy — measured / synthetic ECG
    D          : (2, 2) numpy or None — conductivity tensor (None = isotropic)
    V0, V1     : action potential range (mV)
    G_in       : intracellular conductivity scalar
    eps        : normalisation guard
    name       : loss name (for Trainer logging)
    """

    def __init__(self, grid, eig_vecs, grad_Z, t_grid, V_meas,
                 D=None, V0=-80.0, V1=20.0, G_in=1.0, eps=1e-8,
                 name='ECGLoss'):
        self.grid     = grid
        self.eig_vecs = eig_vecs
        self.name     = name
        self.V0, self.V1 = V0, V1
        self.G_in     = G_in
        self.eps      = eps

        dtype = eig_vecs.dtype
        dev   = eig_vecs.device

        # Conductivity: U s.t. D = U^T U  (upper Cholesky factor)
        if D is None:
            self.U = None
        else:
            L = np.linalg.cholesky(np.array(D, dtype=np.float64))
            self.U = torch.tensor(L.T, dtype=dtype, device=dev)  # upper triangular

        # Fixed tensors
        self.grad_Z  = torch.tensor(grad_Z,  dtype=dtype, device=dev)  # (n_elec, F, 2)
        self.t_grid  = torch.tensor(t_grid,  dtype=dtype, device=dev)  # (Nt,)
        self.V_meas  = torch.tensor(V_meas,  dtype=dtype, device=dev)  # (n_elec, Nt)
        self.areas   = torch.tensor(grid._areas, dtype=dtype, device=dev)  # (F,)

        f = grid.faces
        self.f0 = torch.tensor(f[:, 0], device=dev)
        self.f1 = torch.tensor(f[:, 1], device=dev)
        self.f2 = torch.tensor(f[:, 2], device=dev)

        # Per-lead normalisation factor (fixed from V_meas)
        self.norm_per_lead = self.V_meas.abs().max(dim=1, keepdim=True).values + eps

    # ------------------------------------------------------------------

    def _apply_conductivity(self, p):
        """q = p / (||Up|| + eps).  For isotropic (U=None), q = p."""
        if self.U is None:
            return p
        Up   = (self.U @ p.T).T                           # (N, 2)
        norm = torch.norm(Up, dim=-1, keepdim=True)        # (N, 1)
        return p / (norm + self.eps)

    def _ecg_forward(self, phi):
        """
        Differentiable ECG forward.
        phi : (N,) torch tensor
        Returns V_pred : (n_elec, Nt)
        """
        # Activation time per triangle
        phi_T    = (phi[self.f0] + phi[self.f1] + phi[self.f2]) / 3.0  # (F,)

        # Gradient of phi per triangle via grid FEM operator
        grad_phi = self.grid.gradient(phi)                  # (F, 2)

        # dot(∇φ_T, ∇Z_l_T) * area_T  →  (n_elec, F)
        dot_w = torch.einsum('efd,fd->ef', self.grad_Z, grad_phi) * self.areas[None, :]
        dot_w = dot_w * self.G_in

        # Ṽ'(t - φ_T)  →  (Nt, F)
        xi  = self.t_grid[:, None] - phi_T[None, :]
        src = 0.5 * (self.V1 - self.V0) / torch.cosh(xi) ** 2

        return dot_w @ src.T                                # (n_elec, Nt)

    def forward(self, model):
        """
        Trainer-compatible: returns scalar loss tensor.
        """
        p     = model(self.eig_vecs)                        # (N, 2)
        q     = self._apply_conductivity(p)
        b     = self.grid.assemble_rhs(q.double())
        phi   = self.grid.solve_poisson(b).to(q.dtype)
        V_pred = self._ecg_forward(phi)

        residual = (self.V_meas - V_pred) / self.norm_per_lead
        return residual.pow(2).mean()


# ------------------------------------------------------------------
# Async plot worker (module-level so multiprocessing can pickle it)
# ------------------------------------------------------------------

def _plot_snapshot_worker(h5_path, vertices, faces, t_grid, out_dir, n_leads=9):
    """Reads one HDF5 snapshot and saves a comparison figure to out_dir."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as hf:
        phi_hat  = np.array(hf['phi_hat'])
        phi_true = np.array(hf['phi_true'])
        V_pred   = np.array(hf['V_pred'])
        V_meas   = np.array(hf['V_meas'])

    triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], faces)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, field, title, cmap in [
        (axes[0], phi_true,                   'φ true',  'hot'),
        (axes[1], phi_hat,                    'φ pred',  'hot'),
        (axes[2], np.abs(phi_hat - phi_true), '|error|', 'Reds'),
    ]:
        tc = ax.tripcolor(triang, field, shading='gouraud', cmap=cmap)
        plt.colorbar(tc, ax=ax)
        ax.set_aspect('equal')
        ax.set_title(title)

    ax = axes[3]
    cmap_lines = plt.cm.tab10
    for i in range(min(n_leads, len(V_meas))):
        c = cmap_lines(i / max(n_leads, 1))
        ax.plot(t_grid, V_meas[i], color=c, linewidth=1.2, label=f'L{i}')
        ax.plot(t_grid, V_pred[i], color=c, linewidth=1.2, linestyle='--')
    ax.set_title('ECG: meas (—) vs pred (--)')
    ax.set_xlabel('t')
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'comparison.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[plot] saved {out_path}', flush=True)


# ------------------------------------------------------------------
# ECG Validator — Trainer-compatible, dumps phi / ECG to HDF5
# ------------------------------------------------------------------

class ECGValidator:
    """
    Validation loop for the Delta-PoIssoNN pipeline.

    Runs the full model → p → Poisson → φ → ECG pipeline under
    torch.no_grad(), compares φ_hat against φ_true, and appends
    error statistics to an HDF5 file.  Every `dump_f` calls it also
    writes a full snapshot (φ_hat, φ_true, V_pred, V_meas).

    Interface mirrors fisiocomPinn.Validator so Trainer.add_validator()
    accepts it directly.

    Parameters
    ----------
    loss_fn    : ECGLoss — provides grid, eig_vecs, and ECG operators
    phi_true   : (N,) numpy array — ground-truth activation map
    folder     : str   — output directory for HDF5 files
    name       : str   — file name prefix
    dump_f     : int   — snapshot every dump_f validation calls (default 1)
    plot_async : bool  — spawn a background process to plot each snapshot
                         into a subfolder it_XXXXXX/ (default True)
    """

    def __init__(self, loss_fn, phi_true, folder, name='ecg_val',
                 dump_f=1, plot_async=True):
        self.loss_fn    = loss_fn
        self.phi_true   = torch.tensor(phi_true, dtype=loss_fn.eig_vecs.dtype)
        self.name       = name
        self.dump_f     = dump_f
        self.plot_async = plot_async
        self.count      = 0
        # cache mesh arrays for pickling into worker processes
        self._vertices  = loss_fn.grid.vertices.copy()
        self._faces     = loss_fn.grid.faces.copy()
        self._t_grid    = loss_fn.t_grid.numpy().copy()
        self._procs     = []   # track live worker processes
        self.setFolder(folder)

    def setFolder(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        with h5py.File(f"{folder}/{self.name}_err.h5", "w") as hf:
            hf.create_dataset("error_stats", data=np.empty((0, 2), dtype=np.float32))

    def val(self, model):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            p_hat   = model(self.loss_fn.eig_vecs)
            q_hat   = self.loss_fn._apply_conductivity(p_hat)
            b_hat   = self.loss_fn.grid.assemble_rhs(q_hat.double())
            phi_hat = self.loss_fn.grid.solve_poisson(b_hat).to(p_hat.dtype)
            V_pred  = self.loss_fn._ecg_forward(phi_hat)
        if was_training:
            model.train()

        phi_hat_np  = phi_hat.numpy()
        phi_true_np = self.phi_true.numpy()
        mean_err = float(np.mean(np.abs(phi_hat_np - phi_true_np)))
        max_err  = float(np.max(np.abs(phi_hat_np - phi_true_np)))

        # Append error row
        with h5py.File(f"{self.folder}/{self.name}_err.h5", "a") as hf:
            old = np.array(hf["error_stats"])
            del hf["error_stats"]
            row = np.array([[mean_err, max_err]], dtype=np.float32)
            hf.create_dataset("error_stats", data=np.vstack([old, row]))

        # Full snapshot + optional async plot
        if self.count % self.dump_f == 0:
            it_dir = os.path.join(self.folder, f'it_{self.count:06d}')
            os.makedirs(it_dir, exist_ok=True)
            snap = os.path.join(it_dir, f'{self.name}_{self.count:06d}.h5')
            with h5py.File(snap, "w") as hf:
                hf.create_dataset("phi_hat",  data=phi_hat_np.astype(np.float32))
                hf.create_dataset("phi_true", data=phi_true_np.astype(np.float32))
                hf.create_dataset("V_pred",   data=V_pred.numpy().astype(np.float32))
                hf.create_dataset("V_meas",   data=self.loss_fn.V_meas.numpy().astype(np.float32))

            if self.plot_async:
                p = mp.Process(
                    target=_plot_snapshot_worker,
                    args=(snap, self._vertices, self._faces, self._t_grid, it_dir),
                    daemon=True,
                )
                p.start()
                self._procs.append(p)

        self.count += 1
        return mean_err


# ------------------------------------------------------------------
# Standalone: analytical eikonal ground truth + 3 symmetric leads
# ------------------------------------------------------------------

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import matplotlib.pyplot as plt
    from fisiocomPinn import Grid, structured_mesh
    from visualizer import Visualizer
    from ground_truth import (wall_nodes, nearest_nodes, mesh_geodesic_eikonal,
                              make_t_grid, make_electrodes,
                              precompute_lead_gradients, ecg_forward)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    WALL   = None
    STIM   = (0.05, 0.05)
    ELEC_H = 1

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    vertices, faces = structured_mesh(n=33, L=1.0)
    grid = Grid(vertices, faces)
    viz  = Visualizer(grid)

    # ------------------------------------------------------------------
    # Ground truth via eikonal (Dijkstra on mesh)
    # ------------------------------------------------------------------
    if WALL is not None:
        source_nodes = wall_nodes(vertices, wall=WALL)
    else:
        source_nodes = nearest_nodes(vertices, [STIM])
    phi_true  = mesh_geodesic_eikonal(grid, source_nodes)
    t_grid_np = make_t_grid(phi_true, Nt=200)

    # ------------------------------------------------------------------
    # 9 leads on a 3x3 grid
    # ------------------------------------------------------------------
    electrodes = make_electrodes(L=1.0, h=ELEC_H, n_elec=9)
    grad_Z_np  = precompute_lead_gradients(grid, electrodes)
    V_meas_np  = ecg_forward(phi_true, grid, grad_Z_np, t_grid_np)

    print(f"phi_true range : {phi_true.min():.3f} – {phi_true.max():.3f}")
    print(f"V_meas   shape : {V_meas_np.shape}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: activation map + electrode positions
    ax = axes[0]
    viz.plot_field(phi_true, title='phi_true + electrodes', ax=ax, cmap='hot')
    ax.plot(STIM[0], STIM[1], 'w*', ms=12, markeredgecolor='k', label='source')
    for k, (xe, ye, _) in enumerate(electrodes):
        ax.plot(xe, ye, 'cv', ms=8, markeredgecolor='k')
        ax.text(xe + 0.02, ye + 0.02, str(k), fontsize=7, color='cyan')
    ax.legend(fontsize=8)

    # Right: 9 ECG leads
    ax = axes[1]
    cmap9 = plt.cm.tab10
    for i in range(9):
        xe, ye, _ = electrodes[i]
        c = cmap9(i / 9)
        ax.plot(t_grid_np, V_meas_np[i], color=c, linewidth=1.2,
                label=f'L{i} ({xe:.2f},{ye:.2f})')
    ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax.set_xlabel('t')
    ax.set_ylabel('V (a.u.)')
    ax.set_title('ECG leads')
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig('ecg_loss_verification.png', dpi=120)
    print("Saved ecg_loss_verification.png")
    plt.show()

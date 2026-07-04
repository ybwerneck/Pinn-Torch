"""
Application-level ECG loss for the Delta-PoIssoNN mock.
Compatible with the FisiocomPINN Trainer: loss_obj.forward(model) -> scalar.
"""

import sys
import os
import numpy as np
import torch


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
# Standalone: round-trip verification + ground truth ECG plot
# ------------------------------------------------------------------

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import matplotlib.pyplot as plt
    from fisiocomPinn import Grid, structured_mesh, EigenDirectionNet
    from visualizer import Visualizer
    from ground_truth import (make_phi_true, make_t_grid,
                              make_electrodes, precompute_lead_gradients,
                              ecg_forward)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    N_SIDE  = 33
    D_ANISO = np.array([[3.0, 0.5], [0.5, 1.0]])
    STIM    = (0.05, 0.05)

    vertices, faces = structured_mesh(n=N_SIDE, L=1.0)
    grid = Grid(vertices, faces)
    viz  = Visualizer(grid)

    phi_true   = make_phi_true(vertices, STIM, D=D_ANISO)
    t_grid_np  = make_t_grid(phi_true, Nt=100)
    electrodes = make_electrodes(L=1.0, h=0.3, n_elec=9)
    grad_Z_np  = precompute_lead_gradients(grid, electrodes)
    V_meas_np  = ecg_forward(phi_true, grid, grad_Z_np, t_grid_np)

    # ------------------------------------------------------------------
    # Round-trip: phi_true -> p_true -> q_true -> Poisson -> phi_recon
    # ------------------------------------------------------------------
    # Step 1: gradient of phi_true per triangle (forward direction)
    grad_phi_np = grid.gradient(phi_true)                    # (F, 2)

    # Step 2: map to nodes (average adjacent triangles)
    p_nodes = np.zeros((grid.N, 2))
    count   = np.zeros(grid.N)
    for k in range(3):
        np.add.at(p_nodes, grid.faces[:, k], grad_phi_np)
        np.add.at(count,   grid.faces[:, k], 1)
    p_nodes /= count[:, None]

    # Step 3: normalize -> unit direction p_true
    p_norms = np.linalg.norm(p_nodes, axis=1, keepdims=True)
    p_true  = p_nodes / (p_norms + 1e-8)

    # Step 4: apply conductivity correction -> q = p / ||Up||
    L_chol  = np.linalg.cholesky(D_ANISO)
    U_np    = L_chol.T                                       # upper triangular
    Up      = (U_np @ p_true.T).T                           # (N, 2)
    Up_norm = np.linalg.norm(Up, axis=1, keepdims=True)
    q_true  = p_true / (Up_norm + 1e-8)                    # should ≈ grad_phi

    # Step 5: run through the differentiable Poisson solver (torch)
    q_t   = torch.tensor(q_true, dtype=torch.float64)
    b_t   = grid.assemble_rhs(q_t)
    phi_t = grid.solve_poisson(b_t)
    phi_recon = phi_t.numpy()
    phi_recon -= phi_recon.min()                             # gauge align

    error = np.mean(np.abs(phi_recon - phi_true))
    print(f"Round-trip error (mean |phi_recon - phi_true|): {error:.5f}")
    print(f"phi_true  range: {phi_true.min():.3f} – {phi_true.max():.3f}")
    print(f"phi_recon range: {phi_recon.min():.3f} – {phi_recon.max():.3f}")

    # ------------------------------------------------------------------
    # Plot: round-trip comparison + ECG
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    viz.plot_field(phi_true,  title='phi_true',      ax=axes[0], cmap='hot')
    viz.plot_field(phi_recon, title='phi_recon',     ax=axes[1], cmap='hot')
    viz.plot_field(np.abs(phi_recon - phi_true),
                   title='|error|', ax=axes[2], cmap='Reds')

    ax = axes[3]
    for i in range(len(V_meas_np)):
        ax.plot(t_grid_np, V_meas_np[i], linewidth=1.0, label=f'L{i}')
    ax.set_title('ECG leads (ground truth)')
    ax.set_xlabel('t')
    ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig('ecg_loss_verification.png', dpi=120)
    print("Saved ecg_loss_verification.png")
    plt.show()

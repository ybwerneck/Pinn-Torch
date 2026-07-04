"""
Delta-PoIssoNN mock on [0,1]^2.
Learns the anisotropic activation map from ECG leads using a PINN
on Laplace-Beltrami eigenfunctions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from fisiocomPinn import Grid, structured_mesh, EigenDirectionNet
from visualizer import Visualizer
from ground_truth import (make_phi_true, make_t_grid,
                          make_electrodes, precompute_lead_gradients,
                          ecg_forward)
from ecg_loss import ECGLoss

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
N_SIDE    = 33
L         = 1.0
N_EIG     = 16
N_LAYERS  = 5
WIDTH     = 64
N_ITER    = 500
LR        = 1e-3
LOG_EVERY = 50

STIM    = (0.05, 0.05)
D_ANISO = np.array([[3.0, 0.5],
                    [0.5, 1.0]])

# ------------------------------------------------------------------
# 1. Mesh + visualiser
# ------------------------------------------------------------------
vertices, faces = structured_mesh(n=N_SIDE, L=L)
grid = Grid(vertices, faces)
viz  = Visualizer(grid)
print(f"Mesh: {grid.N} nodes, {len(grid.faces)} triangles")

# ------------------------------------------------------------------
# 2. Laplace-Beltrami eigenfunctions
# ------------------------------------------------------------------
print(f"Computing {N_EIG} eigenfunctions...")
eig_np   = grid.eigenfunctions(n_eig=N_EIG)
eig_vecs = torch.tensor(eig_np, dtype=torch.float32)
print(f"Done. Shape: {eig_vecs.shape}")

# ------------------------------------------------------------------
# 3. Ground truth activation map + ECG
# ------------------------------------------------------------------
phi_true   = make_phi_true(vertices, STIM, D=D_ANISO)
t_grid_np  = make_t_grid(phi_true, Nt=100)
electrodes = make_electrodes(L=L, h=0.3, n_elec=9)
grad_Z_np  = precompute_lead_gradients(grid, electrodes)
V_meas_np  = ecg_forward(phi_true, grid, grad_Z_np, t_grid_np)

print(f"phi_true range: {phi_true.min():.3f} – {phi_true.max():.3f}")
print(f"t_grid   0 – {t_grid_np[-1]:.3f}  ({len(t_grid_np)} steps)")
print(f"V_meas   shape={V_meas_np.shape}  range={V_meas_np.min():.3f} – {V_meas_np.max():.3f}")

# ------------------------------------------------------------------
# 4. Network + loss
# ------------------------------------------------------------------
model = EigenDirectionNet(Ne=N_EIG, n_layers=N_LAYERS, width=WIDTH)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

loss_fn = ECGLoss(
    grid     = grid,
    eig_vecs = eig_vecs,
    grad_Z   = grad_Z_np,
    t_grid   = t_grid_np,
    V_meas   = V_meas_np,
    D        = D_ANISO,
)

optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------------
history = []

print(f"\nTraining for {N_ITER} iterations...")
for it in range(N_ITER):
    optimizer.zero_grad()
    loss = loss_fn.forward(model)
    loss.backward()
    optimizer.step()

    val = loss.item()
    history.append(val)
    if (it + 1) % LOG_EVERY == 0 or it == 0:
        print(f"  iter {it+1:4d}/{N_ITER}  loss={val:.6f}")

# ------------------------------------------------------------------
# 6. Final evaluation
# ------------------------------------------------------------------
model.eval()
with torch.no_grad():
    p_hat   = model(eig_vecs)                              # (N, 2)
    q_hat   = loss_fn._apply_conductivity(p_hat)
    b_hat   = grid.assemble_rhs(q_hat.double())
    phi_hat = grid.solve_poisson(b_hat).float().numpy()
    phi_hat -= phi_hat.min()

    phi_hat_t = grid.solve_poisson(grid.assemble_rhs(q_hat.double())).to(q_hat.dtype)
    V_pred    = loss_fn._ecg_forward(phi_hat_t).numpy()

error_phi = np.mean(np.abs(phi_hat - phi_true))
print(f"\nFinal  loss={history[-1]:.6f}  |phi_hat - phi_true| mean={error_phi:.4f}")

# ------------------------------------------------------------------
# 7. Plots
# ------------------------------------------------------------------

# Loss curve
fig_loss, ax_l = plt.subplots(figsize=(6, 3))
ax_l.semilogy(history)
ax_l.set_xlabel('iteration')
ax_l.set_ylabel('ECG loss')
ax_l.set_title('Training loss')
fig_loss.tight_layout()
fig_loss.savefig('loss_curve.png', dpi=120)
print("Saved loss_curve.png")

# Activation map comparison
fig_phi, axes = plt.subplots(1, 3, figsize=(14, 4))
viz.plot_field(phi_true,                     title='phi_true',  ax=axes[0], cmap='hot')
viz.plot_field(phi_hat,                      title='phi_hat',   ax=axes[1], cmap='hot')
viz.plot_field(np.abs(phi_hat - phi_true),   title='|error|',   ax=axes[2], cmap='Reds')
fig_phi.tight_layout()
fig_phi.savefig('phi_comparison.png', dpi=120)
print("Saved phi_comparison.png")

# ECG overlay
fig_ecg = viz.plot_leads(t_grid_np, V_meas_np, V_pred=V_pred)
fig_ecg.suptitle('ECG: measured (black) vs predicted (red)', y=1.01)
fig_ecg.tight_layout()
fig_ecg.savefig('ecg_overlay.png', dpi=120)
print("Saved ecg_overlay.png")

# Summary: direction field + phi + ECG
fig_sum = viz.plot_summary(
    phi_hat, p_hat.numpy(), t_grid_np, V_meas_np,
    electrodes=electrodes, V_pred=V_pred, n_leads_shown=9,
)
fig_sum.savefig('summary.png', dpi=120)
print("Saved summary.png")

plt.show()

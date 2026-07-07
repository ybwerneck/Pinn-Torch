"""
Delta-PoIssoNN mock — non-identifiability case.

Source at (0.05, 0.25) and its y-reflection (0.05, 0.75) are
indistinguishable when electrodes lie on the y=0.5 symmetry axis.
The PINN has two equally valid solutions; which one it finds depends
on the random seed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from fisiocomPinn import Grid, structured_mesh, EigenDirectionNet
from fisiocomPinn.Trainer import Trainer
from visualizer import Visualizer
from ground_truth import (make_phi_true, make_t_grid,
                          precompute_lead_gradients, ecg_forward)
from ecg_loss import ECGLoss, ECGValidator

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
N         = 33
N_EIG     = 100
N_LAYERS  = 12
WIDTH     = 128
N_ITER    = 1000
LR        = 5e-4
VAL_FREQ  = 200
DUMP_FREQ = 1
ELEC_H    = 0.05
OUT_DIR   = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))

STIM_A = (0.05, 0.25)           # primary source
STIM_B = (0.05, 0.75)           # y-reflected source — same ECG

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Mesh + visualiser
# ------------------------------------------------------------------
vertices, faces = structured_mesh(n=N, L=1.0)
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
# 3. Ground truth — both sources via eikonal
# ------------------------------------------------------------------
phi_A = make_phi_true(vertices, stim_point=STIM_A)
phi_B = make_phi_true(vertices, stim_point=STIM_B)
t_grid_np = make_t_grid(phi_A, Nt=100)

# Electrodes: 5-point line along y=0.5 (the symmetry axis)
electrodes = np.array([
    [0.10, 0.5, ELEC_H],
    [0.30, 0.5, ELEC_H],
    [0.50, 0.5, ELEC_H],
    [0.70, 0.5, ELEC_H],
    [0.90, 0.5, ELEC_H],
])
grad_Z_np = precompute_lead_gradients(grid, electrodes)

V_A = ecg_forward(phi_A, grid, grad_Z_np, t_grid_np)
V_B = ecg_forward(phi_B, grid, grad_Z_np, t_grid_np)

# Verify non-identifiability: V_A and V_B should be (near-)identical
max_diff = np.max(np.abs(V_A - V_B))
print(f"\n--- Non-identifiability check ---")
print(f"  max |V_A - V_B| = {max_diff:.2e}  (should be ~0)")
print(f"  phi_A range: {phi_A.min():.3f} – {phi_A.max():.3f}")
print(f"  phi_B range: {phi_B.min():.3f} – {phi_B.max():.3f}")
print(f"---------------------------------\n")

# Train on V_A; the PINN can converge to phi_A or phi_B
V_meas_np = V_A

# ------------------------------------------------------------------
# 4. Loss, model, validator
# ------------------------------------------------------------------
loss_fn = ECGLoss(
    grid     = grid,
    eig_vecs = eig_vecs,
    grad_Z   = grad_Z_np,
    t_grid   = t_grid_np,
    V_meas   = V_meas_np,
    D        = None,
)

model = EigenDirectionNet(Ne=N_EIG, n_layers=N_LAYERS, width=WIDTH)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

validator = ECGValidator(
    loss_fn  = loss_fn,
    phi_true = phi_A,          # ground truth we'll compare against
    folder   = OUT_DIR,
    name     = 'ecg_val',
    dump_f   = DUMP_FREQ,
)

# ------------------------------------------------------------------
# 5. Train
# ------------------------------------------------------------------
trainer = Trainer(
    n_epochs    = N_ITER,
    model       = model,
    adaptive    = False,
    lr          = LR,
    print_steps = 50,
)
trainer.add_loss(loss_fn, weigth=1)
trainer.add_validator(validator, freq=VAL_FREQ)

model, loss_dict = trainer.train()

# ------------------------------------------------------------------
# 6. Plots
# ------------------------------------------------------------------
history = loss_dict[loss_fn.name]
fig_loss, ax_l = plt.subplots(figsize=(6, 3))
ax_l.semilogy(history)
ax_l.set_xlabel('iteration')
ax_l.set_ylabel('ECG loss')
ax_l.set_title('Training loss')
fig_loss.tight_layout()
fig_loss.savefig(f'{OUT_DIR}/loss_curve.png', dpi=120)
print("Saved loss_curve.png")

# Summary: phi_A, phi_B, and what the network found
with torch.no_grad():
    p_hat   = model(eig_vecs)
    q_hat   = loss_fn._apply_conductivity(p_hat)
    b_hat   = loss_fn.grid.assemble_rhs(q_hat.double())
    phi_hat = loss_fn.grid.solve_poisson(b_hat).float().numpy()
    phi_hat -= phi_hat.min()

err_A = np.mean(np.abs(phi_hat - phi_A))
err_B = np.mean(np.abs(phi_hat - phi_B))
print(f"\nFinal mean error vs phi_A: {err_A:.4f}")
print(f"Final mean error vs phi_B: {err_B:.4f}")
print(f"Network converged to: {'phi_A' if err_A < err_B else 'phi_B'}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, field, title in [
    (axes[0], phi_A,   f'phi_A  (source {STIM_A})'),
    (axes[1], phi_B,   f'phi_B  (source {STIM_B})'),
    (axes[2], phi_hat, 'phi_hat (network output)'),
]:
    viz.plot_field(field, title=title, ax=ax, cmap='hot')
for xe, ye, _ in electrodes:
    for ax in axes:
        ax.plot(xe, ye, 'cv', ms=7, markeredgecolor='k')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/non_identifiability.png', dpi=120)
print("Saved non_identifiability.png")

fig_err = Visualizer.plot_err_h5(f'{OUT_DIR}/ecg_val_err.h5')
fig_err.savefig(f'{OUT_DIR}/val_error.png', dpi=120)

plt.show()

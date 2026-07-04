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
import matplotlib.pyplot as plt
from datetime import datetime

from fisiocomPinn import Grid, annular_mesh, EigenDirectionNet
from fisiocomPinn.Trainer import Trainer
from visualizer import Visualizer
from ground_truth import (make_phi_true_annular, make_t_grid,
                          make_electrodes, precompute_lead_gradients,
                          ecg_forward)
from ecg_loss import ECGLoss, ECGValidator

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
N_R         = 20       # radial layers
N_THETA     = 60       # nodes per ring
N_EIG       = 20       # more eigenfunctions for curved geometry
N_LAYERS    = 5
WIDTH       = 64
N_ITER      = 500
LR          = 1e-3
VAL_FREQ    = 50
DUMP_FREQ   = 1
OUT_DIR     = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))

# Annular domain: inner boundary = stimulus
CENTER  = (0.0, 0.0)
INNER_R = 0.4
OUTER_R = 1.0
ELEC_L  = 2 * OUTER_R   # bounding box side for electrode placement

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Mesh + visualiser
# ------------------------------------------------------------------
vertices, faces = annular_mesh(N_R, N_THETA, INNER_R, OUTER_R, center=CENTER)
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
phi_true   = make_phi_true_annular(vertices, CENTER, INNER_R)
t_grid_np  = make_t_grid(phi_true, Nt=100)
# electrodes above the annulus, centred on CENTER
electrodes = make_electrodes(L=ELEC_L, h=0.3, n_elec=9)
electrodes[:, :2] -= OUTER_R   # shift so [0,2r]^2 -> [-r,r]^2
grad_Z_np  = precompute_lead_gradients(grid, electrodes)
V_meas_np  = ecg_forward(phi_true, grid, grad_Z_np, t_grid_np)

print(f"phi_true range: {phi_true.min():.3f} – {phi_true.max():.3f}")
print(f"V_meas   shape={V_meas_np.shape}  range={V_meas_np.min():.3f} – {V_meas_np.max():.3f}")

# ------------------------------------------------------------------
# 4. Loss, model, validator
# ------------------------------------------------------------------
loss_fn = ECGLoss(
    grid     = grid,
    eig_vecs = eig_vecs,
    grad_Z   = grad_Z_np,
    t_grid   = t_grid_np,
    V_meas   = V_meas_np,
    D        = None,    # isotropic
)

model = EigenDirectionNet(Ne=N_EIG, n_layers=N_LAYERS, width=WIDTH)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

validator = ECGValidator(
    loss_fn  = loss_fn,
    phi_true = phi_true,
    folder   = OUT_DIR,
    name     = 'ecg_val',
    dump_f   = DUMP_FREQ,
)

# ------------------------------------------------------------------
# 5. Train with Trainer
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

# Loss curve from Trainer dict
history = loss_dict[loss_fn.name]
fig_loss, ax_l = plt.subplots(figsize=(6, 3))
ax_l.semilogy(history)
ax_l.set_xlabel('iteration')
ax_l.set_ylabel('ECG loss')
ax_l.set_title('Training loss')
fig_loss.tight_layout()
fig_loss.savefig(f'{OUT_DIR}/loss_curve.png', dpi=120)
print("Saved loss_curve.png")

# Validation error from HDF5
fig_err = Visualizer.plot_err_h5(f'{OUT_DIR}/ecg_val_err.h5')
fig_err.savefig(f'{OUT_DIR}/val_error.png', dpi=120)
print("Saved val_error.png")

# Final snapshot comparison (last dump)
last_snap = sorted(
    [f for f in os.listdir(OUT_DIR) if f.startswith('ecg_val_') and f.endswith('.h5') and 'err' not in f]
)[-1]
fig_snap = viz.plot_from_h5(f'{OUT_DIR}/{last_snap}', t_grid_np, n_leads_shown=9)
fig_snap.suptitle(f'Final validation: {last_snap}', y=1.01)
fig_snap.savefig(f'{OUT_DIR}/final_comparison.png', dpi=120)
print(f"Saved final_comparison.png  (from {last_snap})")

plt.show()

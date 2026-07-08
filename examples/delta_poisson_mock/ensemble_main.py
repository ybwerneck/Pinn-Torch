"""
Ensemble non-identifiability demo.

Trains M independent members simultaneously on the reflection-symmetric
non-identifiable case. Members with different random inits may converge
to phi_A or phi_B — revealing the two modes of the inverse problem.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from fisiocomPinn import Grid, structured_mesh, EnsembleNet
from fisiocomPinn.Trainer import Trainer
from visualizer import Visualizer
from ground_truth import (make_phi_true, make_t_grid,
                          precompute_lead_gradients, ecg_forward)
from ecg_loss import ECGLoss, ECGValidator

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
M         = 6           # ensemble size
N         = 33
N_EIG     = 100
N_LAYERS  = 4
WIDTH     = 32
N_ITER    = 10000
LR        = 5e-4
VAL_FREQ  = 1000
DUMP_FREQ = 1
ELEC_H    = 0.05
OUT_DIR   = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))

STIM_A = [(0.10, 0.25), (0.90, 0.25)]   # two sources, lower half
STIM_B = [(0.10, 0.75), (0.90, 0.75)]   # y-reflected — same ECG at y=0.5

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Mesh
# ------------------------------------------------------------------
vertices, faces = structured_mesh(n=N, L=1.0)
grid = Grid(vertices, faces)
viz  = Visualizer(grid)
print(f"Mesh: {grid.N} nodes, {len(grid.faces)} triangles")

# ------------------------------------------------------------------
# 2. Eigenfunctions
# ------------------------------------------------------------------
print(f"Computing {N_EIG} eigenfunctions...")
eig_np   = grid.eigenfunctions(n_eig=N_EIG)
eig_vecs = torch.tensor(eig_np, dtype=torch.float32)
print(f"Done.")

# ------------------------------------------------------------------
# 3. Ground truth — both symmetric solutions
# ------------------------------------------------------------------
phi_A = np.minimum(make_phi_true(vertices, STIM_A[0]),
                   make_phi_true(vertices, STIM_A[1]))
phi_B = np.minimum(make_phi_true(vertices, STIM_B[0]),
                   make_phi_true(vertices, STIM_B[1]))
t_grid_np = make_t_grid(phi_A, Nt=100)

electrodes = np.array([
    [0.10, 0.5, ELEC_H],
    [0.30, 0.5, ELEC_H],
    [0.50, 0.5, ELEC_H],
    [0.70, 0.5, ELEC_H],
    [0.90, 0.5, ELEC_H],
])
grad_Z_np = precompute_lead_gradients(grid, electrodes)
V_meas_np = ecg_forward(phi_A, grid, grad_Z_np, t_grid_np)

print(f"max |V_A - V_B| = {np.max(np.abs(V_meas_np - ecg_forward(phi_B, grid, grad_Z_np, t_grid_np))):.2e}")

# ------------------------------------------------------------------
# 4. Ensemble model + loss
# ------------------------------------------------------------------
model = EnsembleNet(M=M, Ne=N_EIG, n_layers=N_LAYERS, width=WIDTH)
print(f"Ensemble: {M} members × {sum(p.numel() for p in model.members[0].parameters())} params")

loss_fn = ECGLoss(
    grid     = grid,
    eig_vecs = eig_vecs,
    grad_Z   = grad_Z_np,
    t_grid   = t_grid_np,
    V_meas   = V_meas_np,
    D        = None,
)

validator = ECGValidator(
    loss_fn  = loss_fn,
    phi_true = phi_A,
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
# 6. Extract each member's phi and classify
# ------------------------------------------------------------------
model.eval()
with torch.no_grad():
    p_all = model(eig_vecs)           # (M, N_nodes, 2)

phi_hats = []
for pi in p_all:
    qi  = loss_fn._apply_conductivity(pi)
    bi  = loss_fn.grid.assemble_rhs(qi.double())
    phi_i = loss_fn.grid.solve_poisson(bi).float().numpy()
    phi_i -= phi_i.min()
    phi_hats.append(phi_i)

err_A = [np.mean(np.abs(ph - phi_A)) for ph in phi_hats]
err_B = [np.mean(np.abs(ph - phi_B)) for ph in phi_hats]
labels = ['phi_A' if a < b else 'phi_B' for a, b in zip(err_A, err_B)]

print(f"\n--- Ensemble results ---")
for i, (la, ea, eb) in enumerate(zip(labels, err_A, err_B)):
    print(f"  Member {i}: → {la}  (err_A={ea:.4f}, err_B={eb:.4f})")
n_A = labels.count('phi_A')
print(f"  {n_A}/{M} converged to phi_A, {M-n_A}/{M} to phi_B")

# ------------------------------------------------------------------
# 7. Plot
# ------------------------------------------------------------------
ncols = M + 3
fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4))

viz.plot_field(phi_A, title='phi_A', ax=axes[0], cmap='hot')
viz.plot_field(phi_B, title='phi_B', ax=axes[1], cmap='hot')

phi_mean = np.mean(phi_hats, axis=0)
phi_mean -= phi_mean.min()
viz.plot_field(phi_mean, title='ensemble mean', ax=axes[2], cmap='hot')

for i, (phi_i, la) in enumerate(zip(phi_hats, labels)):
    ax = axes[3 + i]
    viz.plot_field(phi_i, title=f'M{i} → {la}', ax=ax, cmap='hot')
    for xe, ye, _ in electrodes:
        ax.plot(xe, ye, 'cv', ms=6, markeredgecolor='k')

for ax in axes[:3]:
    for xe, ye, _ in electrodes:
        ax.plot(xe, ye, 'cv', ms=6, markeredgecolor='k')
    for sx, sy in STIM_A:
        ax.plot(sx, sy, 'w*', ms=8, markeredgecolor='k')

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/ensemble_result.png', dpi=120)
print(f"Saved ensemble_result.png")

# Loss curve
history = loss_dict[loss_fn.name]
fig2, ax2 = plt.subplots(figsize=(6, 3))
ax2.semilogy(history)
ax2.set_xlabel('iteration')
ax2.set_ylabel('mean ECG loss')
ax2.set_title(f'Ensemble training loss  (M={M})')
fig2.tight_layout()
fig2.savefig(f'{OUT_DIR}/loss_curve.png', dpi=120)

plt.show()

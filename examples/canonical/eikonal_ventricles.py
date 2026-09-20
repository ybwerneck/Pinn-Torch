"""
Eikonal on a biventricular tetrahedral mesh, 7387 nodes and 24353 tets.

The problem and the code of eikonal_2d on a real geometry. ENCODING selects
the network input:

    'eigen'    Laplace-Beltrami eigenfunctions of the mesh, the Delta-PINN
               encoding of Costabal et al. (2022)
    'coords'   raw (x, y, z) node coordinates

The mesh is normalised to unit extent because the loss weights are
scale-dependent.

The wall is not convex, so the Euclidean distance is not the geodesic.
ventricles_mesh.npz ships a Fast Iterative Method solution as the reference;
on a mesh this coarse it carries a few percent of its own error.

The mesh holds geometry only: no fibres, conduction velocities, Purkinje
trees, activation times or lead fields.

Output goes to output/eikonal_ventricles_<ENCODING>/.
"""

import os

import numpy as np
import torch

from fisiocomPinn import Grid3D, FullyConnectedNetwork, GridLoss
from fisiocomPinn.Trainer import Trainer

from custom_validator import MeshValidator

# ------------------------------------------------------------------
N_EIG    = 60
WIDTH    = 64
LAYERS   = 4
N_ITER   = 10000
LR       = 3e-4
W_SRC    = 100.0   # source condition u(x_s) = 0
W_MAX    = 0.1     # maximality: the distance is the LARGEST solution
VAL_FREQ = 500
DUMP_F   = 3
ENCODING = 'eigen'   # 'eigen' (Delta-PINN) or 'coords' (ordinary PINN)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT      = os.path.dirname(os.path.abspath(__file__))
RUN      = os.path.join(OUT, 'output', f'eikonal_ventricles_{ENCODING}')


# ------------------------------------------------------------------
# Mesh, source, reference geodesic
# ------------------------------------------------------------------
data = np.load(os.path.join(OUT, 'ventricles_mesh.npz'))
verts, tets = data['vertices'], data['tets']
# the loss weights are scale-dependent; normalise the longest extent to 1
SCALE = np.ptp(verts, axis=0).max()
verts = verts / SCALE
grid = Grid3D(verts, tets)
print(f'Mesh: {grid.N} nodes, {grid.n_tets} tets   device={DEVICE}')

src_idx = int(data['source_index'])            # earliest-activated node
u_true = data['geodesic_source'] / SCALE       # same normalisation as the mesh
print(f'Reference geodesic from apex: max {u_true.max():.4f}')


# ------------------------------------------------------------------
# Loss, identical to eikonal_2d
# ------------------------------------------------------------------
class EikonalLoss(GridLoss):
    def __init__(self, grid, V, src_idx):
        super().__init__(grid, name='eikonal')
        self.V = V
        self.src = src_idx

    def loss(self, model):
        u = model(self.V).squeeze(-1)                 # (N,)
        g = self.grad(u)                              # (n_tets, 3) FEM gradient
        gnorm = torch.sqrt((g ** 2).sum(-1) + 1e-12)
        return (((gnorm - 1.0) ** 2).mean()
                + W_SRC * u[self.src] ** 2
                - W_MAX * u.mean())


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
if ENCODING == 'eigen':
    V = torch.tensor(grid.eigenfunctions(N_EIG), dtype=torch.float32, device=DEVICE)
else:
    V = torch.tensor(verts, dtype=torch.float32, device=DEVICE)
model = FullyConnectedNetwork(V.shape[1], 1, [WIDTH] * LAYERS)
print(f'Encoding: {ENCODING}  ({V.shape[1]} features per node)')

val = MeshValidator(u_true, grid=grid, name='vs-geodesic', freq=VAL_FREQ,
                    root=RUN, dump_f=DUMP_F, cmap='hot',
                    inputs=V,
                    math_labels=(r'$d_{edge}(x,x_s)$', r'$\hat{u}$',
                                 r'$|\hat{u} - d_{edge}|$'))

trainer = Trainer(n_epochs=N_ITER, model=model, device=str(DEVICE),
                  adaptive=False, lr=LR, print_steps=2000, patience=None)
trainer.add_loss(EikonalLoss(grid, V, src_idx), weigth=1)
trainer.add_validator(val, freq=VAL_FREQ)
model, loss_dict = trainer.train()


# ------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------
model.eval()
with torch.no_grad():
    u_pred = model(V).squeeze(-1).cpu().numpy()

err = np.abs(u_pred - u_true)
print(f'\nvs edge geodesic: L2={np.sqrt(np.mean(err ** 2)):.3e}  max={err.max():.3e}')
print(f'field range:      {u_true.max():.4f}')
print(f'run directory:    {RUN}')


# ------------------------------------------------------------------
# Loss and validator evolution
# ------------------------------------------------------------------
val.plot_history(os.path.join(RUN, 'loss.png'),
                 extra={'train loss': loss_dict['eikonal']})

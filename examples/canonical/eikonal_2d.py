"""
2-D eikonal on the unit square.

    ||grad u|| = 1,   u = 0 at a source node in the corner

    u_hat = NN(x, y)
    loss  = mean( (||grad u_hat|| - 1)^2 )    eikonal residual
          + w_s * u_hat(x_s)^2                source
          - w_m * mean( u_hat )               maximality

The gradient is the FEM operator, so nothing differentiates through the
geometry. The equation with a single pinned value admits many solutions; the
distance function is the largest of them, which the maximality term selects.

The source sits at a corner, so the distance is monotone with no interior
medial axis and the exact solution on this convex domain is the Euclidean
distance. One validator measures pointwise error against it.

Output goes to output/eikonal_2d/.
"""

import os

import numpy as np
import torch

from fisiocomPinn import Grid, structured_mesh, FullyConnectedNetwork, GridLoss
from fisiocomPinn.Trainer import Trainer

from custom_validator import MeshValidator

# ------------------------------------------------------------------
N        = 41      # nodes per side
WIDTH    = 64
LAYERS   = 4
N_ITER   = 10000
LR       = 3e-4
W_SRC    = 100.0   # source condition u(x_s) = 0
W_MAX    = 0.1     # maximality: the distance is the LARGEST solution
VAL_FREQ = 500
DUMP_F   = 3
SOURCE   = (1.0, 0.0)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT      = os.path.dirname(os.path.abspath(__file__))
RUN      = os.path.join(OUT, 'output', 'eikonal_2d')


# ------------------------------------------------------------------
# Mesh, source, exact solution
# ------------------------------------------------------------------
verts, faces = structured_mesh(n=N, L=1.0)
grid = Grid(verts, faces)
print(f'Mesh: {grid.N} nodes, {len(faces)} triangles   device={DEVICE}')

src_idx = int(np.argmin(np.linalg.norm(verts - np.array(SOURCE), axis=1)))
u_true = np.linalg.norm(verts - verts[src_idx], axis=1)   # convex: geodesic = Euclidean


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------
class EikonalLoss(GridLoss):
    def __init__(self, grid, V, src_idx):
        super().__init__(grid, name='eikonal')
        self.V = V
        self.src = src_idx

    def loss(self, model):
        u = model(self.V).squeeze(-1)                 # (N,)
        g = self.grad(u)                              # (n_tri, 2) FEM gradient
        gnorm = torch.sqrt((g ** 2).sum(-1) + 1e-12)
        return (((gnorm - 1.0) ** 2).mean()
                + W_SRC * u[self.src] ** 2
                - W_MAX * u.mean())


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
V = torch.tensor(verts, dtype=torch.float32, device=DEVICE)
model = FullyConnectedNetwork(V.shape[1], 1, [WIDTH] * LAYERS)

val = MeshValidator(u_true, grid=grid, name='vs-exact', freq=VAL_FREQ,
                    root=RUN, dump_f=DUMP_F, cmap='hot',
                    inputs=V,
                    math_labels=(r'$d(x,x_s)$', r'$\hat{u}$',
                                 r'$|\hat{u} - d|$'))

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
print(f'\nvs geodesic:      L2={np.sqrt(np.mean(err ** 2)):.3e}  max={err.max():.3e}')
print(f'field range:      {u_true.max():.3f}')
print(f'run directory:    {RUN}')


# ------------------------------------------------------------------
# Loss and validator evolution
# ------------------------------------------------------------------
val.plot_history(os.path.join(RUN, 'loss.png'),
                 extra={'train loss': loss_dict['eikonal']})

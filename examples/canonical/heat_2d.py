"""
2-D steady heat (Poisson) on the unit square.

    -Delta u = f,   u = 0 on the boundary
    u* = sin(pi x) sin(pi y),   f = 2 pi^2 u*

    u_hat = NN(x, y)
    loss  = mean( (K u_hat / m - f)^2 )_interior + mu * mean( u_hat^2 )_boundary

K and M are the FEM stiffness and mass matrices and m the lumped nodal mass,
so the PDE is imposed through the operators and there is no training data.

Two validators are registered:

    vs-exact    pointwise error against u*
    residual    the PDE residual, which needs no reference solution

Output goes to output/heat_2d/.
"""

import os

import numpy as np
import torch

from fisiocomPinn import Grid, structured_mesh, FullyConnectedNetwork, GridLoss
from fisiocomPinn.Trainer import Trainer

from custom_validator import MeshValidator

# ------------------------------------------------------------------
N        = 33      # nodes per side
WIDTH    = 64
LAYERS   = 4
N_ITER   = 3000
LR       = 1e-3
MU       = 100.0   # boundary (Dirichlet) weight
VAL_FREQ = 200     # how often the Trainer calls the validators
DUMP_F   = 3       # ... and every DUMP_F checks, they write a field
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT      = os.path.dirname(os.path.abspath(__file__))
RUN      = os.path.join(OUT, 'output', 'heat_2d')


# ------------------------------------------------------------------
# Mesh, manufactured problem
# ------------------------------------------------------------------
verts, faces = structured_mesh(n=N, L=1.0)
grid = Grid(verts, faces)
print(f'Mesh: {grid.N} nodes, {len(faces)} triangles   device={DEVICE}')

x, y = verts[:, 0], verts[:, 1]
u_true = np.sin(np.pi * x) * np.sin(np.pi * y)
f = 2.0 * np.pi ** 2 * u_true                       # -Δu* = f

bmask = (np.isclose(x, 0) | np.isclose(x, 1) |
         np.isclose(y, 0) | np.isclose(y, 1))
bidx = np.where(bmask)[0]
iidx = np.where(~bmask)[0]


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------
class HeatLoss(GridLoss):
    def __init__(self, grid, X, f, iidx, bidx, device):
        super().__init__(grid, name='heat')
        self.X = X
        self.K_t = self.sparse(self.K, device)                # torch sparse stiffness
        m = np.asarray(self.M.sum(axis=1)).ravel()            # lumped nodal mass
        self.m = torch.tensor(m, dtype=torch.float32, device=device)
        self.f = torch.tensor(f, dtype=torch.float32, device=device)
        self.iidx = torch.tensor(iidx, dtype=torch.long, device=device)
        self.bidx = torch.tensor(bidx, dtype=torch.long, device=device)

    def loss(self, model):
        u = model(self.X).squeeze(-1)                         # (N,)
        Ku = torch.sparse.mm(self.K_t, u.unsqueeze(1)).squeeze(1)
        res = Ku / self.m - self.f                            # pointwise −Δu − f
        return (res[self.iidx] ** 2).mean() + MU * (u[self.bidx] ** 2).mean()


# ------------------------------------------------------------------
# A validator measuring the residual instead of an error. u_true is None:
# there is nothing to compare against.
# ------------------------------------------------------------------
class WeakResidualValidator(MeshValidator):
    def __init__(self, grid, f, interior, **kw):
        kw.setdefault('grid', grid)
        super().__init__(u_true=None, **kw)
        self.K = grid.K
        self.m = np.asarray(grid.M.sum(axis=1)).ravel()
        self.f = np.asarray(f).ravel()
        self.interior = np.asarray(interior)
        self._fscale = float(np.sqrt(np.mean(self.f[self.interior] ** 2))) or 1.0

    def _residual(self, u):
        return (self.K @ u) / self.m - self.f      # same convention as the loss

    def measure(self, u):
        # normalised by ||f||, so the number reads as a fraction
        r = self._residual(u)[self.interior]
        return {'rel_res': float(np.sqrt(np.mean(r ** 2)) / self._fscale)}

    def arrays(self, u):
        return {'pred': u, 'residual': self._residual(u)}

    def panels(self, u):
        r = self._residual(u)
        m = float(np.abs(r).max()) or 1.0
        return [(r, r'$K\hat{u}/m - f$', self.cmap, -m, m)]


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
X = torch.tensor(verts, dtype=torch.float32, device=DEVICE)   # (N, 2)
model = FullyConnectedNetwork(2, 1, [WIDTH] * LAYERS)

MATH = (r'$u^*$', r'$\hat{u}$', r'$|\hat{u}-u^*|$')
val_exact = MeshValidator(u_true, grid=grid, name='vs-exact', freq=VAL_FREQ,
                          root=RUN, dump_f=DUMP_F, math_labels=MATH,
                          inputs=X)
val_res = WeakResidualValidator(grid, f, iidx, name='residual',
                                freq=VAL_FREQ, root=RUN, dump_f=DUMP_F,
                                cmap='coolwarm', inputs=X)

trainer = Trainer(n_epochs=N_ITER, model=model, device=str(DEVICE),
                  adaptive=False, lr=LR, print_steps=1000, patience=None)
trainer.add_loss(HeatLoss(grid, X, f, iidx, bidx, DEVICE), weigth=1)
for v in (val_exact, val_res):
    trainer.add_validator(v, freq=VAL_FREQ)
model, loss_dict = trainer.train()


# ------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------
model.eval()
with torch.no_grad():
    u_pred = model(X).squeeze(-1).cpu().numpy()

err = np.abs(u_pred - u_true)
print(f'\nvs manufactured:  L2={np.sqrt(np.mean(err ** 2)):.3e}  max={err.max():.3e}')
print(f'run directory:    {RUN}')


# ------------------------------------------------------------------
# Loss and validator evolution
# ------------------------------------------------------------------
val_exact.plot_history(os.path.join(RUN, 'loss.png'), others=(val_res,),
                       extra={'train loss': loss_dict['heat']})

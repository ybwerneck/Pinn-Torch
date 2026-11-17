"""
heat_2d.py with an ensemble: M networks from M seeds, trained as one model.

Problem, loss and validators come from heat_2d.py unchanged. The ensemble
wraps them: EnsembleLoss sums the member losses, and EnsembleStats (an
EnsembleValidator that changes only gather) prints median / min / max of each
metric over the members. The single validators run with return_mode=True, so
they only measure (no files or plots per member).

Output goes to output/heat_2d_ensemble/.
"""

import os

import numpy as np
import torch

from fisiocomPinn import (EnsembleLoss, EnsembleNet, EnsembleValidator,
                          FullyConnectedNetwork)
from fisiocomPinn.Trainer import Trainer

from custom_validator import MeshValidator
from heat_2d import (DEVICE, LAYERS, LR, N_ITER, OUT, VAL_FREQ, WIDTH, X,
                     HeatLoss, WeakResidualValidator, bidx, f, grid, iidx,
                     u_true)

M   = 16
RUN = os.path.join(OUT, 'output', 'heat_2d_ensemble')


class EnsembleStats(EnsembleValidator):
    def gather(self, outs):
        for k, v in outs.items():
            print(f'  [{self.name}]  {k}: median {float(v.median()):.3e}'
                  f'  min {float(v.min()):.3e}  max {float(v.max()):.3e}')
        return outs


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
members = []
for s in range(M):
    torch.manual_seed(s)
    members.append(FullyConnectedNetwork(2, 1, [WIDTH] * LAYERS).to(DEVICE))
model = EnsembleNet(members)

val_exact = MeshValidator(u_true, grid=grid, name='vs-exact', freq=VAL_FREQ,
                          root=RUN, inputs=X, return_mode=True)
val_res = WeakResidualValidator(grid, f, iidx, name='residual', freq=VAL_FREQ,
                                root=RUN, inputs=X, return_mode=True)

trainer = Trainer(n_epochs=N_ITER, model=model, device=str(DEVICE),
                  adaptive=False, lr=LR, print_steps=1000, patience=None)
trainer.add_loss(EnsembleLoss(HeatLoss(grid, X, f, iidx, bidx, DEVICE)), weigth=1)
for v in (val_exact, val_res):
    trainer.add_validator(EnsembleStats(v), freq=VAL_FREQ)
model, loss_dict = trainer.train()


# ------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------
with torch.no_grad():
    U = model(X).squeeze(-1).cpu().numpy()                 # (M, N)

l2 = np.sqrt(np.mean((U - u_true) ** 2, axis=1))
best = int(np.argmin(l2))
print(f'\nL2 vs u*: median {np.median(l2):.3e}, best {l2[best]:.3e} (member {best})')
val_exact.plot_field(U[best], os.path.join(RUN, 'best_member.png'),
                     title=f'best member {best}')

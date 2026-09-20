"""
Trainer-compatible base class for FEM-operator losses.

Holds a Grid or Grid3D and exposes its operators — stiffness K, mass M, the
element gradient, and a prefactorised differentiable solve — so a subclass
writes only the physics.

Subclass and override ``loss(self, model) -> scalar``. For a custom operator,
assemble A from ``self.K`` and ``self.M`` and prefactorise once::

    self.solve_A = self.factorize(self.K + c * self.M)   # in __init__

    u = model(x)                                         # in loss()
    r = self.solve_A(b) - target
    return (r ** 2).mean()
"""

import torch


class GridLoss:
    def __init__(self, grid, name='GridLoss'):
        self.grid = grid
        self.name = name

    # --- operator conveniences ------------------------------------------------
    @property
    def K(self):
        """Stiffness (Laplacian) — scipy sparse."""
        return self.grid.K

    @property
    def M(self):
        """Mass — scipy sparse."""
        return self.grid.M

    def grad(self, f):
        """Discrete gradient of a nodal field (piecewise-constant per element)."""
        return self.grid.gradient(f)

    def factorize(self, A, symmetric=True, neumann=False):
        """Prefactorise an operator A -> differentiable ``solve(b) -> x``."""
        return self.grid.factorize(A, symmetric=symmetric, neumann=neumann)

    def sparse(self, A, device='cpu', dtype=torch.float32):
        """scipy sparse operator -> torch sparse tensor (for differentiable mat-vec)."""
        return self.grid.to_torch_sparse(A, device=device, dtype=dtype)

    # --- Trainer contract -----------------------------------------------------
    def forward(self, model):
        return self.loss(model)

    def loss(self, model):
        raise NotImplementedError("GridLoss subclasses must implement loss(model).")

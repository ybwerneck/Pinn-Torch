"""
FEM meshes and operators.

    Grid    triangular mesh, in the plane or on a surface in 3-D
    Grid3D  tetrahedral volume mesh

Both assemble a stiffness matrix K and a lumped mass matrix M, and expose
Laplace-Beltrami eigenfunctions, a piecewise-constant element gradient, its
weak-form divergence, and a prefactorised differentiable linear solve.

structured_mesh and annular_mesh build simple 2-D meshes.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


class Grid:
    """
    FEM triangular mesh with Laplace-Beltrami operators and a differentiable
    Poisson solver.

    Parameters
    ----------
    vertices : (N, 2) or (N, 3) numpy array — node coordinates
    faces    : (F, 3) int numpy array       — triangle connectivity, 0-indexed
    """

    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces    = np.asarray(faces,    dtype=np.int32)
        self.N        = len(self.vertices)
        self.dim      = self.vertices.shape[1]

        self._areas, self._grads = self._triangle_geometry()
        self.K, self.M           = self._assemble()

    # ------------------------------------------------------------------
    # Private: geometry and assembly
    # ------------------------------------------------------------------

    def _triangle_geometry(self):
        """
        Per-triangle unsigned areas (F,) and basis-function gradients (F, 3, 2).
        Gradients are piecewise constant on each triangle.
        Supports 2-D planar meshes; 3-D surface support is noted where it differs.
        """
        v, f = self.vertices, self.faces
        p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        e01 = p1 - p0
        e02 = p2 - p0

        if self.dim == 2:
            # Signed 2 * area from 2-D cross product
            cross2 = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]  # (F,)
            areas  = np.abs(cross2) / 2.0

            # Linear basis-function gradients (exact formula for linear elements):
            #   ∇ψ_0 = (y1−y2, x2−x1) / (2A_signed)
            #   ∇ψ_1 = (y2−y0, x0−x2) / (2A_signed)
            #   ∇ψ_2 = (y0−y1, x1−x0) / (2A_signed)
            grads = np.zeros((len(f), 3, 2))
            grads[:, 0, 0] = (p1[:, 1] - p2[:, 1]) / cross2
            grads[:, 0, 1] = (p2[:, 0] - p1[:, 0]) / cross2
            grads[:, 1, 0] = (p2[:, 1] - p0[:, 1]) / cross2
            grads[:, 1, 1] = (p0[:, 0] - p2[:, 0]) / cross2
            grads[:, 2, 0] = (p0[:, 1] - p1[:, 1]) / cross2
            grads[:, 2, 1] = (p1[:, 0] - p0[:, 0]) / cross2

        elif self.dim == 3:
            cross_vec = np.cross(e01, e02)                          # (F, 3)
            areas     = np.linalg.norm(cross_vec, axis=1) / 2.0    # (F,)

            # Local 2-D tangent frame per triangle
            e1    = e01 / np.linalg.norm(e01, axis=1, keepdims=True)      # (F, 3)
            n_hat = cross_vec / (2.0 * areas[:, None])                    # (F, 3) unit normal
            e2    = np.cross(n_hat, e1)                                    # (F, 3)

            # Local 2-D coords of p2 in the (e1, e2) frame
            u1 = np.linalg.norm(e01, axis=1)                       # (F,)
            u2 = np.einsum('fd,fd->f', e02, e1)                    # (F,)
            v2 = np.einsum('fd,fd->f', e02, e2)                    # (F,)
            area2 = u1 * v2                                         # (F,) = 2A in local frame

            # 2-D basis-function gradients projected back to 3-D via tangent frame
            # ∇ψ_0: (-v2, u2-u1) / area2  in (e1,e2)
            # ∇ψ_1: ( v2,  -u2 ) / area2
            # ∇ψ_2: (  0,   u1 ) / area2
            grads = np.zeros((len(f), 3, 3))
            grads[:, 0, :] = ((-v2         ) / area2)[:, None] * e1 \
                           + ((u2 - u1     ) / area2)[:, None] * e2
            grads[:, 1, :] = (( v2         ) / area2)[:, None] * e1 \
                           + ((-u2         ) / area2)[:, None] * e2
            grads[:, 2, :] = (( u1         ) / area2)[:, None] * e2
        else:
            raise ValueError(f"Expected dim 2 or 3, got {self.dim}")

        return areas, grads

    def _assemble(self):
        """
        Cotangent stiffness matrix K (Laplace-Beltrami) and lumped mass matrix M.
        Both are scipy CSR sparse matrices of shape (N, N).
        """
        v, f   = self.vertices, self.faces
        p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]

        e01 = p1 - p0
        e02 = p2 - p0
        e12 = p2 - p1

        # Unsigned 2 * area
        if self.dim == 2:
            area2 = np.abs(e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0])
        else:
            area2 = np.linalg.norm(np.cross(e01, e02), axis=1)

        # Cotangents at each vertex (cot_i is opposite to edge not touching vertex i)
        cot0 = np.einsum('fd,fd->f',  e01,  e02) / area2  # at v0 → edge (f1,f2)
        cot1 = np.einsum('fd,fd->f', -e01,  e12) / area2  # at v1 → edge (f0,f2)
        cot2 = np.einsum('fd,fd->f',  e02,  e12) / area2  # at v2 → edge (f0,f1)

        # Off-diagonal stiffness: K[a,b] += −½ cot(vertex opposite to edge a-b)
        rows = np.concatenate([f1, f2, f0, f2, f0, f1])
        cols = np.concatenate([f2, f1, f2, f0, f1, f0])
        vals = np.concatenate([-0.5*cot0, -0.5*cot0,
                               -0.5*cot1, -0.5*cot1,
                               -0.5*cot2, -0.5*cot2])
        K    = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        diag = -np.asarray(K.sum(axis=1)).ravel()
        K   += sp.diags(diag)

        # Lumped mass: each vertex accumulates area/3 per adjacent triangle
        m_rows = np.concatenate([f0, f1, f2])
        m_vals = np.tile(self._areas / 3.0, 3)
        mass   = np.zeros(self.N)
        np.add.at(mass, m_rows, m_vals)
        M = sp.diags(mass)

        return K.tocsr(), M.tocsr()

    @staticmethod
    def _pin(K):
        """
        Symmetrically pin node 0 to fix the Neumann Poisson gauge:
        set row 0 and col 0 to the identity row/column.
        This preserves symmetry so the same LU handles both forward and adjoint.
        """
        K = K.tolil()
        K[0, :] = 0
        K[:, 0] = 0
        K[0, 0] = 1
        return K.tocsr()

    # ------------------------------------------------------------------
    # Public: operators
    # ------------------------------------------------------------------

    def eigenfunctions(self, n_eig):
        """
        Laplace-Beltrami eigenfunctions on the mesh.
        Solves the generalised eigenvalue problem  K v = λ M v.

        Parameters
        ----------
        n_eig : int — number of eigenfunctions (smallest eigenvalues first)

        Returns
        -------
        vecs : (N, n_eig) numpy array, columns are eigenfunctions sorted by λ
        """
        vals, vecs = spla.eigsh(self.K, k=n_eig, M=self.M, sigma=1e-8, which='LM')
        order = np.argsort(vals)
        return vecs[:, order]

    def gradient(self, f):
        """
        Piecewise constant gradient of a scalar field.

        Parameters
        ----------
        f : (N,) torch tensor or numpy array of nodal values

        Returns
        -------
        grad_f : (F, 2) gradient per triangle, same type as input
        """
        use_torch = isinstance(f, torch.Tensor)
        grads_t   = torch.tensor(self._grads, dtype=f.dtype, device=f.device) \
                    if use_torch else self._grads
        f0v = f[self.faces[:, 0] if not use_torch else
                torch.tensor(self.faces[:, 0], device=f.device)]
        f1v = f[self.faces[:, 1] if not use_torch else
                torch.tensor(self.faces[:, 1], device=f.device)]
        f2v = f[self.faces[:, 2] if not use_torch else
                torch.tensor(self.faces[:, 2], device=f.device)]

        if use_torch:
            return (f0v[:, None] * grads_t[:, 0, :]
                  + f1v[:, None] * grads_t[:, 1, :]
                  + f2v[:, None] * grads_t[:, 2, :])
        else:
            f0v, f1v, f2v = f[self.faces[:, 0]], f[self.faces[:, 1]], f[self.faces[:, 2]]
            return (f0v[:, None] * self._grads[:, 0, :]
                  + f1v[:, None] * self._grads[:, 1, :]
                  + f2v[:, None] * self._grads[:, 2, :])

    def divergence(self, q):
        """
        Weak-form divergence of a nodal vector field: nodal vectors -> nodal scalar.

        q is averaged onto elements first, then b_i = Σ_T A_T (q_T·∇ψ_i). With A
        that node-to-element average, the exact identity is

            ⟨grad u, A q⟩_A = ⟨u, div q⟩

        so this is the adjoint of ``gradient`` composed with A, not of ``gradient``
        alone. Used wherever the source is the divergence of a flux.

        q : (N, 2) torch tensor (differentiable) or numpy array
        """
        f = self.faces
        if isinstance(q, torch.Tensor):
            dev  = q.device
            f0t  = torch.tensor(f[:, 0], dtype=torch.long, device=dev)
            f1t  = torch.tensor(f[:, 1], dtype=torch.long, device=dev)
            f2t  = torch.tensor(f[:, 2], dtype=torch.long, device=dev)
            At   = torch.tensor(self._areas, dtype=q.dtype, device=dev)
            Gt   = torch.tensor(self._grads, dtype=q.dtype, device=dev)
            q_T  = (q[f0t] + q[f1t] + q[f2t]) / 3.0
            dots = torch.einsum('fd,fld->fl', q_T, Gt)
            contrib = At[:, None] * dots
            b = torch.zeros(self.N, dtype=q.dtype, device=dev)
            b.scatter_add_(0, f0t, contrib[:, 0])
            b.scatter_add_(0, f1t, contrib[:, 1])
            b.scatter_add_(0, f2t, contrib[:, 2])
            return b
        else:
            q   = np.asarray(q, dtype=np.float64)
            q_T = (q[f[:, 0]] + q[f[:, 1]] + q[f[:, 2]]) / 3.0
            dots    = np.einsum('fd,fld->fl', q_T, self._grads)
            contrib = self._areas[:, None] * dots
            b = np.zeros(self.N)
            np.add.at(b, f[:, 0], contrib[:, 0])
            np.add.at(b, f[:, 1], contrib[:, 1])
            np.add.at(b, f[:, 2], contrib[:, 2])
            return b

    def factorize(self, A, symmetric=True, neumann=False):
        """
        Prefactorise a sparse operator A and return a differentiable solver
        ``solve(b) -> x`` for  A x = b  (canonical FEM primitive).

        Assemble A from the primitives (e.g. K, K + c·M, K − k²·M, an anisotropic
        stiffness) and call this once. The returned closure backprops through the
        solve via the adjoint (A^{-T}; A^{-1} when symmetric), and accepts torch
        (differentiable) or numpy input.

        Parameters
        ----------
        A         : scipy sparse operator (N, N)
        symmetric : if False, also factorise A^T for the adjoint
        neumann   : if True, apply the Neumann-nullspace gauge (pin node 0,
                    project b, shift min → 0) for the singular pure-Laplacian
        """
        Acsc = A.tocsc()
        if neumann:
            Acsc = self._pin(Acsc.tolil()).tocsc()
        lu   = spla.factorized(Acsc)
        lu_T = None if symmetric else spla.factorized(Acsc.T.tocsc())

        def solve(b):
            if isinstance(b, torch.Tensor):
                return _LinearSolve.apply(b, lu, lu_T, neumann)
            b = np.asarray(b, dtype=np.float64).copy()
            if neumann:
                b -= b.mean(); b[0] = 0.0
            x = lu(b)
            if neumann:
                x -= x.min()
            return x
        return solve

    def to_torch_sparse(self, A, device='cpu', dtype=torch.float32):
        """Convert a scipy sparse operator (e.g. self.K, self.M) to a coalesced
        torch sparse tensor, for differentiable mat-vecs ``torch.sparse.mm``."""
        A = A.tocoo()
        idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
        val = torch.tensor(A.data, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(idx, val, A.shape, device=device).coalesce()


# ------------------------------------------------------------------
# Differentiable linear solve (custom autograd)
# ------------------------------------------------------------------

class _LinearSolve(torch.autograd.Function):
    """
    Differentiable solve  A x = b  for a prefactorised operator A.

    Parameters passed via apply(b, lu, lu_T, neumann):
      lu      : callable applying A^{-1}       (scipy factorised solve)
      lu_T    : callable applying A^{-T}, or None if A is symmetric (then lu is reused)
      neumann : if True, apply the Neumann-nullspace gauge — project b (mean-subtract,
                pin node 0), and shift the solution so min = 0. This is the pure-
                Laplacian case; leave False for well-posed operators (K+cM, Dirichlet).

    The adjoint of A x = b is A^{-T} (grad_x); for symmetric A that is A^{-1}.
    """

    @staticmethod
    def forward(ctx, b, lu, lu_T, neumann):
        b_np = b.detach().cpu().numpy().copy()
        if neumann:
            b_np -= b_np.mean()
            b_np[0] = 0.0
        x_np = lu(b_np)
        if neumann:
            x_np -= x_np.min()
        ctx.lu_T    = lu_T if lu_T is not None else lu
        ctx.neumann = neumann
        ctx.dtype   = b.dtype
        return torch.tensor(x_np, dtype=b.dtype, device=b.device)

    @staticmethod
    def backward(ctx, grad_x):
        g = grad_x.detach().cpu().numpy().copy()
        if ctx.neumann:
            g[0] = 0.0          # adjoint of node-0 pinning
            g -= g.mean()       # adjoint of mean subtraction (solvability projection)
        grad_b = ctx.lu_T(g)
        return torch.tensor(grad_b, dtype=ctx.dtype, device=grad_x.device), None, None, None


# ------------------------------------------------------------------
# Mesh utilities
# ------------------------------------------------------------------

def structured_mesh(n, L=1.0):
    """
    Uniform triangular mesh on [0, L]² with n nodes per side.
    Each square cell is split into two triangles (lower-left and upper-right).

    Parameters
    ----------
    n : int  — grid points per side
    L : float — domain side length

    Returns
    -------
    vertices : (n*n, 2) numpy array
    faces    : (2*(n-1)², 3) int numpy array
    """
    x = np.linspace(0.0, L, n)
    y = np.linspace(0.0, L, n)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    vertices = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v00 = i * n + j
            v10 = (i + 1) * n + j
            v01 = i * n + (j + 1)
            v11 = (i + 1) * n + (j + 1)
            faces.append([v00, v10, v11])   # lower-right triangle
            faces.append([v00, v11, v01])   # upper-left triangle

    return vertices, np.array(faces, dtype=np.int32)


def annular_mesh(n_r, n_theta, inner_r, outer_r, center=(0.0, 0.0)):
    """
    Structured triangular mesh on an annular (ring-shaped) domain.

    Nodes are placed on n_r concentric circles with n_theta equally-spaced
    angles each.  Each quad cell is split into two triangles; the last
    angular cell wraps back to theta=0 so the ring is closed.

    Parameters
    ----------
    n_r     : int   — number of radial layers (≥ 2)
    n_theta : int   — number of nodes per ring
    inner_r : float — inner radius
    outer_r : float — outer radius
    center  : (2,) — centre of the annulus

    Returns
    -------
    vertices : (n_r * n_theta, 2) numpy array
    faces    : (2 * (n_r-1) * n_theta, 3) int numpy array
    """
    cx, cy = center
    radii  = np.linspace(inner_r, outer_r, n_r)
    theta  = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    # vertices: ring 0 first (inner), ring n_r-1 last (outer)
    verts = np.array([
        [cx + r * np.cos(t), cy + r * np.sin(t)]
        for r in radii
        for t in theta
    ], dtype=np.float64)

    # faces: each quad (i, j) → (i+1, j) split into 2 triangles
    # theta wraps: j+1 taken mod n_theta
    faces = []
    for i in range(n_r - 1):
        for j in range(n_theta):
            j1  = (j + 1) % n_theta
            v00 = i       * n_theta + j
            v10 = (i + 1) * n_theta + j
            v01 = i       * n_theta + j1
            v11 = (i + 1) * n_theta + j1
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    return verts, np.array(faces, dtype=np.int32)


# ------------------------------------------------------------------
# 3-D tetrahedral mesh
# ------------------------------------------------------------------

class Grid3D:
    """
    FEM tetrahedral volume mesh with a 3-D Laplacian and differentiable Poisson solver.

    Parameters
    ----------
    vertices : (N, 3) numpy array — node coordinates
    tets     : (T, 4) int numpy array — tetrahedral connectivity, 0-indexed
    """

    def __init__(self, vertices, tets):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.tets     = np.asarray(tets,     dtype=np.int32)
        self.N        = len(self.vertices)
        self.n_tets   = len(self.tets)

        self._volumes, self._grads = self._tet_geometry()
        self.K, self.M             = self._assemble()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _tet_geometry(self):
        """
        Per-tet volumes (n_tets,) and basis-function gradients (n_tets, 4, 3).

        For a linear tet with vertices p0-p3:
          Jacobian B = [p1-p0 | p2-p0 | p3-p0]  (3×3, columns = edge vectors)
          Volume = |det B| / 6
          ∇ψ_i = B^{-T} e_i  (for i=1,2,3);  ∇ψ_0 = −(∇ψ_1+∇ψ_2+∇ψ_3)
        """
        v, t = self.vertices, self.tets
        p0, p1, p2, p3 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]], v[t[:, 3]]

        B = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=2)   # (T, 3, 3) – columns = edges
        dets    = np.linalg.det(B)                             # (T,)
        volumes = np.abs(dets) / 6.0

        # B^{-T} @ e_i  =  row i of B^{-1}
        Binv = np.linalg.inv(B)                                # (T, 3, 3)

        grads = np.zeros((self.n_tets, 4, 3))
        grads[:, 1, :] = Binv[:, 0, :]                        # ∇ψ_1
        grads[:, 2, :] = Binv[:, 1, :]                        # ∇ψ_2
        grads[:, 3, :] = Binv[:, 2, :]                        # ∇ψ_3
        grads[:, 0, :] = -Binv.sum(axis=1)                    # ∇ψ_0

        return volumes, grads

    def _assemble(self):
        """
        Stiffness K (Laplacian) and lumped mass M.
        K[a,b] = Σ_T  V_T (∇ψ_a^T · ∇ψ_b^T)
        M[i]   = Σ_T  V_T / 4   for each tet T touching node i
        """
        t = self.tets

        # Element stiffness: K_T = V_T * grads_T grads_T^T   (T, 4, 4)
        K_local = self._volumes[:, None, None] * np.einsum(
            'tid,tjd->tij', self._grads, self._grads)

        rows, cols, vals = [], [], []
        for a in range(4):
            for b in range(4):
                rows.append(t[:, a])
                cols.append(t[:, b])
                vals.append(K_local[:, a, b])

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        K    = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))

        mass = np.zeros(self.N)
        for i in range(4):
            np.add.at(mass, t[:, i], self._volumes / 4.0)
        M = sp.diags(mass)

        return K.tocsr(), M.tocsr()

    @staticmethod
    def _pin(K):
        K = K.tolil()
        K[0, :] = 0
        K[:, 0] = 0
        K[0, 0] = 1
        return K.tocsr()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def eigenfunctions(self, n_eig):
        """Laplacian eigenfunctions: K v = λ M v."""
        vals, vecs = spla.eigsh(self.K, k=n_eig, M=self.M, sigma=1e-8, which='LM')
        return vecs[:, np.argsort(vals)]

    def gradient(self, phi):
        """
        Piecewise-constant 3-D gradient per tet: ∇φ_T = Σ_i φ_i ∇ψ_i.

        Parameters
        ----------
        phi : (N,) torch tensor or numpy array

        Returns
        -------
        grad_phi : (n_tets, 3)
        """
        t = self.tets
        if isinstance(phi, torch.Tensor):
            dev = phi.device
            tt  = torch.tensor(t, dtype=torch.long, device=dev)
            Gt  = torch.tensor(self._grads, dtype=phi.dtype, device=dev)
            phi_nodes = torch.stack([phi[tt[:, i]] for i in range(4)], dim=1)  # (T, 4)
            return torch.einsum('ti,tid->td', phi_nodes, Gt)                   # (T, 3)
        else:
            phi_nodes = phi[t]                                                  # (T, 4)
            return np.einsum('ti,tid->td', phi_nodes, self._grads)             # (T, 3)

    def divergence(self, q):
        """
        Weak-form divergence of a nodal vector field: nodal vectors -> nodal scalar.

        q is averaged onto elements first, then b_i = Σ_T V_T (q_T·∇ψ_i). With A
        that node-to-element average, the exact identity is
        ⟨grad u, A q⟩_V = ⟨u, div q⟩ — the adjoint of ``gradient`` composed with A,
        not of ``gradient`` alone.

        q : (N, 3) torch tensor or numpy array
        """
        t = self.tets
        if isinstance(q, torch.Tensor):
            dev  = q.device
            tt   = torch.tensor(t, dtype=torch.long, device=dev)
            Vt   = torch.tensor(self._volumes, dtype=q.dtype, device=dev)
            Gt   = torch.tensor(self._grads,   dtype=q.dtype, device=dev)
            q_T  = torch.stack([q[tt[:, i]] for i in range(4)], dim=1).mean(dim=1)
            dots = torch.einsum('td,tid->ti', q_T, Gt)
            contrib = Vt[:, None] * dots
            b = torch.zeros(self.N, dtype=q.dtype, device=dev)
            for i in range(4):
                b.scatter_add_(0, tt[:, i], contrib[:, i])
            return b
        else:
            q     = np.asarray(q, dtype=np.float64)
            q_T   = q[t].mean(axis=1)
            dots  = np.einsum('td,tid->ti', q_T, self._grads)
            contrib = self._volumes[:, None] * dots
            b = np.zeros(self.N)
            for i in range(4):
                np.add.at(b, t[:, i], contrib[:, i])
            return b

    def factorize(self, A, symmetric=True, neumann=False):
        """Prefactorise a sparse operator A -> differentiable ``solve(b) -> x``
        for A x = b (canonical FEM primitive). See Grid.factorize for details."""
        Acsc = A.tocsc()
        if neumann:
            Acsc = self._pin(Acsc.tolil()).tocsc()
        lu   = spla.factorized(Acsc)
        lu_T = None if symmetric else spla.factorized(Acsc.T.tocsc())

        def solve(b):
            if isinstance(b, torch.Tensor):
                return _LinearSolve.apply(b, lu, lu_T, neumann)
            b = np.asarray(b, dtype=np.float64).copy()
            if neumann:
                b -= b.mean(); b[0] = 0.0
            x = lu(b)
            if neumann:
                x -= x.min()
            return x
        return solve

    def to_torch_sparse(self, A, device='cpu', dtype=torch.float32):
        """scipy sparse (e.g. self.K, self.M) -> coalesced torch sparse tensor."""
        A = A.tocoo()
        idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
        val = torch.tensor(A.data, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(idx, val, A.shape, device=device).coalesce()

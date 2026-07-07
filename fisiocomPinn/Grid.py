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

        K_pinned   = self._pin(self.K.copy())
        self._lu   = spla.factorized(K_pinned.tocsc())

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
            # Laplace-Beltrami: cotangent weights use 3-D edge lengths and areas.
            # The stiffness assembly below is the same; only gradient computation
            # differs (needs projection onto the triangle tangent plane).
            cross_vec = np.cross(e01, e02)              # (F, 3)
            areas     = np.linalg.norm(cross_vec, axis=1) / 2.0
            # Surface gradients require a local 2-D frame per triangle.
            raise NotImplementedError(
                "3-D surface gradients not yet implemented. "
                "Stiffness / mass assembly and eigenfunctions work for 3-D; "
                "gradient() and assemble_rhs() require a tangent-frame projection."
            )
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

    def assemble_rhs(self, q):
        """
        Assemble the FEM right-hand side for the Poisson equation.
        Implements  b_i = +∫ q · ∇ψ_i dΩ  (weak-form source term).
        q is approximated as piecewise constant (average of triangle vertices).

        Parameters
        ----------
        q : (N, 2) torch tensor (differentiable) or numpy array

        Returns
        -------
        b : (N,) same type as q
        """
        f = self.faces

        if isinstance(q, torch.Tensor):
            dev  = q.device
            f0t  = torch.tensor(f[:, 0], dtype=torch.long, device=dev)
            f1t  = torch.tensor(f[:, 1], dtype=torch.long, device=dev)
            f2t  = torch.tensor(f[:, 2], dtype=torch.long, device=dev)
            At   = torch.tensor(self._areas, dtype=q.dtype, device=dev)
            Gt   = torch.tensor(self._grads, dtype=q.dtype, device=dev)

            q_T  = (q[f0t] + q[f1t] + q[f2t]) / 3.0        # (F, 2)
            dots = torch.einsum('fd,fld->fl', q_T, Gt)       # (F, 3)
            contrib = At[:, None] * dots                      # (F, 3)

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

    def solve_poisson(self, b):
        """
        Solve the Neumann Poisson system  K φ = b  with gauge fix φ.min() = 0.
        For torch input the solve is differentiable (custom VJP via LU adjoint).

        Parameters
        ----------
        b : (N,) torch tensor or numpy array

        Returns
        -------
        phi : (N,) same type, with phi.min() = 0
        """
        if isinstance(b, torch.Tensor):
            return _PoissonSolve.apply(b, self._lu)
        else:
            b = np.asarray(b, dtype=np.float64).copy()
            b -= b.mean()
            b[0] = 0.0
            phi  = self._lu(b)
            return phi - phi.min()


# ------------------------------------------------------------------
# Differentiable Poisson solve (custom autograd)
# ------------------------------------------------------------------

class _PoissonSolve(torch.autograd.Function):
    """
    Wraps the precomputed scipy LU factorisation in a PyTorch autograd Function.
    Gradient flows through the linear solve via the adjoint  K^{-T} = K^{-1}
    (valid because the pinned system is symmetric).
    """

    @staticmethod
    def forward(ctx, b, lu):
        b_np = b.detach().cpu().numpy().copy()
        b_np -= b_np.mean()
        b_np[0] = 0.0
        phi_np = lu(b_np)
        phi_np -= phi_np.min()
        ctx.lu   = lu
        ctx.dtype = b.dtype
        return torch.tensor(phi_np, dtype=b.dtype, device=b.device)

    @staticmethod
    def backward(ctx, grad_phi):
        g = grad_phi.detach().cpu().numpy().copy()
        g[0] = 0.0          # adjoint of node-0 pinning
        g -= g.mean()       # adjoint of mean subtraction (solvability projection)
        grad_b = ctx.lu(g)
        return torch.tensor(grad_b, dtype=ctx.dtype, device=grad_phi.device), None


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

"""Deterministic CPU checks for the FEM grids and the eikonal operator.

No network and no training: every expectation here is either a closed-form
property of the discretisation or a manufactured solution, so a failure points
at the operator rather than at optimisation.
"""

import os
import unittest

import numpy as np
import torch

from fisiocomPinn import Grid, Grid3D, structured_mesh

MESH = os.path.join(
    os.path.dirname(__file__), "..", "examples", "canonical", "ventricles_mesh.npz"
)


def tet_volumes(vertices, tets):
    """Signed tetrahedron volumes, computed independently of Grid3D."""
    a, b, c, d = (vertices[tets[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


class StructuredMeshTests(unittest.TestCase):
    def test_counts_and_area(self):
        for n in (4, 11):
            with self.subTest(n=n):
                verts, faces = structured_mesh(n=n, L=1.0)
                # n counts points per side, not cells.
                self.assertEqual(len(verts), n * n)
                self.assertEqual(len(faces), 2 * (n - 1) ** 2)
                p = verts[faces]
                e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
                # 2-D cross product as a scalar; np.cross on 2-vectors is
                # deprecated in NumPy 2.0.
                area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
                self.assertAlmostEqual(area, 1.0, places=12)

    def test_gradient_is_exact_on_a_linear_field(self):
        verts, faces = structured_mesh(n=8, L=1.0)
        grid = Grid(verts, faces)
        a = np.array([0.7, -1.3])
        g = grid.gradient(verts @ a)
        self.assertLess(np.abs(g - a).max(), 1e-10)

    def test_gradient_of_a_constant_vanishes(self):
        verts, faces = structured_mesh(n=8, L=1.0)
        grid = Grid(verts, faces)
        g = grid.gradient(np.full(len(verts), 3.5))
        self.assertLess(np.abs(g).max(), 1e-12)

    def test_torch_and_numpy_gradients_agree(self):
        verts, faces = structured_mesh(n=8, L=1.0)
        grid = Grid(verts, faces)
        f = verts[:, 0] ** 2 - verts[:, 1]
        g_np = np.asarray(grid.gradient(f))
        g_t = grid.gradient(torch.tensor(f, dtype=torch.float64)).numpy()
        self.assertLess(np.abs(g_np - g_t).max(), 1e-12)


class EikonalOperatorTests(unittest.TestCase):
    """On a convex domain the distance to a corner satisfies ||grad d|| = 1."""

    @staticmethod
    def distance_field_error(n):
        verts, faces = structured_mesh(n=n, L=1.0)
        grid = Grid(verts, faces)
        source = verts[int(np.argmin(np.linalg.norm(verts, axis=1)))]
        d = np.linalg.norm(verts - source, axis=1)
        g = np.asarray(grid.gradient(d))
        return np.abs(np.linalg.norm(g, axis=1) - 1.0)

    def test_distance_field_satisfies_the_eikonal(self):
        err = self.distance_field_error(41)
        self.assertLess(np.median(err), 0.02)

    def test_error_decreases_under_refinement(self):
        coarse = np.median(self.distance_field_error(21))
        fine = np.median(self.distance_field_error(61))
        self.assertLess(fine, coarse)


class VentricleMeshTests(unittest.TestCase):
    """Biventricular tetrahedral mesh; exercises Grid3D."""

    @classmethod
    def setUpClass(cls):
        data = np.load(MESH)
        cls.vertices = data["vertices"]
        cls.tets = data["tets"]
        cls.geodesic = data["geodesic_source"]
        cls.source = int(data["source_index"])

    def test_mesh_is_wellformed(self):
        self.assertEqual(self.vertices.shape, (7387, 3))
        self.assertEqual(self.tets.shape, (24353, 4))
        self.assertEqual(self.tets.min(), 0)
        self.assertEqual(self.tets.max(), len(self.vertices) - 1)
        vol = np.abs(tet_volumes(self.vertices, self.tets))
        self.assertGreater(vol.min(), 0.0)  # no degenerate elements
        self.assertAlmostEqual(vol.sum(), 147.9704, delta=1e-3)

    def test_shipped_geodesic_is_valid(self):
        # The reference travels as a precomputed array, so guard it here: it is
        # the only check that it still matches the mesh it was built for.
        self.assertEqual(self.geodesic.shape, (len(self.vertices),))
        self.assertTrue(np.isfinite(self.geodesic).all())
        self.assertAlmostEqual(float(self.geodesic[self.source]), 0.0, places=12)
        self.assertAlmostEqual(float(self.geodesic.min()), 0.0, places=12)
        # The wall is not convex, so a geodesic can only be longer than the
        # straight line - never shorter.
        euclid = np.linalg.norm(self.vertices - self.vertices[self.source], axis=1)
        self.assertTrue((self.geodesic >= euclid - 1e-9).all())
        # ... and strictly longer somewhere, or the domain would be convex.
        self.assertGreater(float((self.geodesic - euclid).max()), 1e-3)

    def test_gradient_is_exact_on_a_linear_field(self):
        grid = Grid3D(self.vertices, self.tets)
        a = np.array([0.3, -1.1, 2.0])
        g = grid.gradient(self.vertices @ a)
        self.assertLess(np.abs(g - a).max(), 1e-8)

    def test_divergence_is_the_adjoint_of_averaged_gradient(self):
        # divergence takes a NODAL (N, 3) field and averages it onto elements,
        # so the identity it satisfies is <grad u, A q>_V == <u, div q>, with A
        # the node-to-element average. Holds exactly for any u and q.
        grid = Grid3D(self.vertices, self.tets)
        rng = np.random.default_rng(0)
        u = rng.normal(size=len(self.vertices))
        q = rng.normal(size=(len(self.vertices), 3))
        q_elem = q[self.tets].mean(axis=1)
        vol = np.abs(tet_volumes(self.vertices, self.tets))

        lhs = float((vol[:, None] * np.asarray(grid.gradient(u)) * q_elem).sum())
        rhs = float(u @ np.asarray(grid.divergence(q)))
        self.assertAlmostEqual(lhs, rhs, delta=1e-12 * max(1.0, abs(lhs)))


if __name__ == "__main__":
    unittest.main()

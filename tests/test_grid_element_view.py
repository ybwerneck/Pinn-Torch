"""Checks for the uniform element view shared by Grid and Grid3D.

The point of ``elements``, ``measure`` and ``dim`` is that a caller can use
either mesh class without branching on its type, so the tests assert the two
classes agree on the contract rather than on any particular value.
"""

import unittest

import numpy as np

from fisiocomPinn import Grid, Grid3D, structured_mesh


def unit_cube_tets(n=4):
    """A small tetrahedral mesh of the unit cube, built without scipy."""
    g = np.linspace(0.0, 1.0, n)
    verts = np.array([[x, y, z] for x in g for y in g for z in g], dtype=np.float64)

    def vid(i, j, k):
        return (i * n + j) * n + k

    # Each cube cell is split into the standard six tetrahedra, which tile it
    # without overlap, so the volumes must sum to exactly 1.
    tets = []
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                c = [vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k),
                     vid(i, j + 1, k), vid(i, j, k + 1), vid(i + 1, j, k + 1),
                     vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)]
                for a, b, cc, d in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                                    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)):
                    tets.append([c[a], c[b], c[cc], c[d]])
    return verts, np.array(tets, dtype=np.int32)


class ElementViewTests(unittest.TestCase):
    def setUp(self):
        v, f = structured_mesh(n=6, L=1.0)
        self.g2 = Grid(v, f)
        self.g3 = Grid3D(*unit_cube_tets(4))

    def test_elements_alias_the_native_connectivity(self):
        """``elements`` is the same array each class already exposed."""
        np.testing.assert_array_equal(self.g2.elements, self.g2.faces)
        np.testing.assert_array_equal(self.g3.elements, self.g3.tets)

    def test_measure_aliases_area_and_volume(self):
        np.testing.assert_allclose(self.g2.measure, self.g2._areas)
        np.testing.assert_allclose(self.g3.measure, self.g3._volumes)

    def test_measure_sums_to_the_domain(self):
        """Both meshes tile the unit domain, so the measures sum to 1."""
        self.assertAlmostEqual(float(np.sum(self.g2.measure)), 1.0, places=6)
        self.assertAlmostEqual(float(np.sum(self.g3.measure)), 1.0, places=6)

    def test_dim_is_present_on_both(self):
        self.assertEqual(self.g2.dim, 2)
        self.assertEqual(self.g3.dim, 3)

    def test_shapes_agree_with_the_gradient_operator(self):
        """The contract: one row of ``elements`` and one ``measure`` per element,
        matching the leading axis of ``_grads``, whose last axis is ``dim``."""
        for grid, nodes_per_element in ((self.g2, 3), (self.g3, 4)):
            with self.subTest(grid=type(grid).__name__):
                n_elem = len(grid.elements)
                self.assertEqual(grid.elements.shape, (n_elem, nodes_per_element))
                self.assertEqual(grid.measure.shape, (n_elem,))
                self.assertEqual(grid._grads.shape,
                                 (n_elem, nodes_per_element, grid.dim))

    def test_view_is_read_only_aliasing(self):
        """The properties expose the existing arrays, not copies, and cannot be
        assigned over — callers must not be able to desynchronise them from the
        geometry that was built from them."""
        self.assertIs(self.g2.elements, self.g2.faces)
        self.assertIs(self.g3.elements, self.g3.tets)
        with self.assertRaises(AttributeError):
            self.g2.elements = self.g2.faces
        with self.assertRaises(AttributeError):
            self.g3.measure = self.g3._volumes

    def test_a_type_agnostic_routine_runs_on_both(self):
        """The motivating use: integrate a field with no reference to the mesh type."""

        def integrate(grid, nodal):
            per_element = nodal[grid.elements].mean(axis=1)
            return float(np.dot(per_element, grid.measure))

        for grid in (self.g2, self.g3):
            with self.subTest(grid=type(grid).__name__):
                ones = np.ones(grid.N)
                self.assertAlmostEqual(integrate(grid, ones), 1.0, places=6)
                # A linear field integrates to its mean over the unit domain.
                x = grid.vertices[:, 0]
                self.assertAlmostEqual(integrate(grid, x), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()

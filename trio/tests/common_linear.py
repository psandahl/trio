import numpy as np
import unittest

from trio.common.camera import Camera
from trio.common.linear import triangulate

from .utils import equal_arrays


class CommonLinearTestCase(unittest.TestCase):
    def test_linear_triangulate(self):
        c0 = Camera(np.array([4, 3, 0]),
                    np.radians((-90, 0, 0)), np.radians((30, 20)))
        c1 = Camera(np.array([3, 3, 0]),
                    np.radians((-90, 0, 0)), np.radians((30, 20)))

        xyz = np.array([3.5, 0.3, 0.35])

        uv0 = c0.project(xyz)
        uv1 = c1.project(xyz)

        self.assertTrue(equal_arrays(xyz, triangulate(
            c0.projection_matrix, uv0, c1.projection_matrix, uv1)))

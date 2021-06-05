import numpy as np
import unittest

from trio.common.camera import Camera
from trio.common.linear import triangulate, solve_dlt
from trio.common.math import euclidean, homogeneous
from trio.common.matrix import matrix_look_at, matrix_decompose_ypr

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

    def test_linear_simple_dlt(self):
        # Setup camera.
        ypr = matrix_decompose_ypr(
            matrix_look_at(np.array((10, 10, 10)),
                           np.array((0, 0, 0)),
                           np.array((0, 0, 1))))

        c = Camera(np.array((10, 10, 10)), np.array(ypr), np.radians((50, 45)))

        # Produce points.
        xyzs = [np.array((-3.2, 1.3, 1.1)),
                np.array((-1.6, -2, 0.8)),
                np.array((0, 0, -1)),
                np.array((1.8, -1.6, -0.1)),
                np.array((1.2, 2.1, -0.6)),
                np.array((3.1, -2.7, 1.5)),
                np.array((3.3, 2.7, 1.8))]

        points = list()
        for xyz in xyzs:
            px = c.project(xyz)
            point = dict()
            point["x"] = xyz[0]
            point["y"] = xyz[1]
            point["z"] = xyz[2]
            point["u"] = px[0]
            point["v"] = px[1]
            points.append(point)

        # Run dlt.
        res, p = solve_dlt(points)

        self.assertTrue(res)

        # Compare projections.
        for xyz in xyzs:
            px1 = c.project(xyz)
            px2 = euclidean(p @ homogeneous(xyz))

            self.assertTrue(equal_arrays(px1, px2))

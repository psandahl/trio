import numpy as np
import numpy.linalg as linalg
import math
import unittest

from trio.common.math import normalize, column, euclidean, homogeneous
from trio.common.matrix import matrix_rank, matrix_ypr, \
    matrix_decompose_ypr, matrix_look_at, matrix_intrinsic


def equal_matrices(m, n):
    """
    Helper function to test if two matrices are somewhat equal.
    """
    if m.shape == n.shape and m.dtype == m.dtype:
        for row in range(m.shape[0]):
            for col in range(m.shape[1]):
                if math.fabs(m[row, col] - n[row, col]) > 0.000001:
                    return False
    else:
        return False

    return True


def normalized_camera_ray(fov):
    """
    Helper function to generate a normalized camera ray from angles.
    """
    x = math.tan(fov[0])
    y = math.tan(fov[1])
    z = 1.0
    return normalize(np.array([x, y, z]))


class CommonMatrixTestCase(unittest.TestCase):
    def test_matrix_ypr(self):
        # Start with a matrix with no rotation.
        zero_m = matrix_ypr(np.radians((0, 0, 0)))
        self.assertEqual(3, matrix_rank(zero_m))
        self.assertEqual((3, 3), zero_m.shape)

        # It shall be equal to the identity matrix.
        eye_m = np.eye(3, dtype=float)
        self.assertTrue(equal_matrices(eye_m, zero_m))

        # Continue with a "random" rotation.
        random_m = matrix_ypr(np.radians((-87, 13.2, 37)))
        self.assertEqual(3, matrix_rank(random_m))
        self.assertEqual((3, 3), random_m.shape)

        # Rotation matrix multiplied with its transpose shall be the identity.
        self.assertTrue(equal_matrices(eye_m, random_m.T @ random_m))

    def test_matrix_decompose_ypr(self):
        # Start with a matrix with no rotation.
        zero_m = matrix_ypr(np.radians((0, 0, 0)))
        zero_d = np.degrees(matrix_decompose_ypr(zero_m))
        self.assertAlmostEqual(0, zero_d[0])
        self.assertAlmostEqual(0, zero_d[1])
        self.assertAlmostEqual(0, zero_d[2])

        # Continue with a "random" rotation.
        random_m = matrix_ypr(np.radians((-87, 13.2, 37)))
        random_d = np.degrees(matrix_decompose_ypr(random_m))
        self.assertAlmostEqual(-87, random_d[0])
        self.assertAlmostEqual(13.2, random_d[1])
        self.assertAlmostEqual(37, random_d[2])

    def test_matrix_look_at_decompose_ypr(self):
        # Start with a matrix that shall have no rotation.
        zero_m = matrix_look_at(np.array([0, 0, 0]),
                                np.array([2, 0, 0]),
                                np.array([0, 0, 1]))
        self.assertEqual(3, matrix_rank(zero_m))

        # It shall be equal to the identity matrix.
        eye_m = np.eye(3, dtype=float)
        self.assertTrue(equal_matrices(eye_m, zero_m))

        # Continue with a random look at.
        random_m = matrix_look_at(np.array([4, 3.3, 2.9]),
                                  np.array([0, 0, 0]),
                                  np.array([0, 0.6, 0.7]))
        self.assertEqual(3, matrix_rank(random_m))

        # Decompose and recreate with ypr - shall be equal matrices.
        ypr = matrix_decompose_ypr(random_m)
        ypr_m = matrix_ypr(np.array(ypr))
        self.assertTrue(equal_matrices(random_m, ypr_m))

    def intrinsic_matrix(self, fov, rect):
        i = matrix_intrinsic(fov, rect)
        iInv = linalg.inv(i)

        # Build test data for the image mid point and the four corners.
        xs = [(np.array([0, 0]),
               np.array([rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0])),
              (np.array([-fov[0], -fov[1]]) / 2.0,
               np.array([rect[0], rect[1]])),
              (np.array([fov[0], -fov[1]]) / 2.0,
               np.array([rect[0] + rect[2], rect[1]])),
              (np.array([-fov[0], fov[1]]) / 2.0,
               np.array([rect[0], rect[1] + rect[3]])),
              (np.array([fov[0], fov[1]]) / 2.0,
               np.array([rect[0] + rect[2], rect[1] + rect[3]]))
              ]

        for x in xs:
            ray = normalized_camera_ray(x[0])
            px = euclidean(i @ column(ray))
            self.assertTrue(equal_matrices(column(x[1]), px))

            # Test inversion.
            ray2 = normalize(iInv @ homogeneous(px))
            self.assertTrue(equal_matrices(column(ray), ray2))

    def test_intrinsic_matrix(self):
        # Test unit size image size.
        self.intrinsic_matrix(np.radians((30, 20)),
                              np.array([-0.5, -0.5, 1.0, 1.0]))
        # Test ordinary image size type.
        self.intrinsic_matrix(np.radians((30, 20)),
                              np.array([0, 0, 720 - 1, 480 - 1]))

    def test_matrix_rank(self):
        # A matrix of zeros has rank zero.
        self.assertEqual(0, matrix_rank(np.zeros((3, 3))))

        # An identity matrix has full rank.
        self.assertEqual(3, matrix_rank(np.eye(3)))

        # A matrix of equal non-zero values has rank one.
        self.assertEqual(1, matrix_rank(np.ones((3, 3))))

        # A matrix with one column has rank one.
        m = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape((3, 3))
        self.assertEqual(1, matrix_rank(m))

        # A matrix with two independent colums has rank two.
        m = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1]).reshape((3, 3))
        self.assertEqual(2, matrix_rank(m))

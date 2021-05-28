import numpy as np
import unittest

from trio.common.matrix import matrix_rank


class CommonMatrixTestCase(unittest.TestCase):
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

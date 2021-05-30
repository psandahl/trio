import numpy as np
import unittest

from trio.common.camera import Camera
from trio.common.math import column

from .utils import equal_matrices


class CommonCameraTestCase(unittest.TestCase):
    def test_camera_space(self):
        # Create a simple camera.
        c = Camera(np.array([3, 3, 0]),
                   np.radians((-90, 0, 0)),
                   np.radians((30, 20)))

        # The camera's world coordinates shall be zero in camera space.
        self.assertTrue(equal_matrices(column(np.array([0, 0, 0])),
                                       column(c.camera_space(np.array([3, 3, 0])))))

        # Just ahead of camera shall be in positive z.
        self.assertTrue(equal_matrices(column(np.array([0, 0, 3])),
                                       column(c.camera_space(np.array([3, 0, 0])))))

        # To the "right" shall be positive x.
        self.assertTrue(equal_matrices(column(np.array([1, 0, 3])),
                                       column(c.camera_space(np.array([2, 0, 0])))))

        # To the "left" shall be negative x.
        self.assertTrue(equal_matrices(column(np.array([-1, 0, 3])),
                                       column(c.camera_space(np.array([4, 0, 0])))))

        # "Up" shall be negative y.
        self.assertTrue(equal_matrices(column(np.array([0, -1, 3])),
                                       column(c.camera_space(np.array([3, 0, 1])))))

        # "Down" shall be positive y.
        self.assertTrue(equal_matrices(column(np.array([0, 1, 3])),
                                       column(c.camera_space(np.array([3, 0, -1])))))

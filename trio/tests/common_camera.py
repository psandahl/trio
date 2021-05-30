import numpy as np
import unittest

from trio.common.camera import Camera
from trio.common.math import column

from .utils import equal_arrays


class CommonCameraTestCase(unittest.TestCase):
    def test_camera_space(self):
        # Create a simple camera.
        c = Camera(np.array([3, 3, 0]),
                   np.radians((-90, 0, 0)),
                   np.radians((30, 20)))

        # The camera's world coordinates shall be zero in camera space.
        self.assertTrue(equal_arrays(np.array([0, 0, 0]),
                                     c.camera_space(np.array([3, 3, 0]))))

        # Just ahead of camera shall be in positive z.
        self.assertTrue(equal_arrays(np.array([0, 0, 3]),
                                     c.camera_space(np.array([3, 0, 0]))))

        # To the "right" shall be positive x.
        self.assertTrue(equal_arrays(np.array([1, 0, 3]),
                                     c.camera_space(np.array([2, 0, 0]))))

        # To the "left" shall be negative x.
        self.assertTrue(equal_arrays(np.array([-1, 0, 3]),
                                     c.camera_space(np.array([4, 0, 0]))))

        # "Up" shall be negative y.
        self.assertTrue(equal_arrays(np.array([0, -1, 3]),
                                     c.camera_space(np.array([3, 0, 1]))))

        # "Down" shall be positive y.
        self.assertTrue(equal_arrays(np.array([0, 1, 3]),
                                     c.camera_space(np.array([3, 0, -1]))))

    def test_camera_center_of_projection(self):
        # Center of projection shall always return world position.
        c = Camera(np.array([1899.8, 3678, -8765.5]),
                   np.radians((-90, 33, 4)),
                   np.radians((30, 20)))

        self.assertTrue(equal_arrays(np.array([1899.8, 3678, -8765.5]),
                                     c.center_of_projection()))

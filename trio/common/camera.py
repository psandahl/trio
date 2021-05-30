from enum import Enum
import numpy as np

from .math import column, euclidean, homogeneous
from .matrix import matrix_ypr, matrix_intrinsic, matrix_permute_ecef


class Permutation(Enum):
    """
    Enumeration of the different permuations.
    """
    ECEF = 1


class Camera:

    camera_matrix = np.zeros((3, 4), dtype='float')
    projection_matrix = np.zeros((3, 4), dtype='float')

    def __init__(self, position, orientation, fov,
                 rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                 perm=Permutation.ECEF):
        if type(position) == np.ndarray and position.size == 3 and \
                type(orientation) == np.ndarray and orientation.size == 3 and \
                type(fov) == np.ndarray and fov.size == 2 and \
                type(rect) == np.ndarray and rect.size == 4 and \
                type(perm) == Permutation:

            # Only ECEF atm.
            permute = matrix_permute_ecef()

            # Create rotation matrix.
            r = matrix_ypr(orientation) @ permute

            # Create translation vector.
            t = r.T @ column(position * -1.0)

            # Put together camera matrix.
            self.camera_matrix = np.hstack((r.T, t))

        else:
            raise TypeError("Not the expected types to Camera")

    def camera_space(self, xyz):
        """
        Get the xyz coordinate mapped into the camera space.
        """
        return self.camera_matrix @ homogeneous(xyz)

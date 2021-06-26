from enum import Enum
import numpy as np
from scipy import linalg

from .math import column, euclidean, homogeneous
from .matrix import matrix_ypr, matrix_intrinsic, matrix_permute_ecef, \
    matrix_permute_ned


class Permutation(Enum):
    """
    Enumeration of the different permuations.
    """
    ECEF = 1
    NED = 2


class Camera:

    intrinsic_matrix = np.zeros((3, 3), dtype='float')
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

            permute = matrix_permute_ecef()
            if perm == Permutation.NED:
                permute = matrix_permute_ned()

            # Create rotation matrix.
            r = matrix_ypr(orientation) @ permute

            # Create translation vector.
            t = r.T @ column(position * -1.0)

            # Put together camera matrix.
            self.camera_matrix = np.hstack((r.T, t))

            # Create intrinsic matrix and put together the projection matrix.
            intrinsic_matrix = matrix_intrinsic(fov, rect)
            self.projection_matrix = intrinsic_matrix @ self.camera_matrix

        else:
            raise TypeError("Not the expected types to Camera")

    def camera_space(self, xyz):
        """
        Get the xyz coordinate mapped into the camera space.
        """
        return self.camera_matrix @ homogeneous(xyz)

    def center_of_projection(self):
        """
        Get the camera's position in world coordinates.
        """
        r = self.camera_matrix[:, :3]
        t = self.camera_matrix[:, 3:]

        return (r.T @ (t * -1.0)).flatten()

    def rotation_matrix(self):
        """
        Get the camera's rotation matrix.
        """
        return self.camera_matrix[:, :3].T

    def project(self, xyz):
        """
        Project the xyz coordinate into image space.
        """
        return euclidean(self.projection_matrix @ homogeneous(xyz))


def camera_from_param(param, rect=np.array([-0.5, -0.5, 1.0, 1.0]),
                      perm=Permutation.ECEF):
    """
    Create a camera from parameters.
    """
    position = np.array((param["x"], param["y"], param["z"]))
    orientation = np.array((param["yaw"], param["pitch"], param["roll"]))
    fov = np.array((param["horizontal-fov"], param["vertical-fov"]))
    return Camera(position, np.radians(orientation), np.radians(fov), rect, perm)


def camera_reprojection_errors(points, camera):
    """
    Reproject the 3d points through the camera and return the reprojection
    errors.
    """
    err = []
    for point in points:
        xyz = np.array((point["x"], point["y"], point["z"]))
        uv0 = np.array((point["u"], point["v"]))
        uv1 = camera.project(xyz)
        err.append(linalg.norm(uv1 - uv0))

    return err

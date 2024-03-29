# Matrix helper functions.
# The world coordinate system for trio is a right handed system like
# the following.
#
#            Z
#            |
#            |____ Y
#            /
#           /
#          X
#
# The Euler rotations - yaw, pitch and roll - will be a rotation on
# the world axis in the order Y, Y and X.
#
#
# The camera coordinate system is a right hand system where the axis
# are permuted like:
#
#      X ____
#           /|
#          / |
#         Z  Y
#
# I.e. the Z axis is the direction for the camera, X is to the right,
# and Y is down.
import math
import numpy as np
from scipy import linalg
from scipy.linalg import svdvals

from .math import normalize, focal_length


def matrix_ypr(ypr):
    """
    Create a 3x3 rotation matrix relative the world coordinate system.
    """
    if type(ypr) == np.ndarray and ypr.size == 3:
        sy = math.sin(ypr[0])
        cy = math.cos(ypr[0])
        sp = math.sin(ypr[1])
        cp = math.cos(ypr[1])
        sr = math.sin(ypr[2])
        cr = math.cos(ypr[2])
        m = [
            cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp, cp * sr, cp * cr
        ]
        return np.array(m).reshape(3, 3)
    else:
        raise TypeError("ypr must be a numpy array of size three")


def matrix_decompose_ypr(m):
    """
    Decompose a world rotation matrix into a tuple of yaw, pitch and roll.
    """
    if type(m) == np.ndarray and m.shape == (3, 3):
        return (math.atan2(m[1, 0], m[0, 0]),
                -math.asin(m[2, 0]),
                math.atan2(m[2, 1], m[2, 2]))
    else:
        raise TypeError("m must be a 3x3 matrix")


def matrix_look_at(eye, at, up):
    """
    Create a 3x3 rotation matrix relative to world coordinate system.
    """
    if type(eye) == np.ndarray and eye.size == 3 and \
            type(at) == np.ndarray and at.size == 3 and \
            type(up) == np.ndarray and up.size == 3:
        front = normalize(at - eye)
        side = normalize(np.cross(up, front))
        up = np.cross(front, side)
        return np.column_stack((front, side, up))
    else:
        raise TypeError("eye, at and up must be arrays with three components")


def matrix_intrinsic(fov, rect):
    """
    Create a intrinsic matrix. fov is an array with two components, the
    horizontal field of view and the vertical field of view. The rect is n array
    with four components, the x start, the y start, the width and the height.
    """
    if type(fov) == np.ndarray and fov.size == 2 and \
            type(rect) == np.ndarray and rect.size == 4:
        fx = focal_length(fov[0], rect[2])
        fy = focal_length(fov[1], rect[3])
        cx = rect[0] + rect[2] / 2.0
        cy = rect[1] + rect[3] / 2.0
        m = [
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1
        ]
        return np.array(m).reshape(3, 3)
    else:
        raise TypeError("fov and rect must both be arrays")


def matrix_permute_ecef():
    """
    Create a permutation matrix that transforms from ecef to camera axes.
    """
    return np.array([0.0,  0.0, 1.0,
                     -1.0, 0.0, 0.0,
                     0.0, -1.0, 0.0]).reshape(3, 3)


def matrix_permute_ned():
    """
    Create a permutation matrix that transforms from ned to camera axes.
    """
    return np.array([0.0, 0.0, 1.0,
                     1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0]).reshape(3, 3)


def matrix_skew_symmetric(vec3):
    """
    Create a skew symmetric matrix from a 3d vector.
    """
    x, y, z = vec3
    m = [
        0, -z, y,
        z, 0, -x,
        -y, x, 0
    ]
    return np.array(m).reshape(3, 3)


def matrix_relative_rotation(r0, r1):
    """
    Find the relative rotation to go from r0 to r1.
    """
    return r1 @ r0.T


def matrix_essential(r, t):
    """
    Get the essential matrix from rotation and translation.
    """
    return r @ matrix_skew_symmetric(t)


def matrix_decompose_camera(camera, permute):
    """
    Given camera and permute matrices decompose into ypr angles and translation.
    """
    if type(camera) == np.ndarray and camera.shape == (3, 4) and \
            type(permute) == np.ndarray and permute.shape == (3, 3):
        # Split camera matrix into rotation and translation.
        r = camera[:, :3]
        r = r.T

        t = camera[:, 3:]
        t = r @ (t * -1.0)

        # Adjust rotation and decompose into ypr.
        r = r @ permute.T

        return (matrix_decompose_ypr(r), t.flatten())
    else:
        raise TypeError("Matrices are not the expected types")


def matrix_decompose_projection(projection, intrinsic, permute):
    """
    Given instrinsic and permute matrices decompose projection matrix
    into ypr angles and translation.
    """
    if type(projection) == np.ndarray and projection.shape == (3, 4) and \
            type(intrinsic) == np.ndarray and intrinsic.shape == (3, 3) and \
            type(permute) == np.ndarray and permute.shape == (3, 3):

        # Remove the instrinsic matrix and decompose camera part.
        return matrix_decompose_camera(linalg.inv(intrinsic) @ projection, permute)
    else:
        raise TypeError("Matrices are not the expected types")


def matrix_rank(m):
    """
    Calculate the matrix rank for a matrix.
    """
    svals = svdvals(m)
    rank = 0
    for s in svals:
        if s < 0.00000001:
            return rank
        else:
            rank += 1
    return rank

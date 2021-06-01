import numpy as np
from scipy import linalg

from .camera import Camera
from .math import euclidean


def triangulate(p0, uv0, p1, uv1):
    """
    Triangulate a 3d position from two projection matrices and two image points.
    """
    if type(p0) == np.ndarray and p0.shape == (3, 4) and \
            type(uv0) == np.ndarray and uv0.size == 2 and \
            type(p1) == np.ndarray and p1.shape == (3, 4) and \
            type(uv1) == np.ndarray and uv1.size == 2:
        # Extract the rows of projection matrices p0 and p1.
        p0_1 = p0[0:1, :]
        p0_2 = p0[1:2, :]
        p0_3 = p0[2:3, :]

        p1_1 = p1[0:1, :]
        p1_2 = p1[1:2, :]
        p1_3 = p1[2:3, :]

        # Construct matrix A.
        A = np.vstack(
            (uv0[0] * p0_3 - p0_1,
             uv0[1] * p0_3 - p0_2,
             uv1[0] * p1_3 - p1_1,
             uv1[1] * p1_3 - p1_2
             )
        )

        # Decompose A.
        U, s, Vt = linalg.svd(A)

        return euclidean(Vt[3:4, :].T)
    else:
        raise TypeError("Not the expected types for triangulate")

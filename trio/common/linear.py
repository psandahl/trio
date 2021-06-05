import numpy as np
import math
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


def solve_dlt(points):
    if type(points) == list and len(points) >= 6:
        A = np.zeros((len(points) * 2, 12), dtype=float)

        for i in range(len(points)):
            point = points[i]

            r1 = i * 2
            A[r1, 0] = point["x"]
            A[r1, 1] = point["y"]
            A[r1, 2] = point["z"]
            A[r1, 3] = 1.0
            A[r1, 8] = -point["u"] * point["x"]
            A[r1, 9] = -point["u"] * point["y"]
            A[r1, 10] = -point["u"] * point["z"]
            A[r1, 11] = -point["u"]

            r2 = r1 + 1
            A[r2, 4] = point["x"]
            A[r2, 5] = point["y"]
            A[r2, 6] = point["z"]
            A[r2, 7] = 1.0
            A[r2, 8] = -point["v"] * point["x"]
            A[r2, 9] = -point["v"] * point["y"]
            A[r2, 10] = -point["v"] * point["z"]
            A[r2, 11] = -point["v"]

        U, s, Vt = linalg.svd(A)
        smallest_singular_value = s[s.shape[0] - 1]
        if smallest_singular_value > 0.000001:
            return (False, np.zeros((3, 4)))

        h = Vt[11:, :]
        if h[0, 11] < 0.0:
            h *= -1.0

        norm = math.sqrt(h[0, 8] ** 2 + h[0, 9] ** 2 + h[0, 10] ** 2)
        h /= norm

        p = h.reshape(3, 4)
        return (True, p)
    else:
        raise TypeError("Not the expected types or length for solve_dlt")

# Common math helper functions.
import numpy.linalg as linalg


def normalize(m):
    """
    Normalize an array by its norm.
    """
    return m / linalg.norm(m)

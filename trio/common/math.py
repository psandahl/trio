# Common math helper functions.
import numpy as np
import numpy.linalg as linalg
import math


def normalize(a):
    """
    Normalize an array by its norm.
    """
    return a / linalg.norm(a)


def column(v):
    """
    Make a column matrix of a 1D vector.
    """
    if type(v) == np.ndarray and v.ndim == 1:
        return v.reshape(v.size, 1)
    else:
        raise TypeError("v must be 1D array")


def focal_length(theta, side):
    """
    Calculate the focal length given an angle theta and a side.
    """
    return (side / 2.0) / math.tan(theta / 2.0)


def euclidean(v):
    """
    Convert to a euclidean array.
    """
    if type(v) == np.ndarray and v.size == 3:
        return (v / v[2])[:2]
    elif type(v) == np.ndarray and v.size == 4:
        return (v / v[3])[:3]
    else:
        raise TypeError("v must be array of size 3 or 4")


def homogeneous(v, w=1.0):
    """
    Convert to a homogeneous array.
    """
    if type(v) == np.ndarray and v.ndim == 1:
        return np.append(v, w)
    elif type(v) == np.ndarray and v.shape[1] == 1:
        return column(np.append(v, w))
    else:
        raise TypeError("v must be 1D array or column vector")

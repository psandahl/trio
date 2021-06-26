# Common math helper functions.
from functools import reduce
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


def ssd(xs):
    """
    Square and sum the values.
    """
    return reduce(lambda acc, x: acc + x**2, xs, 0.0)


def sad(xs):
    """
    Sum the absolute of the values.
    """
    return reduce(lambda acc, x: acc + abs(x), xs, 0.0)


def epipolar_line(F, p):
    """
    Get the epipolar line for a point. Line is on format ax + bx + c = 0.
    """
    a, b, c = F @ homogeneous(p)

    # Normalize so that a ** 2 + b ** 2 is 1.
    norm = math.sqrt(a ** 2 + b ** 2)

    return np.array((a, b, c)) / norm


def plot_on_line(l, x):
    """
    Plot the x on the line.
    """
    a, b, c = l
    return (a * x + c) / -b

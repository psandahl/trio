import math


def equal_matrices(m, n):
    """
    Helper function to test if two matrices are somewhat equal.
    """
    if m.shape == n.shape and m.dtype == m.dtype and m.ndim > 1:
        for row in range(m.shape[0]):
            for col in range(m.shape[1]):
                if math.fabs(m[row, col] - n[row, col]) > 0.000001:
                    return False
    else:
        raise TypeError("m and n are not matrices")

    return True


def equal_arrays(m, n):
    """
    Helper function to test if two arrays are somewhat equal.
    """
    if m.size == n.size and m.dtype == m.dtype and m.ndim == 1:
        for i in range(m.size):
            if math.fabs(m[i] - n[i]) > 0.000001:
                return False
    else:
        raise TypeError("m and n are not arrays")

    return True

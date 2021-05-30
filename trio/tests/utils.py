import math


def equal_matrices(m, n):
    """
    Helper function to test if two matrices are somewhat equal.
    """
    if m.shape == n.shape and m.dtype == m.dtype:
        for row in range(m.shape[0]):
            for col in range(m.shape[1]):
                if math.fabs(m[row, col] - n[row, col]) > 0.000001:
                    return False
    else:
        return False

    return True

import numpy as np
from scipy.linalg import svdvals


def matrix_rank(m):
    svals = svdvals(m)
    rank = 0
    for s in svals:
        if s < 0.00001:
            return rank
        else:
            rank += 1
    return rank

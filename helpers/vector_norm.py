import numpy as np


def vector_norm(v):
    return np.sqrt(sum(x**2 for x in v))
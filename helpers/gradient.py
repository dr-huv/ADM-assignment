import numpy as np


def gradient(x):
    '''Just computing the gradient values directly'''
    x1, x2 = x
    df_dx1 = 6*x1**2*(x1**3 - x2) - 8*(x2-x1)**3
    df_dx2 = -2*(x1**3 - x2) +8*(x2-x1)**3
    return np.array([df_dx1, df_dx2])
import numpy as np


def hessain(x):
    '''Now we find the hessian as a 2x2 matrix'''
    x1, x2 = x
    d2f_dx1_2 = 12*x1*(x1**3 - x2) + 6*x1**2*6*x1 + 24*(x2 - x1)**2
    d2f_dx1_dx2 = -6*x1**2 - 24*(x2 - x1)**2
    d2f_dx2_2 = 2 + 24*(x2 - x1)**2
    return np.array([[d2f_dx1_2, d2f_dx1_dx2], [d2f_dx1_dx2, d2f_dx2_2]])

import numpy as np

def matrix_inverse(A):
    '''Here we find the inverse of the matrix , works only for 2x2 matrices'''
    #first compute the determinant det as d = ad - bc
    det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
    
    #now say we get a det = 0 or a really small value (would result in integer overflow), to avoid zero division error we add a little regularisation
    if abs(det) < 1e-10:
        A[0,0] += 1e-6
        A[1,1] += 1e-6
        det = A[0, 0] * A[1,1] - A[0,1] * A[1,0]
    
    inv = np.zeros((2,2)) #initialised a 2x2 0 matrix

    #now we do the classic inv for 2x2 matrices exchange diagnoals and negate cross diagnoals
    inv[0,0] = A[1,1]/det
    inv[0,1] = -A[0,1]/det
    inv[1,0] = -A[1,0]/det
    inv[1,1] = A[0,0]/det

    return inv
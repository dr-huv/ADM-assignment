import numpy as np

def matrix_vector_multiply(A,v):
    '''Multiplying a matrix A with a vector v'''
    n = len(v)
    result = np.zeros(n)

    for i in range(n):
        for j in range(n):
            result[i]+= A[i,j] * v[j]

    return result 


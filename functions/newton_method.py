from helpers.f import f
from helpers.gradient import gradient
from helpers.hessian import hessain
from helpers.matrix_inverse import matrix_inverse
from helpers.matrix_vector_multiply import matrix_vector_multiply
from helpers.vector_norm import vector_norm


def newton_method(x0, max_iter=1000, stop_condition='relative_f', tol_f=0.001, tol_x=0.01):
    '''basic implimentation of the newton method
        The parameters are
        x0: This is our initial guess i.e (2,0) in the question
        max_iter: maximum number of iterations to be performed (default: 1000)
        stop_condition: we use this to choose which stop condition to use relative f or absolute x
        tol_f: the lower limit of how much fluctuation we can tolerate in f before the changes are considered too small for further iterations
        told_x: same as above but for norm(x)
    '''
    # We start with an initial guess x0
    x = x0.copy()  # we use .copy() because x0 is a list and we want to store a copy of it not a reference
    # We store a list of all the values x and f have taken, (useful for when we want to graph this)
    x_history = [x.copy()]
    f_history = [f(x)]

    # run the loop for max_iter times, or break out earlier if we reach within the threshold
    for iter in range(max_iter):
        x_prev = x.copy()
        f_prev = f(x)

        # compute gradient for the current value of x
        grad = gradient(x)
        H = hessain(x)

        # Now we compute the newton direction from matrix inversion
        H_inv = matrix_inverse(H)
        # This step solves the equation H * delta_x = -grad
        delta_x = matrix_vector_multiply(H_inv, grad)

        # Update the value of x since we got the delta x
        x -= delta_x

        # Now add this new value of x to our history, and f as well
        x_history.append(x.copy())
        f_history.append(f(x))

        # Now we have implimented 2 stopping conditions, either relative change in f or absolute change in x
        if stop_condition == 'relative_f':
            # Relative change in function value: |f(x_new) - f(x_old)| / |f(x_old)| <= tol_f
            if abs((f(x) - f_prev)/f_prev) <= tol_f:
                break
        else:
            # Absolute change in x:  ||x_new - x_old ||  <= tol_x
            if vector_norm(x - x_prev) <= tol_x:
                break

    return x, f(x), x_history, f_history, iter+1

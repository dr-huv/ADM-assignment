from helpers.f import f
from helpers.gradient import gradient
from helpers.vector_norm import vector_norm


def steepest_descent(x0, max_iter=1000, alpha=0.01, stop_condition='relative_f', tol_f=0.001, tol_x=0.01):
    '''Basic implimentation of steepest descent
        The parameters are:-
        x0: This is our initial guess i.e (2,0) in the question
        max_iter: maximum number of iterations to be performed (default: 1000)
        alpha: the alpha value given in question
        stop_condition: we use this to choose which stop condition to use relative f or absolute x
        tol_f: the lower limit of how much fluctuation we can tolerate in f before the changes are considered too small for further iterations
        told_x: same as above but for norm(x)
    '''
    x = x0.copy()
    x_history = [x0.copy()]
    f_history = [f(x)]

    for iter in range(max_iter):
        x_prev = x
        f_prev = f(x)

        grad = gradient(x)
        x = x - alpha*grad

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

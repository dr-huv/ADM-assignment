from helpers.f import f
from helpers.gradient import gradient
from helpers.vector_norm import vector_norm


def cyclic_coordinate(x0, max_iter=1000, alpha=0.01, stop_condition = 'relative_f', tol_f = 0.001, tol_x=0.01):
    '''Basic implimentation of Cyclic Coordinate method'''
    x = x0.copy()
    x_history = [x.copy()]
    f_history = [f(x)]

    for iter in range(max_iter):
        x_prev = x.copy()
        f_prev = f(x)

        #optimise for x1
        grad = gradient(x)
        x[0] = x[0] - alpha * grad[0]

        #optimise for x2
        grad = gradient(x)
        x[1] = x[1] - alpha * grad[1]

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

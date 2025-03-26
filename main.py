import numpy as np

from functions.cylic_coordinate import cyclic_coordinate
from functions.steepest_descent import steepest_descent
from functions.newton_method import newton_method
from tools.plot_convergence import plot_convergence
from tools.plot_surface import plot_surface
from helpers.points_are_close import points_are_close


if __name__ == "__main__":
    # Starting point
    x0 = np.array([2.0, 0.0])

    # Parameters
    alpha_cc = 0.01  # Step size for Cyclic Coordinate
    alpha_sd = 0.01  # Step size for Steepest Descent

    # Run methods with stopping condition based on relative function value
    print("Using stopping condition: |f(k+1) - f(k)| / |f(k)| <= 0.001")

    cc_result_f, cc_fval_f, cc_x_history_f, cc_f_history_f, cc_iters_f = cyclic_coordinate(
        x0, alpha=alpha_cc, stop_condition='relative_f')

    sd_result_f, sd_fval_f, sd_x_history_f, sd_f_history_f, sd_iters_f = steepest_descent(
        x0, alpha=alpha_sd, stop_condition='relative_f')

    newton_result_f, newton_fval_f, newton_x_history_f, newton_f_history_f, newton_iters_f = newton_method(
        x0, stop_condition='relative_f')

    print(
        f"Cyclic Coordinate Method: x = {cc_result_f}, f(x) = {cc_fval_f}, iterations = {cc_iters_f}")
    print(
        f"Steepest Descent Method: x = {sd_result_f}, f(x) = {sd_fval_f}, iterations = {sd_iters_f}")
    print(
        f"Newton's Method: x = {newton_result_f}, f(x) = {newton_fval_f}, iterations = {newton_iters_f}")

    # Plot convergence for relative function value stopping condition
    plot_convergence(
        [cc_x_history_f, sd_x_history_f, newton_x_history_f],
        [cc_f_history_f, sd_f_history_f, newton_f_history_f],
        ['Cyclic Coordinate', 'Steepest Descent', 'Newton']
    )

    print("\nUsing stopping condition: |x(k+1) - x(k)| <= 0.01")

    # Run methods with stopping condition based on absolute change in x
    cc_result_x, cc_fval_x, cc_x_history_x, cc_f_history_x, cc_iters_x = cyclic_coordinate(
        x0, alpha=alpha_cc, stop_condition='absolute_x')

    sd_result_x, sd_fval_x, sd_x_history_x, sd_f_history_x, sd_iters_x = steepest_descent(
        x0, alpha=alpha_sd, stop_condition='absolute_x')

    newton_result_x, newton_fval_x, newton_x_history_x, newton_f_history_x, newton_iters_x = newton_method(
        x0, stop_condition='absolute_x')

    print(
        f"Cyclic Coordinate Method: x = {cc_result_x}, f(x) = {cc_fval_x}, iterations = {cc_iters_x}")
    print(
        f"Steepest Descent Method: x = {sd_result_x}, f(x) = {sd_fval_x}, iterations = {sd_iters_x}")
    print(
        f"Newton's Method: x = {newton_result_x}, f(x) = {newton_fval_x}, iterations = {newton_iters_x}")

    # Plot convergence for absolute change in x stopping condition
    plot_convergence(
        [cc_x_history_x, sd_x_history_x, newton_x_history_x],
        [cc_f_history_x, sd_f_history_x, newton_f_history_x],
        ['Cyclic Coordinate', 'Steepest Descent', 'Newton']
    )

    # Plot 3D surface with optimal points
    optimal_points = [cc_result_f, sd_result_f, newton_result_f]
    plot_surface(min(p[0] for p in optimal_points) - 0.5,
                 max(p[0] for p in optimal_points) + 0.5,
                 min(p[1] for p in optimal_points) - 0.5,
                 max(p[1] for p in optimal_points) + 0.5,
                 optimal_points)

    # Compare if methods converge to the same point
    print("\nDo the methods converge to the same point?")
    print(
        f"Cyclic Coordinate vs Steepest Descent: {'Yes' if points_are_close(cc_result_f, sd_result_f) else 'No'}")
    print(
        f"Cyclic Coordinate vs Newton: {'Yes' if points_are_close(cc_result_f, newton_result_f) else 'No'}")
    print(
        f"Steepest Descent vs Newton: {'Yes' if points_are_close(sd_result_f, newton_result_f) else 'No'}")

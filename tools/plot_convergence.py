import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(x_history_list, f_history_list, method_names):
    plt.figure(figsize=(12, 10))

    # Plot function value convergence
    plt.subplot(2, 1, 1)
    for i, f_history in enumerate(f_history_list):
        plt.semilogy(range(len(f_history)), f_history, label=method_names[i])
    plt.xlabel('Iteration')
    plt.ylabel('Function Value (log scale)')
    plt.title('Convergence of Function Value')
    plt.legend()
    plt.grid(True)

    # Plot path in x1-x2 space
    plt.subplot(2, 1, 2)
    for i, x_history in enumerate(x_history_list):
        x_history = np.array(x_history)
        plt.plot(x_history[:, 0], x_history[:, 1], 'o-',
                 label=method_names[i], markersize=3)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Optimization Path')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
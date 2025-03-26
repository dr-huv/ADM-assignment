import matplotlib.pyplot as plt
import numpy as np
from helpers.f import f


def plot_surface(x_min, x_max, y_min, y_max, optimal_points):
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    # Plot optimal points
    for i, point in enumerate(optimal_points):
        ax.scatter(point[0], point[1], f(point), color='red',
                   s=100, label=f'Optimal Point {i+1}')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('Objective Function Surface')
    plt.legend()
    plt.show()

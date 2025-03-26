from helpers.vector_norm import vector_norm


def points_are_close(p1, p2, tol=1e-2):
    return vector_norm(p1 - p2) <= tol

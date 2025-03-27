from src.linalg.matrix import Matrix


def covariance_matrix(x_centered: Matrix) -> Matrix:
    result = 1 / (x_centered.row_len - 1) * x_centered.transpose() * x_centered
    return result

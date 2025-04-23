from typing import List, Tuple

from algorithms_pca.matrix import Matrix


def center_data(X: Matrix) -> Tuple[Matrix, List[float]]:
    n, m = X.rows, X.cols
    means: List[float] = [
        sum(X._data[i][j] for i in range(n)) / n for j in range(m)
    ]
    centered = [
        [X._data[i][j] - means[j] for j in range(m)] for i in range(n)
    ]
    return Matrix(centered), means


def covariance_matrix(Xc: Matrix) -> Matrix:
    n = Xc.rows
    return Xc.T.matmul(Xc) / (n - 1)

from algorithms_pca.matrix import Matrix


def center_data(X: Matrix) -> Matrix:
    n, m = X.rows, X.cols
    col_means = X.mean(axis=0)
    centered = [
        [X._data[i][j] - col_means._data[0][j] for j in range(m)]
        for i in range(n)
    ]
    return Matrix(centered)


def covariance_matrix(X_centered: Matrix) -> Matrix:
    n = X_centered.rows
    XT = X_centered.T
    C = XT.matmul(X_centered)
    return C / (n - 1)

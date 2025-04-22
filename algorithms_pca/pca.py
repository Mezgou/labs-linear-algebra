from algorithms_pca.matrix import Matrix
from algorithms_pca.eigen import find_eigenvalues, find_eigenvectors
from algorithms_pca.statistics import center_data, covariance_matrix


def explained_variance_ratio(eigenvalues: list[float], k: int) -> float:
    if not isinstance(eigenvalues, list) or not all(isinstance(ev, (int, float)) for ev in eigenvalues):
        raise TypeError("eigenvalues must be a list of numbers.")
    m = len(eigenvalues)
    if k < 1 or k > m:
        raise ValueError(f"k must be between 1 and {m}, got {k}.")
    total = sum(eigenvalues)
    if total == 0:
        raise ValueError("Sum of eigenvalues is zero; cannot compute ratio.")
    ev_sorted = sorted(eigenvalues, reverse=True)
    explained = sum(ev_sorted[:k])
    return explained / total


def pca(X: Matrix, k: int) -> tuple[Matrix, float]:
    if not isinstance(X, Matrix):
        raise TypeError("X must be a Matrix instance.")
    n, m = X.rows, X.cols
    if k < 1 or k > m:
        raise ValueError(f"k must be between 1 and {m}")

    X_centered = center_data(X)

    C = covariance_matrix(X_centered)

    eigenvalues = find_eigenvalues(C)
    eigenvectors = find_eigenvectors(C, eigenvalues)
    idx_sorted = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)
    top_evecs = [eigenvectors[i] for i in idx_sorted[:k]]

    proj_cols = [[evec._data[r][0] for evec in top_evecs] for r in range(m)]
    V = Matrix(proj_cols)
    X_proj = X_centered.matmul(V)

    explained_ratio = explained_variance_ratio(eigenvalues, k)
    return X_proj, explained_ratio

from algorithms_pca.matrix import Matrix
from algorithms_pca.eigen import find_eigenvalues, find_eigenvectors
from algorithms_pca.statistics import center_data, covariance_matrix


def explained_variance_ratio(eigenvalues: list[float], k: int) -> float:
    """
    Compute the ratio of explained variance for the first k eigenvalues.

    Args:
        eigenvalues: list of eigenvalues (not necessarily sorted)
        k: number of top components

    Returns:
        Ratio of explained variance: sum(lambda_i for i=1..k) / sum(lambda_i for i=1..m)
    """
    if not isinstance(eigenvalues, list) or not all(isinstance(ev, (int, float)) for ev in eigenvalues):
        raise TypeError("eigenvalues must be a list of numbers.")
    m = len(eigenvalues)
    if k < 1 or k > m:
        raise ValueError(f"k must be between 1 and {m}, got {k}.")
    total = sum(eigenvalues)
    if total == 0:
        raise ValueError("Sum of eigenvalues is zero; cannot compute ratio.")
    # sort in descending order
    ev_sorted = sorted(eigenvalues, reverse=True)
    explained = sum(ev_sorted[:k])
    return explained / total


def pca(X: Matrix, k: int) -> tuple[Matrix, float]:
    """
    Perform Principal Component Analysis.

    Args:
        X: data matrix of shape (n x m) where rows are samples
        k: number of principal components to keep

    Returns:
        X_proj: projected data matrix (n x k)
        explained_ratio: ratio of explained variance for top k components
    """
    if not isinstance(X, Matrix):
        raise TypeError("X must be a Matrix instance.")
    n, m = X.rows, X.cols
    if k < 1 or k > m:
        raise ValueError(f"k must be between 1 and {m}")

    # 1. Center data
    X_centered = center_data(X)

    # 2. Covariance matrix (m x m)
    C = covariance_matrix(X_centered)

    # 3. Eigen decomposition
    eigenvalues = find_eigenvalues(C)
    eigenvectors = find_eigenvectors(C, eigenvalues)
    # sort eigenvalues and eigenvectors together by descending eigenvalue
    idx_sorted = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)
    top_evecs = [eigenvectors[i] for i in idx_sorted[:k]]  # each is (m x 1) vector

    # 4. Project data: X_centered (n x m) times eigenvector matrix (m x k)
    # build a projection matrix V of shape m x k
    proj_cols = [[evec._data[r][0] for evec in top_evecs] for r in range(m)]
    V = Matrix(proj_cols)  # m x k
    X_proj = X_centered.matmul(V)  # yields n x k

    explained_ratio = explained_variance_ratio(eigenvalues, k)
    return X_proj, explained_ratio

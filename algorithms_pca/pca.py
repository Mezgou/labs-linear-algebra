from typing import List, Tuple
from algorithms_pca.matrix import Matrix
from algorithms_pca.eigen import find_eigenvalues, find_eigenvectors
from algorithms_pca.statistics import center_data, covariance_matrix


# Доля объяснённой дисперсии
def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    if k < 1 or k > len(eigenvalues):
        raise ValueError(f"k must be between 1 and {len(eigenvalues)}")
    total = sum(eigenvalues)
    if total == 0:
        raise ValueError("Sum of eigenvalues is zero.")
    top = sum(sorted(eigenvalues, reverse=True)[:k])
    return top / total


# Простой PCA
def pca(X: Matrix, k: int) -> Tuple[Matrix, float]:
    if k < 1 or k > X.cols:
        raise ValueError(f"k must be between 1 and {X.cols}")
    Xc, _ = center_data(X)
    C = covariance_matrix(Xc)
    vals = find_eigenvalues(C)
    vecs = find_eigenvectors(C, vals)
    idx = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:k]
    V = Matrix([[vec._data[r][0] for vec in [vecs[i] for i in idx]] for r in range(X.cols)])
    Xp = Xc.matmul(V)
    ratio = explained_variance_ratio(vals, k)
    return Xp, ratio


# PCA с QR
def pca_qr(X: Matrix, k: int) -> Tuple[Matrix, float, List[float], Matrix, List[float]]:
    Xc, means = center_data(X)
    Cov = covariance_matrix(Xc)
    raw_vals, Qtot = Cov.qr_eigen(max_iter=1000, tol=1e-12)
    pairs = sorted([(val, [Qtot._data[r][i] for r in range(Cov.rows)]) for i, val in enumerate(raw_vals)], key=lambda x: x[0], reverse=True)
    top_vals, top_vecs = zip(*pairs[:k])
    W = Matrix([[vec[row] for vec in top_vecs] for row in range(Cov.rows)])
    Xp = Xc.matmul(W)
    ratio = sum(top_vals) / sum(raw_vals) if sum(raw_vals) else 1.0
    return Xp, ratio, list(raw_vals), W, means

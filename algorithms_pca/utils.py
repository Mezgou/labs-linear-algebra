import math
import random

from algorithms_pca.matrix import Matrix, DimensionError
from algorithms_pca.pca import pca_qr
from algorithms_pca.statistics import covariance_matrix, center_data
from algorithms_pca.eigen import find_eigenvalues
from sklearn.datasets import load_iris


# MSE восстановления
def reconstruction_error(X_orig: Matrix, X_recon: Matrix) -> float:
    if not isinstance(X_orig, Matrix) or not isinstance(X_recon, Matrix):
        raise DimensionError("Inputs must be Matrix instances.")
    if X_orig.rows != X_recon.rows or X_orig.cols != X_recon.cols:
        raise DimensionError("Matrices must have the same dimensions.")
    n, m = X_orig.rows, X_orig.cols
    total = 0.0
    for i in range(n):
        for j in range(m):
            diff = X_orig._data[i][j] - X_recon._data[i][j]
            total += diff * diff
    return total / (n * m)


# Подбор k
def auto_select_k(eigenvalues: list[float], threshold: float = 0.95) -> int:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    vals = sorted(eigenvalues, reverse=True)
    total = sum(vals)
    cum = 0.0
    for idx, v in enumerate(vals, start=1):
        cum += v
        if cum / total >= threshold:
            return idx
    return len(vals)


# Заполнение NaN
def handle_missing_values(X: Matrix) -> Matrix:
    if not isinstance(X, Matrix):
        raise DimensionError("X must be a Matrix instance.")
    n, m = X.rows, X.cols
    sums = [0.0] * m
    counts = [0] * m
    for i in range(n):
        for j in range(m):
            v = X._data[i][j]
            if v == v:
                sums[j] += v
                counts[j] += 1
    means = [(sums[j] / counts[j]) if counts[j] > 0 else 0.0 for j in range(m)]
    filled = []
    for i in range(n):
        row = []
        for j in range(m):
            v = X._data[i][j]
            row.append(v if v == v else means[j])
        filled.append(row)
    return Matrix(filled)


# Шум и сравнение
def add_noise_and_compare(X: Matrix, noise_level: float = 0.1):
    X_centered, _ = center_data(X)
    C = covariance_matrix(X_centered)
    eigenvalues = find_eigenvalues(C)
    k = auto_select_k(eigenvalues)
    X_proj_orig, ratio_orig, *_ = pca_qr(X, k)
    n, m = X.rows, X.cols
    means = X.mean(axis=0)._data[0]
    stds = []
    for j in range(m):
        col = [X._data[i][j] for i in range(n)]
        var = sum((v - means[j])**2 for v in col) / n
        stds.append(math.sqrt(var))
    noisy_data = []
    for i in range(n):
        row = []
        for j in range(m):
            noise = random.gauss(0, stds[j] * noise_level)
            row.append(X._data[i][j] + noise)
        noisy_data.append(row)
    X_noisy = Matrix(noisy_data)
    X_proj_noisy, ratio_noisy, *_ = pca_qr(X_noisy, k)
    return {
        'k': k,
        'ratio_orig': ratio_orig,
        'ratio_noisy': ratio_noisy,
        'X_proj_orig': X_proj_orig,
        'X_proj_noisy': X_proj_noisy,
    }


# Датасет и PCA
def apply_pca_to_dataset(dataset_name: str, k: int):
    name = dataset_name.lower()
    if name == 'iris':
        data = load_iris()
        X_np = data.data
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")
    X = Matrix(X_np.tolist())
    X_proj, explained_ratio, *_ = pca_qr(X, k)
    return X_proj, explained_ratio

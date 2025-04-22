from algorithms_pca.matrix import Matrix, DimensionError


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


def auto_select_k(eigenvalues: list[float], threshold: float = 0.95) -> int:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")

    vals = sorted(eigenvalues, reverse=True)
    total = sum(vals)
    if total <= 0:
        return 0

    cum = 0.0
    for idx, v in enumerate(vals, start=1):
        cum += v
        if cum / total >= threshold:
            return idx

    return len(vals)


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

import copy

from algorithms_pca.matrix import Matrix, DimensionError


def gauss_solver(A: 'Matrix', b: 'Matrix') -> list['Matrix']:
    """
    Solves the linear system A x = b using Gaussian elimination with partial pivoting (RREF).

    :param A: Coefficient matrix (n x n)
    :param b: RHS column matrix (n x 1)
    :return: List of Matrix. Unique solution -> [x]; Infinite solutions -> [particular, basis...]
    :raises ValueError: on inconsistent system.
    """
    if not isinstance(A, Matrix) or not isinstance(b, Matrix):
        raise DimensionError("A and b must be Matrix instances.")
    n, m = A.rows, A.cols
    if not A.is_square() or b.rows != n or b.cols != 1:
        raise DimensionError("Incompatible dimensions for solving.")

    M = [list(map(float, A._data[i])) + [float(b._data[i][0])] for i in range(n)]
    eps = 1e-12
    row = 0
    pivot_cols: list[int] = []

    for col in range(m):
        pivot_row = max(range(row, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < eps:
            continue
        M[row], M[pivot_row] = M[pivot_row], M[row]
        pivot_val = M[row][col]
        M[row] = [val / pivot_val for val in M[row]]
        for r in range(n):
            if r != row and abs(M[r][col]) > eps:
                factor = M[r][col]
                M[r] = [M[r][c] - factor * M[row][c] for c in range(m + 1)]
        pivot_cols.append(col)
        row += 1
        if row == n:
            break

    for r in range(row, n):
        if abs(M[r][m]) > eps and all(abs(M[r][c]) < eps for c in range(m)):
            raise ValueError("Inconsistent system.")

    if len(pivot_cols) == m:
        x = [0.0] * m
        for i, col in enumerate(pivot_cols):
            x[col] = M[i][m]
        return [Matrix([[xi] for xi in x])]

    free_cols = [c for c in range(m) if c not in pivot_cols]

    x_part = [0.0] * m
    for i, col in enumerate(pivot_cols):
        x_part[col] = M[i][m]
    particular = Matrix([[v] for v in x_part])

    basis: list[Matrix] = []
    for free in free_cols:
        vec = [0.0] * m
        vec[free] = 1.0
        for i, col in enumerate(pivot_cols):
            vec[col] = -M[i][free]
        basis.append(Matrix([[v] for v in vec]))

    return [particular] + basis

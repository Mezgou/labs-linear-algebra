from algorithms_pca.matrix import Matrix, DimensionError
from algorithms_pca.linear_solver import gauss_solver

def find_eigenvalues(C: 'Matrix', tol: float = 1e-6) -> list[float]:
    if not isinstance(C, Matrix):
        raise TypeError("C must be a Matrix instance.")
    if not C.is_square():
        raise ValueError("Matrix must be square to compute eigenvalues.")

    n = C.rows

    r = max(sum(abs(C._data[i][j]) for j in range(n) if j != i) for i in range(n))
    d_min = min(C._data[i][i] for i in range(n))
    d_max = max(C._data[i][i] for i in range(n))
    min_lambda = d_min - r
    max_lambda = d_max + r

    def f(lmbd: float) -> float:
        I = Matrix([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])
        M = C - lmbd * I
        return M.determinant()

    num_intervals = abs(max_lambda + min_lambda) + 100
    xs = [min_lambda + i * (max_lambda - min_lambda) / num_intervals for i in range(num_intervals + 1)]

    eigenvalues = []
    for i in range(num_intervals):
        a = xs[i]
        b = xs[i + 1]
        f_a = f(a)
        f_b = f(b)

        if f_a * f_b <= 0:
            left, right = a, b

            while right - left > tol:
                mid = (left + right) / 2
                f_mid = f(mid)
                if f_a * f_mid <= 0:
                    right = mid
                    f_b = f_mid
                else:
                    left = mid
                    f_a = f_mid
            eigen = (left + right) / 2
            eigenvalues.append(eigen)

    eigenvalues = list(sorted(set(round(ev, 5) for ev in eigenvalues)))
    return eigenvalues


def find_eigenvectors(C: Matrix, eigenvalues: list[float]) -> list[Matrix]:
    if not isinstance(C, Matrix):
        raise DimensionError("C must be a Matrix instance.")
    if not C.is_square():
        raise DimensionError("Eigenvectors defined only for square matrices.")

    m = C.rows
    I = Matrix([[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)])

    eigenvecs: list[Matrix] = []
    for lam in eigenvalues:
        M = C - lam * I
        zero = Matrix([[0.0] for _ in range(m)])

        sols = gauss_solver(M, zero)

        basis = sols[1:] if len(sols) > 1 else [sols[0]]
        eigenvecs.extend(basis)

    return eigenvecs

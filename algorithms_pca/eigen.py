from typing import List

from algorithms_pca.linear_solver import gauss_solver
from algorithms_pca.matrix import Matrix


# Поиск собственных значений методом бисекции по характеристическому многочлену
def find_eigenvalues(C: Matrix, tol: float = 1e-6) -> List[float]:
    n = C.rows
    # Оценка диапазона значений: Gershgorin
    r = max(
        sum(abs(C._data[i][j]) for j in range(n) if j != i) for i in range(n)
    )
    d_min = min(C._data[i][i] for i in range(n))
    d_max = max(C._data[i][i] for i in range(n))
    lower = d_min - r
    upper = d_max + r

    def det_diff(lmbd: float) -> float:
        # det(C - λI)
        return (C - Matrix.identity(n) * lmbd).determinant()

    # Разбиваем отрезок [lower, upper] на равные части и ищем смену знака
    intervals = 10000
    xs = [lower + i * (upper - lower) / intervals for i in range(intervals + 1)]
    roots: List[float] = []

    for i in range(intervals):
        a, b = xs[i], xs[i + 1]
        fa, fb = det_diff(a), det_diff(b)
        if fa * fb <= 0:
            left, right = a, b
            while right - left > tol:
                mid = (left + right) / 2
                fm = det_diff(mid)
                if fa * fm <= 0:
                    right, fb = mid, fm
                else:
                    left, fa = mid, fm
            roots.append((left + right) / 2)

    # Убираем дубли и округляем
    return sorted({round(val, 5) for val in roots})


def find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]:
    n = C.rows
    I = Matrix.identity(n)
    eigenvectors = []
    # Для каждого λ решаем (C - λI)x = 0
    for eigenvalue in eigenvalues:
        M = C - I * eigenvalue
        eigenvector = gauss_solver(M, Matrix([[0.0] for _ in range(M.rows)]))
        eigenvectors.append(eigenvector[1:])
    return eigenvectors

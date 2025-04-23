import copy
import math

from typing import Union, Tuple, List


class MatrixError(Exception):
    pass


class DimensionError(MatrixError):
    pass


class SingularMatrixError(MatrixError):
    pass


class Matrix:
    def __init__(self, data: List[List[float]]):
        if not data or any(not isinstance(row, list) for row in data):
            raise DimensionError("Data must be a non-empty list of lists.")
        if len({len(row) for row in data}) != 1:
            raise DimensionError("All rows must have the same length.")
        self._data = copy.deepcopy(data)
        self.rows = len(data)
        self.cols = len(data[0])

    def __repr__(self) -> str:
        return f"Matrix({self._data})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Matrix) and self._data == other._data

    def _check_same_dimensions(self, other: 'Matrix'):
        if self.rows != other.rows or self.cols != other.cols:
            raise DimensionError("Matrices must have the same dimensions.")

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, Matrix):
            raise DimensionError("Can only add another Matrix.")
        self._check_same_dimensions(other)
        return Matrix([[self._data[i][j] + other._data[i][j]
                       for j in range(self.cols)]
                      for i in range(self.rows)])

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, Matrix):
            raise DimensionError("Can only subtract another Matrix.")
        self._check_same_dimensions(other)
        return Matrix([[self._data[i][j] - other._data[i][j]
                       for j in range(self.cols)]
                      for i in range(self.rows)])

    def __mul__(self, other: Union['Matrix', float, int]) -> 'Matrix':
        if isinstance(other, Matrix):
            self._check_same_dimensions(other)
            return Matrix([[self._data[i][j] * other._data[i][j]
                           for j in range(self.cols)]
                          for i in range(self.rows)])
        if isinstance(other, (int, float)):
            return Matrix([[self._data[i][j] * other for j in range(self.cols)]
                          for i in range(self.rows)])
        raise DimensionError("Unsupported multiplication type.")

    __rmul__ = __mul__

    def __truediv__(self, other: Union['Matrix', float, int]) -> 'Matrix':
        if isinstance(other, Matrix):
            self._check_same_dimensions(other)
            result = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    denom = other._data[i][j]
                    if denom == 0:
                        raise DimensionError("Division by zero in matrix division.")
                    row.append(self._data[i][j] / denom)
                result.append(row)
            return Matrix(result)
        if isinstance(other, (int, float)):
            if other == 0:
                raise DimensionError("Division by zero.")
            return Matrix([[self._data[i][j] / other for j in range(self.cols)]
                          for i in range(self.rows)])
        raise DimensionError("Unsupported division type.")

    def matmul(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, Matrix):
            raise DimensionError("Can only matrix-multiply another Matrix.")
        if self.cols != other.rows:
            raise DimensionError("Incompatible dimensions for matrix multiplication.")
        result = [[0.0 for _ in range(other.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for k in range(self.cols):
                if self._data[i][k] == 0:
                    continue
                v = self._data[i][k]
                for j in range(other.cols):
                    result[i][j] += v * other._data[k][j]
        return Matrix(result)

    def transpose(self) -> 'Matrix':
        return Matrix([[self._data[j][i] for j in range(self.rows)]
                      for i in range(self.cols)])

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        return self.matmul(other)

    @property
    def T(self) -> 'Matrix':
        return self.transpose()

    @staticmethod
    def identity(n: int) -> 'Matrix':
        return Matrix([[1.0 if i == j else 0.0 for j in range(n)]
                       for i in range(n)])

    def __getitem__(self, key: Union[int, slice, Tuple[Union[int, slice], Union[int, slice]]]):
        if isinstance(key, tuple):
            row_key, col_key = key
            if isinstance(row_key, int) and isinstance(col_key, int):
                return self._data[row_key][col_key]
            rows = (range(*row_key.indices(self.rows))
                    if isinstance(row_key, slice) else [row_key])
            cols = (range(*col_key.indices(self.cols))
                    if isinstance(col_key, slice) else [col_key])
            return Matrix([[self._data[i][j] for j in cols] for i in rows])
        if isinstance(key, int):
            return self._data[key][:]
        if isinstance(key, slice):
            rows = range(*key.indices(self.rows))
            return Matrix([self._data[i][:] for i in rows])
        raise IndexError("Invalid index.")

    def __setitem__(self, key: Union[int, Tuple[int, int]], value):
        if isinstance(key, tuple):
            i, j = key
            self._data[i][j] = value
        elif isinstance(key, int):
            if not isinstance(value, list) or len(value) != self.cols:
                raise ValueError("Row assignment must match column count.")
            self._data[key] = value[:]
        else:
            raise IndexError("Invalid index.")

    def is_square(self) -> bool:
        return self.rows == self.cols

    def trace(self) -> float:
        if not self.is_square():
            raise DimensionError("Trace defined only for square matrices.")
        return sum(self._data[i][i] for i in range(self.rows))

    def is_symmetric(self) -> bool:
        if not self.is_square():
            return False
        return all(self._data[i][j] == self._data[j][i]
                   for i in range(self.rows) for j in range(i, self.cols))

    def submatrix(self, exclude_row: int, exclude_col: int) -> 'Matrix':
        if not (0 <= exclude_row < self.rows and 0 <= exclude_col < self.cols):
            raise IndexError("Index out of bounds.")
        data = [[self._data[r][c] for c in range(self.cols) if c != exclude_col]
                for r in range(self.rows) if r != exclude_row]
        return Matrix(data)

    def determinant(self) -> float:
        if not self.is_square():
            raise DimensionError("Determinant defined only for square matrices.")
        A = copy.deepcopy(self._data)
        n = self.rows
        det = 1.0
        for i in range(n):
            pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
            if abs(A[pivot][i]) < 1e-12:
                return 0.0
            if i != pivot:
                A[i], A[pivot] = A[pivot], A[i]
                det *= -1
            det *= A[i][i]
            for j in range(i+1, n):
                factor = A[j][i] / A[i][i]
                for k in range(i, n):
                    A[j][k] -= factor * A[i][k]
        return det

    def lu(self) -> Tuple['Matrix', 'Matrix', 'Matrix']:
        if not self.is_square():
            raise DimensionError("LU decomposition requires square matrix.")
        n = self.rows
        A = copy.deepcopy(self._data)
        P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        L = [[0.0]*n for _ in range(n)]
        U = [[0.0]*n for _ in range(n)]
        for i in range(n):
            pivot = max(range(i, n), key=lambda r: abs(A[r][i]))
            if abs(A[pivot][i]) < 1e-12:
                raise SingularMatrixError("Matrix is singular.")
            A[i], A[pivot] = A[pivot], A[i]
            P[i], P[pivot] = P[pivot], P[i]
            L[i][i] = 1.0
            for j in range(i, n):
                U[i][j] = A[i][j]
            for k in range(i+1, n):
                factor = A[k][i] / U[i][i]
                L[k][i] = factor
                for j in range(i, n):
                    A[k][j] -= factor * U[i][j]
        return Matrix(P), Matrix(L), Matrix(U)

    def inverse(self) -> 'Matrix':
        if not self.is_square():
            raise DimensionError("Inverse requires square matrix.")
        n = self.rows
        aug = [self._data[i] + [1.0 if i == j else 0.0 for j in range(n)]
               for i in range(n)]
        for i in range(n):
            pivot = max(range(i, n), key=lambda r: abs(aug[r][i]))
            if abs(aug[pivot][i]) < 1e-12:
                raise SingularMatrixError("Matrix is singular.")
            aug[i], aug[pivot] = aug[pivot], aug[i]
            factor = aug[i][i]
            aug[i] = [v / factor for v in aug[i]]
            for r in range(n):
                if r != i:
                    factor = aug[r][i]
                    aug[r] = [aug[r][c] - factor * aug[i][c] for c in range(2*n)]
        return Matrix([row[n:] for row in aug])

    def rank(self) -> int:
        M = [row[:] for row in self._data]
        m, n = self.rows, self.cols
        rank = 0
        for col in range(n):
            pivot = next((r for r in range(rank, m) if abs(M[r][col]) > 1e-12), None)
            if pivot is None:
                continue
            M[rank], M[pivot] = M[pivot], M[rank]
            lv = M[rank][col]
            M[rank] = [val / lv for val in M[rank]]
            for r in range(m):
                if r != rank and abs(M[r][col]) > 1e-12:
                    factor = M[r][col]
                    M[r] = [M[r][c] - factor * M[rank][c] for c in range(n)]
            rank += 1
            if rank == m:
                break
        return rank

    def solve(self, b: 'Matrix') -> 'Matrix':
        if not isinstance(b, Matrix) or not self.is_square() or b.cols != 1 or b.rows != self.rows:
            raise DimensionError("Incompatible dimensions for solving.")
        P, L, U = self.lu()
        Pb = P.matmul(b)
        n = self.rows
        y = [0.0]*n
        for i in range(n):
            y[i] = Pb._data[i][0] - sum(L._data[i][j]*y[j] for j in range(i))
        x = [0.0]*n
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - sum(U._data[i][j]*x[j] for j in range(i+1, n))) / U._data[i][i]
        return Matrix([[xi] for xi in x])

    def sum(self, axis: int=None) -> Union[float,'Matrix']:
        if axis is None:
            return sum(sum(row) for row in self._data)
        if axis == 0:
            return Matrix([[sum(self._data[r][c] for r in range(self.rows))
                            for c in range(self.cols)]])
        if axis == 1:
            return Matrix([[sum(row) for row in self._data]])
        raise DimensionError("Axis must be 0, 1, or None.")

    def mean(self, axis: int=None) -> Union[float,'Matrix']:
        if axis is None:
            return self.sum() / (self.rows*self.cols)
        if axis in (0, 1):
            denom = self.rows if axis==0 else self.cols
            return self.sum(axis) / denom
        raise DimensionError("Axis must be 0, 1, or None.")

    def qr_decomposition(self) -> Tuple['Matrix','Matrix']:
        A = Matrix(copy.deepcopy(self._data))
        m, n = self.rows, self.cols
        Q = [[0.0]*n for _ in range(m)]
        R = [[0.0]*n for _ in range(n)]
        for j in range(n):
            v = [A._data[i][j] for i in range(m)]
            for i in range(j):
                qi = [Q[k][i] for k in range(m)]
                R[i][j] = sum(A._data[k][j]*qi[k] for k in range(m))
                for k in range(m): v[k] -= R[i][j]*qi[k]
            norm_v = math.sqrt(sum(val*val for val in v))
            if norm_v < 1e-20:
                raise SingularMatrixError("Matrix is degenerate.")
            R[j][j] = norm_v
            for i in range(m): Q[i][j] = v[i]/norm_v
        return Matrix(Q), Matrix(R)

    def solve_homogeneous(self, tol: float=1e-12, round_decimals: int=6) -> 'Matrix':
        n, m = self.rows, self.cols
        M = [row[:] for row in self._data]
        pivot_row = 0
        pivots = []
        for col in range(m):
            if pivot_row >= n: break
            row = next((r for r in range(pivot_row, n) if abs(M[r][col])>tol), None)
            if row is None: continue
            M[pivot_row], M[row] = M[row], M[pivot_row]
            factor = M[pivot_row][col]
            M[pivot_row] = [val/factor for val in M[pivot_row]]
            pivots.append(col)
            for r in range(pivot_row+1, n):
                factor = M[r][col]
                M[r] = [M[r][c]-factor*M[pivot_row][c] for c in range(m)]
            pivot_row += 1
        free = [c for c in range(m) if c not in pivots]
        x = [0.0]*m
        if free: x[free[0]] = 1.0
        else: x[0] = 1.0
        for i in reversed(range(len(pivots))):
            c = pivots[i]
            s = sum(M[i][j]*x[j] for j in range(c+1, m))
            x[c] = -s
        x = [0.0 if abs(val)<tol else round(val/ (math.sqrt(sum(v*v for v in x))), round_decimals) for val in x]
        return Matrix([[val] for val in x])

    def qr_eigen(self, max_iter: int=1000, tol: float=1e-20) -> Tuple[List[float],'Matrix']:
        A_k = Matrix(copy.deepcopy(self._data))
        Q_total = Matrix.identity(self.rows)
        for _ in range(max_iter):
            Q, R = A_k.qr_decomposition()
            A_k = R.matmul(Q)
            Q_total = Q_total.matmul(Q)
            off = sum(A_k._data[i][j]**2 for i in range(self.rows) for j in range(self.cols) if i!=j)
            if math.sqrt(off) < tol: break
        return [A_k._data[i][i] for i in range(self.rows)], Q_total

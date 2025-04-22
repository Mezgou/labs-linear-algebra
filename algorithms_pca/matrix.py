import copy


class MatrixError(Exception):
    """Base class for Matrix exceptions."""
    pass


class DimensionError(MatrixError):
    """Raised when matrix dimensions do not align for an operation or invalid input."""
    pass


class SingularMatrixError(MatrixError):
    """Raised when an operation requires a non-singular matrix but the matrix is singular."""
    pass


class Matrix:
    def __init__(self, data):
        if not data or not all(isinstance(row, list) for row in data):
            raise DimensionError("Data must be a non-empty list of lists.")
        if len({len(row) for row in data}) != 1:
            raise DimensionError("All rows must have the same length.")
        self._data = copy.deepcopy(data)
        self.rows = len(data)
        self.cols = len(data[0])

    def __repr__(self):
        return f"Matrix({self._data})"

    def __eq__(self, other):
        return isinstance(other, Matrix) and self._data == other._data

    def _check_same_dim(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise DimensionError("Matrices must have the same dimensions.")

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise DimensionError("Can only add another Matrix.")
        self._check_same_dim(other)
        data = [
            [self._data[i][j] + other._data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(data)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise DimensionError("Can only subtract another Matrix.")
        self._check_same_dim(other)
        data = [
            [self._data[i][j] - other._data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(data)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            self._check_same_dim(other)
            data = [
                [self._data[i][j] * other._data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
            return Matrix(data)
        if isinstance(other, (int, float)):
            data = [[self._data[i][j] * other for j in range(self.cols)]
                    for i in range(self.rows)]
            return Matrix(data)
        raise DimensionError("Unsupported multiplication type.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def matmul(self, other):
        if not isinstance(other, Matrix):
            raise DimensionError("Can only matrix-multiply another Matrix.")
        if self.cols != other.rows:
            raise DimensionError("Incompatible dimensions for matmul.")
        result = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                s = 0
                for k in range(self.cols):
                    s += self._data[i][k] * other._data[k][j]
                row.append(s)
            result.append(row)
        return Matrix(result)

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            self._check_same_dim(other)
            data = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    denom = other._data[i][j]
                    if denom == 0:
                        raise DimensionError("Division by zero in matrix division.")
                    row.append(self._data[i][j] / denom)
                data.append(row)
            return Matrix(data)
        if isinstance(other, (int, float)):
            if other == 0:
                raise DimensionError("Division by zero.")
            data = [[self._data[i][j] / other for j in range(self.cols)]
                    for i in range(self.rows)]
            return Matrix(data)
        raise DimensionError("Unsupported division type.")

    def transpose(self):
        data = [[self._data[j][i] for j in range(self.rows)]
                for i in range(self.cols)]
        return Matrix(data)

    @property
    def T(self):
        return self.transpose()

    def is_square(self):
        return self.rows == self.cols

    def trace(self):
        if not self.is_square():
            raise DimensionError("Trace defined only for square matrices.")
        return sum(self._data[i][i] for i in range(self.rows))

    def is_symmetric(self):
        if not self.is_square():
            return False
        return all(
            self._data[i][j] == self._data[j][i]
            for i in range(self.rows) for j in range(i, self.cols)
        )

    def submatrix(self, i, j):
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise IndexError("Index out of bounds.")
        data = [
            [self._data[r][c] for c in range(self.cols) if c != j]
            for r in range(self.rows) if r != i
        ]
        return Matrix(data)

    def minor(self, i, j):
        return self.submatrix(i, j).determinant()

    def _lu_decomposition(self):
        if not self.is_square():
            raise DimensionError("LU decomposition requires square matrix.")
        n = self.rows
        A = copy.deepcopy(self._data)
        P = [[float(i == j) for j in range(n)] for i in range(n)]
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]

        for i in range(n):
            pivot_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            if abs(A[pivot_row][i]) < 1e-12:
                raise SingularMatrixError("Matrix is singular.")
            A[i], A[pivot_row] = A[pivot_row], A[i]
            P[i], P[pivot_row] = P[pivot_row], P[i]
            L[i], L[pivot_row] = L[pivot_row], L[i]

            for j in range(i, n):
                U[i][j] = A[i][j]

            for k in range(i + 1, n):
                factor = A[k][i] / U[i][i]
                L[k][i] = factor
                for j in range(i, n):
                    A[k][j] -= factor * U[i][j]

        for i in range(n):
            L[i][i] = 1.0

        return Matrix(P), Matrix(L), Matrix(U)

    def lu(self):
        return self._lu_decomposition()

    def determinant(self):
        if not self.is_square():
            raise DimensionError("Determinant is defined only for square matrices.")

        A = copy.deepcopy(self._data)
        n = self.rows
        det = 1
        for i in range(n):
            max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            if abs(A[max_row][i]) < 1e-12:
                return 0

            if i != max_row:
                A[i], A[max_row] = A[max_row], A[i]
                det *= -1

            det *= A[i][i]
            for j in range(i + 1, n):
                factor = A[j][i] / A[i][i]
                for k in range(i, n):
                    A[j][k] -= factor * A[i][k]

        return det

    def inverse(self):
        if not self.is_square():
            raise DimensionError("Inverse requires square matrix.")
        n = self.rows
        aug = [self._data[i] + [float(i == j) for j in range(n)]
               for i in range(n)]
        for i in range(n):
            pivot_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
            if abs(aug[pivot_row][i]) < 1e-12:
                raise SingularMatrixError("Matrix is singular.")
            aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

            piv = aug[i][i]
            aug[i] = [v / piv for v in aug[i]]
            for r in range(n):
                if r != i:
                    factor = aug[r][i]
                    aug[r] = [aug[r][c] - factor * aug[i][c]
                              for c in range(2 * n)]

        inv_data = [row[n:] for row in aug]
        return Matrix(inv_data)

    def rank(self):
        M = [row[:] for row in self._data]
        m, n = self.rows, self.cols
        rank = 0
        for col in range(n):
            pivot = None
            for r in range(rank, m):
                if abs(M[r][col]) > 1e-12:
                    pivot = r
                    break
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

    def solve(self, b):
        if not isinstance(b, Matrix):
            raise DimensionError("Right-hand side must be a Matrix.")
        if not self.is_square() or b.cols != 1 or b.rows != self.rows:
            raise DimensionError("Incompatible dimensions for solving.")
        P, L, U = self._lu_decomposition()
        Pb = P.matmul(b)
        n = self.rows
        y = [0.0] * n
        for i in range(n):
            y[i] = Pb._data[i][0] - sum(L._data[i][j] * y[j] for j in range(i))
        x = [0.0] * n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(U._data[i][j] * x[j] for j in range(i + 1, n))) / U._data[i][i]
        return Matrix([[xi] for xi in x])

    def sum(self, axis=None):
        if axis is None:
            return sum(sum(row) for row in self._data)
        if axis == 0:
            return Matrix([[sum(self._data[r][c] for r in range(self.rows)) for c in range(self.cols)]])
        if axis == 1:
            return Matrix([[sum(row) for row in self._data]])
        raise DimensionError("Axis must be 0, 1, or None.")

    def mean(self, axis=None):
        if axis is None:
            return self.sum() / (self.rows * self.cols)
        if axis in (0, 1):
            return self.sum(axis) / (self.rows if axis == 0 else self.cols)
        raise DimensionError("Axis must be 0, 1, or None.")

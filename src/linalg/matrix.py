import math

from typing import List, Union


class Matrix:
    def __init__(self, matrix: List[List[float]]) -> None:
        if not matrix:
            raise ValueError("The matrix should not be empty")
        self.row_len = len(matrix[0])
        self.col_len = len(matrix)
        for row in matrix:
            if len(row) != self.row_len:
                raise ValueError("All rows of the matrix must have the same length")
        self.matrix = matrix

    def __str__(self) -> str:
        return "\n".join("\t".join(map(str, row)) for row in self.matrix)

    def __repr__(self) -> str:
        return f"Matrix({self.matrix})"

    def __add__(self, other: "Matrix") -> "Matrix":
        if len(self.matrix) != len(other.matrix) or len(self.matrix[0]) != len(other.matrix[0]):
            raise ValueError("Matrix must have the same length")
        result = [
            [a + b for a, b in zip(row_self, row_other)]
            for row_self, row_other in zip(self.matrix, other.matrix)
        ]
        return Matrix(result)

    def __sub__(self, other: "Matrix") -> "Matrix":
        if len(self.matrix) != len(other.matrix) or len(self.matrix[0]) != len(other.matrix[0]):
            raise ValueError("Matrix must have the same length")
        result = [
            [a - b for a, b in zip(row_self, row_other)]
            for row_self, row_other in zip(self.matrix, other.matrix)
        ]
        return Matrix(result)

    def __mul__(self, other: Union["Matrix", int, float]) -> "Matrix":
        # Matrix * scalar
        if isinstance(other, (int, float)):
            result = [
                [a * other for a in row]
                for row in self.matrix
            ]
            return Matrix(result)

        # Matrix * Matrix
        if len(self.matrix[0]) != len(other.matrix):
            raise ValueError("The number of columns of the first matrix should be equal to the number of rows of the second")
        result = []
        for i in range(len(self.matrix)):
            result_row = []
            for j in range(len(other.matrix[0])):
                s = 0
                for k in range(len(self.matrix[0])):
                    s += self.matrix[i][k] * other.matrix[k][j]
                result_row.append(s)
            result.append(result_row)
        return Matrix(result)

    def __rmul__(self, other: Union[int, float]) -> 'Matrix':
        return self.__mul__(other)

    def transpose(self) -> "Matrix":
        rows = len(self.matrix)
        cols = len(self.matrix[0])
        result = [[self.matrix[i][j] for i in range(rows)] for j in range(cols)]
        return Matrix(result)

    def determinant(self) -> float:
        if len(self.matrix) != len(self.matrix[0]):
            raise ValueError("The determinant is defined only for square matrices")
        n = len(self.matrix)
        if n == 1:
            return self.matrix[0][0]
        if n == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        det = 0
        for c in range(n):
            minor = self._minor(0, c)
            det += ((-1) ** c) * self.matrix[0][c] * minor.determinant()
        return det

    def _minor(self, i: int, j: int) -> "Matrix":
        minor = [row[:j] + row[j + 1:] for k, row in enumerate(self.matrix) if k != i]
        return Matrix(minor)

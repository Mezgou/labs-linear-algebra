import random

from types import NotImplementedType
from typing import Sequence, Union, Optional, List
from vector import Vector


class Matrix:
    def __init__(self, data: Sequence[Sequence[Union[int, float]]]) -> None:
        self.data: List[List[Union[int, float]]] = [list(row) for row in data]
        self.rows: int = len(self.data)
        self.cols: int = len(self.data[0]) if self.rows > 0 else 0
        if any(len(row) != self.cols for row in self.data):
            raise ValueError("All rows must have the same length")

    def __repr__(self) -> str:
        return f"Matrix({self.data})"

    def __str__(self) -> str:
        return '\n'.join([str(row) for row in self.data])

    def __getitem__(self, idx: int) -> List[Union[int, float]]:
        return self.data[idx]

    def __add__(self, other: Union['Matrix', int, float]) -> Union['Matrix', NotImplementedType]:
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrices must have the same dimensions for addition")
            result: List[List[Union[int, float]]] = [
                [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
            return Matrix(result)
        elif isinstance(other, (int, float)):
            result: List[List[Union[int, float]]] = [
                [self.data[i][j] + other for j in range(self.cols)]
                for i in range(self.rows)
            ]
            return Matrix(result)
        else:
            return NotImplemented

    def __sub__(self, other: Union['Matrix', int, float]) -> Union['Matrix', NotImplementedType]:
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrices must have the same dimensions for subtraction")
            result: List[List[Union[int, float]]] = [
                [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
            return Matrix(result)
        elif isinstance(other, (int, float)):
            result: List[List[Union[int, float]]] = [
                [self.data[i][j] - other for j in range(self.cols)]
                for i in range(self.rows)
            ]
            return Matrix(result)
        else:
            return NotImplemented

    def __mul__(self, other: Union[int, float, Vector, 'Matrix']) -> Union['Matrix', Vector, NotImplementedType]:
        if isinstance(other, (int, float)):
            result: List[List[Union[int, float]]] = [
                [self.data[i][j] * other for j in range(self.cols)]
                for i in range(self.rows)
            ]
            return Matrix(result)
        elif isinstance(other, Vector):
            if self.cols != len(other):
                raise ValueError("Matrix columns must match vector size for multiplication")
            result_vector: List[Union[int, float]] = [
                sum(self.data[i][j] * other[j] for j in range(self.cols))
                for i in range(self.rows)
            ]
            return Vector(result_vector)
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Matrix dimensions incompatible for multiplication")
            result: List[List[Union[int, float]]] = [
                [
                    sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                    for j in range(other.cols)
                ]
                for i in range(self.rows)
            ]
            return Matrix(result)
        else:
            return NotImplemented

    def __rmul__(self, other: Union[int, float]) -> 'Matrix':
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def transpose(self) -> 'Matrix':
        transposed: List[List[Union[int, float]]] = [
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ]
        return Matrix(transposed)

    def determinant(self) -> float:
        if self.rows != self.cols:
            raise ValueError("Determinant can only be calculated for square matrices")
        if self.rows == 1:
            return float(self.data[0][0])
        if self.rows == 2:
            return float(
                self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
            )
        det: float = 0.0
        for j in range(self.cols):
            submatrix: List[List[Union[int, float]]] = [
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ]
            cofactor: float = self.data[0][j] * Matrix(submatrix).determinant()
            det += cofactor if j % 2 == 0 else -cofactor
        return det

    def inverse(self) -> 'Matrix':
        if self.rows != self.cols:
            raise ValueError("Only square matrices can be inverted")
        det: float = self.determinant()
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular and cannot be inverted")
        if self.rows == 1:
            return Matrix([[1.0 / self.data[0][0]]])
        if self.rows == 2:
            a, b = self.data[0][0], self.data[0][1]
            c, d = self.data[1][0], self.data[1][1]
            factor: float = 1.0 / det
            return Matrix([
                [factor * d, -factor * b],
                [-factor * c, factor * a]
            ])
        cofactors: List[List[float]] = []
        for i in range(self.rows):
            cofactor_row: List[float] = []
            for j in range(self.cols):
                submatrix: List[List[Union[int, float]]] = [
                    [self.data[r][c] for c in range(self.cols) if c != j]
                    for r in range(self.rows) if r != i
                ]
                sign: int = 1 if (i + j) % 2 == 0 else -1
                cofactor: float = sign * Matrix(submatrix).determinant()
                cofactor_row.append(cofactor)
            cofactors.append(cofactor_row)
        adjugate: Matrix = Matrix(cofactors).transpose()
        return adjugate * (1.0 / det)

    @classmethod
    def identity(cls, size: int) -> 'Matrix':
        data: List[List[float]] = [
            [1.0 if i == j else 0.0 for j in range(size)]
            for i in range(size)
        ]
        return cls(data)

    @classmethod
    def zeros(cls, rows: int, cols: Optional[int] = None) -> 'Matrix':
        if cols is None:
            cols = rows
        data: List[List[float]] = [
            [0.0 for _ in range(cols)] for _ in range(rows)
        ]
        return cls(data)

    @classmethod
    def random(
        cls,
        rows: int,
        cols: Optional[int] = None,
        min_val: Union[int, float] = 0,
        max_val: Union[int, float] = 1
    ) -> 'Matrix':
        if cols is None:
            cols = rows
        if isinstance(min_val, int) and isinstance(max_val, int):
            data: List[List[Union[int, float]]] = [
                [random.randint(min_val, max_val) for _ in range(cols)]
                for _ in range(rows)
            ]
        else:
            data: List[List[float]] = [
                [random.uniform(min_val, max_val) for _ in range(cols)]
                for _ in range(rows)
            ]
        return cls(data)
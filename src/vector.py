from types import NotImplementedType
from typing import Sequence, Union, List

class Vector:
    def __init__(self, data: Sequence[Union[int, float]]) -> None:
        self.data: List[Union[int, float]] = list(data)
        self.size: int = len(data)

    def __repr__(self) -> str:
        return f"Vector({self.data})"

    def __str__(self) -> str:
        return str(self.data)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Union[int, float]:
        return self.data[idx]

    def __setitem__(self, idx: int, value: Union[int, float]) -> None:
        self.data[idx] = value

    def __add__(self, other: Union['Vector', int, float]) -> Union['Vector', NotImplementedType]:
        if isinstance(other, Vector):
            if self.size != other.size:
                raise ValueError("Vectors must be the same size for addition")
            return Vector([self.data[i] + other[i] for i in range(self.size)])
        elif isinstance(other, (int, float)):
            return Vector([x + other for x in self.data])
        else:
            return NotImplemented

    def __sub__(self, other: Union['Vector', int, float]) -> Union['Vector', NotImplementedType]:
        if isinstance(other, Vector):
            if self.size != len(other):
                raise ValueError("Vectors must be the same size for subtraction")
            return Vector([self.data[i] - other[i] for i in range(self.size)])
        elif isinstance(other, (int, float)):
            return Vector([x - other for x in self.data])
        else:
            return NotImplemented

    def __mul__(self, other: Union[int, float, 'Vector']) -> Union['Vector', float, NotImplementedType]:
        if isinstance(other, (int, float)):
            return Vector([x * other for x in self.data])
        elif isinstance(other, Vector):
            if self.size != len(other):
                raise ValueError("Vectors must be the same size for multiplication")
            return Vector([self.data[i] * other[i] for i in range(self.size)])
        else:
            return NotImplemented

    def __rmul__(self, other: Union[int, float, 'Vector']) -> Union['Vector', float, NotImplementedType]:
        return self.__mul__(other)

    def __neg__(self) -> 'Vector':
        return Vector([-x for x in self.data])

    def dot(self, other: 'Vector') -> Union[int, float]:
        if not isinstance(other, Vector) or self.size != len(other):
            raise ValueError("Dot product requires vectors of the same size")
        return sum(self.data[i] * other[i] for i in range(self.size))

    def magnitude(self) -> float:
        return (sum(x * x for x in self.data)) ** 0.5

    def normalize(self) -> 'Vector':
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Vector([x / mag for x in self.data])
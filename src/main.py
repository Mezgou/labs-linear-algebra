from linalg.matrix import Matrix
from linalg.algorithms import covariance_matrix


def main():
    mat1 = Matrix([[2, 4, 8], [4, 5, 6], [7, 8, 9], [1, 2, 3]])
    mat2 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(covariance_matrix(mat1))


if __name__ == "__main__":
    main()

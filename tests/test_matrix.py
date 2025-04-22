import unittest

from algorithms_pca.matrix import Matrix, DimensionError, SingularMatrixError


class TestMatrixArithmetic(unittest.TestCase):
    def test_add_sub(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        self.assertEqual(A + B, Matrix([[6, 8], [10, 12]]))
        self.assertEqual(B - A, Matrix([[4, 4], [4, 4]]))
        with self.assertRaises(DimensionError):
            _ = A + Matrix([[1]])

    def test_mul_div(self):
        A = Matrix([[2, 4], [6, 8]])
        B = Matrix([[1, 2], [3, 4]])
        self.assertEqual(A * B, Matrix([[2, 8], [18, 32]]))
        self.assertEqual(A / 2, Matrix([[1, 2], [3, 4]]))
        with self.assertRaises(DimensionError):
            _ = A / Matrix([[0, 1], [2, 3]])


class TestMatrixProperties(unittest.TestCase):
    def test_transpose_trace(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(A.T, Matrix([[1, 4], [2, 5], [3, 6]]))
        B = Matrix([[7, 8], [9, 10]])
        self.assertEqual(B.trace(), 17)

    def test_symmetry(self):
        S = Matrix([[1, 2], [2, 1]])
        self.assertTrue(S.is_symmetric())
        self.assertFalse(Matrix([[1, 0], [1, 1]]).is_symmetric())


class TestMatrixDecomposition(unittest.TestCase):
    def test_det_inverse(self):
        A = Matrix([[4, 7], [2, 6]])
        detA = A.determinant()
        self.assertAlmostEqual(detA, 10)
        invA = A.inverse()
        I = A.matmul(invA)
        self.assertTrue(all(abs(I._data[i][j] - (1 if i == j else 0)) < 1e-6
                            for i in range(2) for j in range(2)))
        with self.assertRaises(SingularMatrixError):
            Matrix([[1, 2], [2, 4]]).inverse()

    def test_lu(self):
        A = Matrix([[2, 3], [5, 4]])
        P, L, U = A.lu()
        PLU = P.matmul(L).matmul(U)
        self.assertEqual(PLU, A)

    def test_rank(self):
        A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        self.assertEqual(A.rank(), 1)
        B = Matrix([[1, 0], [0, 1]])
        self.assertEqual(B.rank(), 2)


class TestMatrixSolve(unittest.TestCase):
    def test_solve(self):
        A = Matrix([[3, 1], [1, 2]])
        b = Matrix([[9], [8]])
        x = A.solve(b)
        self.assertTrue(all(abs(x._data[i][0] - v) < 1e-6
                            for i, v in enumerate([2, 3])))
        with self.assertRaises(DimensionError):
            A.solve(Matrix([[1, 2]]))


if __name__ == '__main__':
    unittest.main()

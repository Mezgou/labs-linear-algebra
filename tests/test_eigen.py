import unittest
from algorithms_pca.eigen import find_eigenvalues, find_eigenvectors
from algorithms_pca.matrix import Matrix


class TestEigen(unittest.TestCase):
    def assertMatrixAlmostEqual(self, A, B, tol=1e-5):
        self.assertEqual(A.rows, B.rows)
        self.assertEqual(A.cols, B.cols)
        for i in range(A.rows):
            for j in range(A.cols):
                self.assertAlmostEqual(A._data[i][j], B._data[i][j], delta=tol)

    def test_2x2_matrix(self):
        C = Matrix([[2, 1],
                    [1, 2]])
        eigenvalues = sorted(find_eigenvalues(C), key=lambda x: round(x, 5))
        self.assertEqual(len(eigenvalues), 2)
        self.assertAlmostEqual(eigenvalues[0], 1.0, delta=1e-5)
        self.assertAlmostEqual(eigenvalues[1], 3.0, delta=1e-5)

        eigenvectors = find_eigenvectors(C, eigenvalues)
        for lam, vec in zip(eigenvalues, eigenvectors):
            Cv = C.matmul(vec)
            lv = lam * vec
            self.assertMatrixAlmostEqual(Cv, lv)

    def test_identity_matrix(self):
        I = Matrix([[1, 0],
                    [0, 1]])
        eigenvalues = sorted(find_eigenvalues(I))
        self.assertEqual(len(eigenvalues), 1)
        for val in eigenvalues:
            self.assertAlmostEqual(val, 1.0, delta=1e-5)

        eigenvectors = find_eigenvectors(I, eigenvalues)
        for vec in eigenvectors:
            Iv = I.matmul(vec)
            self.assertMatrixAlmostEqual(Iv, vec)

    def test_diagonal_matrix(self):
        D = Matrix([[5, 0, 0],
                    [0, 3, 0],
                    [0, 0, 1]])
        eigenvalues = sorted(find_eigenvalues(D))
        expected = [1.0, 3.0, 5.0]
        self.assertEqual(len(eigenvalues), len(expected))
        for val, exp in zip(eigenvalues, expected):
            self.assertAlmostEqual(val, exp, delta=1e-5)

        eigenvectors = find_eigenvectors(D, eigenvalues)
        for lam, vec in zip(eigenvalues, eigenvectors):
            Dv = D.matmul(vec)
            lv = lam * vec
            self.assertMatrixAlmostEqual(Dv, lv)


if __name__ == '__main__':
    unittest.main()

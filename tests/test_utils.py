import unittest

from algorithms_pca.matrix import Matrix
from algorithms_pca.utils import reconstruction_error, auto_select_k, handle_missing_values


class TestPCAExtensions(unittest.TestCase):
    def test_reconstruction_error_zero(self):
        X = Matrix([[1, 2], [3, 4]])
        self.assertAlmostEqual(reconstruction_error(X, X), 0.0)

    def test_reconstruction_error_nonzero(self):
        X = Matrix([[1.0, 2.0], [3.0, 4.0]])
        Y = Matrix([[2.0, 0.0], [0.0, 2.0]])
        mse = ((1-2)**2 + (2-0)**2 + (3-0)**2 + (4-2)**2) / 4
        self.assertAlmostEqual(reconstruction_error(X, Y), mse)

    def test_auto_select_k(self):
        vals = [4.0, 1.0, 0.5, 0.5]
        self.assertEqual(auto_select_k(vals, 0.90), 3)
        self.assertEqual(auto_select_k(vals, 0.50), 1)
        self.assertEqual(auto_select_k(vals, 0.20), 1)
        self.assertEqual(auto_select_k(vals, 0.99), 4)
        self.assertEqual(auto_select_k(vals, 1.0), 4)

    def test_handle_missing_values(self):
        nan = float('nan')
        X = Matrix([[1.0, nan, 3.0], [nan, 2.0, nan], [4.0, nan, 6.0]])
        filled = handle_missing_values(X)
        self.assertEqual(filled, Matrix([[1.0, 2.0, 3.0], [2.5, 2.0, 4.5], [4.0, 2.0, 6.0]]))


if __name__ == '__main__':
    unittest.main()

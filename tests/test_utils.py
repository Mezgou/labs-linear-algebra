import unittest

from algorithms_pca.matrix import Matrix
from algorithms_pca.utils import reconstruction_error, auto_select_k, handle_missing_values, add_noise_and_compare, apply_pca_to_dataset


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

    def test_add_noise_and_compare_no_noise(self):
        X = Matrix([[0, 1], [1, 0], [2, 2]])
        res = add_noise_and_compare(X, noise_level=0.0)
        self.assertEqual(res['k'], res['k'])  # k определено
        self.assertAlmostEqual(res['ratio_orig'], res['ratio_noisy'], places=6)
        self.assertEqual(res['X_proj_orig'], res['X_proj_noisy'])

    def test_apply_pca_to_dataset_iris(self):
        Xp, ratio = apply_pca_to_dataset('iris', 1)
        self.assertIsInstance(Xp, Matrix)
        self.assertEqual(Xp.cols, 1)
        self.assertTrue(0 < ratio <= 1)

    def test_invalid_dataset(self):
        with self.assertRaises(ValueError):
            apply_pca_to_dataset('unknown', 2)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            apply_pca_to_dataset('iris', 0)


if __name__ == '__main__':
    unittest.main()

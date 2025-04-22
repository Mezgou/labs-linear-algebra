import unittest
import random

from algorithms_pca.matrix import Matrix
from algorithms_pca.pca import pca, explained_variance_ratio


class TestPCAFunctions(unittest.TestCase):
    def test_explained_variance_ratio_basic(self):
        ev = [2.0, 1.0, 0.5]
        self.assertAlmostEqual(explained_variance_ratio(ev, 1), 2.0/3.5)
        self.assertAlmostEqual(explained_variance_ratio(ev, 2), 3.0/3.5)
        self.assertEqual(explained_variance_ratio(ev, 3), 1.0)
        with self.assertRaises(ValueError):
            explained_variance_ratio(ev, 0)
        with self.assertRaises(ValueError):
            explained_variance_ratio(ev, 4)

    def test_pca_identity(self):
        data = [[1, 2], [3, 4], [5, 6]]
        X = Matrix(data)
        X_proj, ratio = pca(X, k=1)
        self.assertAlmostEqual(ratio, 1.0)
        self.assertEqual(X_proj.rows, 3)
        self.assertEqual(X_proj.cols, 1)

    def test_pca_random(self):
        random.seed(0)
        data = [[random.random() for _ in range(3)] for __ in range(5)]
        X = Matrix(data)
        X_proj2, ratio2 = pca(X, k=2)
        self.assertEqual((X_proj2.rows, X_proj2.cols), (5, 2))
        self.assertTrue(0 <= ratio2 <= 1)


if __name__ == '__main__':
    unittest.main()

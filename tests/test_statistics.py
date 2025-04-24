import unittest

from algorithms_pca.matrix import Matrix
from algorithms_pca.statistics import covariance_matrix, center_data


class TestMatrixStats(unittest.TestCase):
    def test_center_data(self):
        X = Matrix([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        X_centered, _ = center_data(X)
        expected_means = [0.0, 0.0, 0.0]
        col_means = X_centered.mean(axis=0)
        for j in range(3):
            self.assertAlmostEqual(col_means._data[0][j], expected_means[j], places=6)

    def test_covariance_matrix(self):
        X = Matrix([
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9]
        ])
        Xc, _ = center_data(X)
        cov = covariance_matrix(Xc)

        self.assertAlmostEqual(cov._data[0][0], 0.61655556, places=5)
        self.assertAlmostEqual(cov._data[0][1], 0.61544444, places=5)
        self.assertAlmostEqual(cov._data[1][0], 0.61544444, places=5)
        self.assertAlmostEqual(cov._data[1][1], 0.71655556, places=5)


if __name__ == "__main__":
    unittest.main()

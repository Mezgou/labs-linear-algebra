import unittest

from algorithms_pca.linear_solver import gauss_solver
from algorithms_pca.matrix import Matrix


class TestUniqueSolution(unittest.TestCase):
    def test_2x2_unique(self):
        A = Matrix([[2, 1], [4, -6]])
        b = Matrix([[3], [-2]])
        sol_list = gauss_solver(A, b)
        self.assertEqual(len(sol_list), 1)
        sol = sol_list[0]
        expected = Matrix([[1.0], [1.0]])
        self.assertEqual(sol, expected)

    def test_already_reduced(self):
        A = Matrix([[1, 0], [0, 1]])
        b = Matrix([[5], [6]])
        sol_list = gauss_solver(A, b)
        self.assertEqual(sol_list, [Matrix([[5.0], [6.0]])])

    def test_1x1(self):
        A = Matrix([[2]])
        b = Matrix([[4]])
        sol_list = gauss_solver(A, b)
        self.assertEqual(sol_list, [Matrix([[2.0]])])


class TestInfiniteSolutions(unittest.TestCase):
    def test_2x2_infinite(self):
        A = Matrix([[1, 2], [2, 4]])
        b = Matrix([[3], [6]])
        sol_list = gauss_solver(A, b)
        self.assertEqual(len(sol_list), 2)
        part, basis = sol_list
        self.assertEqual(part, Matrix([[3.0], [0.0]]))
        self.assertEqual(basis, Matrix([[-2.0], [1.0]]))

    def test_zero_rows(self):
        A = Matrix([[0, 0], [0, 0]])
        b = Matrix([[0], [0]])
        sol_list = gauss_solver(A, b)
        self.assertEqual(len(sol_list), 3)
        part = sol_list[0]
        self.assertEqual(part, Matrix([[0.0], [0.0]]))
        self.assertIn(Matrix([[1.0], [0.0]]), sol_list[1:])
        self.assertIn(Matrix([[0.0], [1.0]]), sol_list[1:])


class TestInconsistentSystem(unittest.TestCase):
    def test_inconsistent(self):
        A = Matrix([[1, 2], [2, 4]])
        b = Matrix([[3], [7]])
        with self.assertRaises(ValueError):
            gauss_solver(A, b)


class TestBoundaryCases(unittest.TestCase):
    def test_3x3_unique(self):
        A = Matrix([[1, 0, 2], [0, 1, -1], [2, -1, 3]])
        b = Matrix([[3], [1], [4]])
        sol_list = gauss_solver(A, b)
        self.assertEqual(len(sol_list), 1)
        sol = sol_list[0]
        self.assertEqual(A.matmul(sol), b)

    def test_3x3_infinite(self):
        A = Matrix([[1, 2, -1], [2, 4, -2], [3, 6, -3]])
        b = Matrix([[1], [2], [3]])
        sol_list = gauss_solver(A, b)
        self.assertEqual(len(sol_list), 3)
        part = sol_list[0]
        basis_list = sol_list[1:]
        self.assertEqual(A.matmul(part), b)
        zero = Matrix([[0], [0], [0]])
        for basis in basis_list:
            self.assertEqual(A.matmul(basis), zero)


class TestGaussSolverInfiniteSolutions(unittest.TestCase):
    def test_infinite_solutions(self):
        A = Matrix([
            [1, 2, -1],
            [-2, -4, 2],
            [3, 6, -3]
        ])
        b = Matrix([
            [1],
            [-2],
            [3]
        ])

        solutions = gauss_solver(A, b)

        self.assertGreater(len(solutions), 1)

        particular = solutions[0]
        result = A.matmul(particular)
        self.assertEqual(result, b)

        for basis_vector in solutions[1:]:
            zero = A.matmul(basis_vector)
            self.assertEqual(zero, Matrix([[0] for _ in range(A.rows)]))


if __name__ == '__main__':
    unittest.main()

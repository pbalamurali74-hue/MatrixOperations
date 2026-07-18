import unittest
import numpy as np
from matrix_operations import (
    add_matrices,
    subtract_matrices,
    multiply_matrices,
    transpose_matrix,
    determinant_matrix,
    inverse_matrix,
    scalar_multiply,
    matrix_power,
    matrix_rank,
)


class TestMatrixOperations(unittest.TestCase):

    def setUp(self):
        # Define standard matrices for testing
        self.A_square = np.array([[1, 2], [3, 4]], dtype=float)
        self.B_square = np.array([[5, 6], [7, 8]], dtype=float)
        self.rect_3x2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        self.rect_2x3 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        self.singular_matrix = np.array([[1, 2], [2, 4]], dtype=float)
        self.identity_2x2 = np.eye(2, dtype=float)

    def test_add_matrices_success(self):
        result = add_matrices(self.A_square, self.B_square)
        expected = np.array([[6, 8], [10, 12]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)

    def test_add_matrices_shape_mismatch(self):
        with self.assertRaises(ValueError):
            add_matrices(self.A_square, self.rect_3x2)

    def test_subtract_matrices_success(self):
        result = subtract_matrices(self.B_square, self.A_square)
        expected = np.array([[4, 4], [4, 4]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)

    def test_subtract_matrices_shape_mismatch(self):
        with self.assertRaises(ValueError):
            subtract_matrices(self.A_square, self.rect_3x2)

    def test_multiply_matrices_success(self):
        # 2x2 * 2x2
        result = multiply_matrices(self.A_square, self.B_square)
        expected = np.array([[19, 22], [43, 50]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)

        # 3x2 * 2x3 -> 3x3
        result_rect = multiply_matrices(self.rect_3x2, self.rect_2x3)
        expected_rect = np.array([
            [9, 12, 15],
            [19, 26, 33],
            [29, 40, 51]
        ], dtype=float)
        np.testing.assert_array_almost_equal(result_rect, expected_rect)

    def test_multiply_matrices_shape_mismatch(self):
        with self.assertRaises(ValueError):
            # Columns of A_square (2) != Rows of rect_3x2 (3)
            multiply_matrices(self.A_square, self.rect_3x2)

    def test_transpose_matrix(self):
        # Square transpose
        result = transpose_matrix(self.A_square)
        expected = np.array([[1, 3], [2, 4]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)

        # Rectangular transpose
        result_rect = transpose_matrix(self.rect_3x2)
        expected_rect = np.array([[1, 3, 5], [2, 4, 6]], dtype=float)
        np.testing.assert_array_almost_equal(result_rect, expected_rect)

    def test_determinant_matrix_success(self):
        # 2x2 det(A) = 1*4 - 2*3 = -2
        det = determinant_matrix(self.A_square)
        self.assertAlmostEqual(det, -2.0)

        # 1x1 matrix det
        one_by_one = np.array([[5.5]])
        self.assertAlmostEqual(determinant_matrix(one_by_one), 5.5)

    def test_determinant_matrix_non_square(self):
        with self.assertRaises(ValueError):
            determinant_matrix(self.rect_3x2)

    def test_inverse_matrix_success(self):
        inv = inverse_matrix(self.A_square)
        # Check A * A^-1 is Identity
        identity_check = np.dot(self.A_square, inv)
        np.testing.assert_array_almost_equal(identity_check, self.identity_2x2)

    def test_inverse_matrix_non_square(self):
        with self.assertRaises(ValueError):
            inverse_matrix(self.rect_3x2)

    def test_inverse_matrix_singular(self):
        with self.assertRaises(ValueError):
            inverse_matrix(self.singular_matrix)

    def test_scalar_multiply(self):
        result = scalar_multiply(self.A_square, 2.5)
        expected = np.array([[2.5, 5.0], [7.5, 10.0]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)

        # Test invalid scalar input
        with self.assertRaises(ValueError):
            scalar_multiply(self.A_square, "invalid_scalar")

    def test_matrix_power_success(self):
        # A^0 is Identity
        np.testing.assert_array_almost_equal(matrix_power(self.A_square, 0), self.identity_2x2)
        # A^1 is A
        np.testing.assert_array_almost_equal(matrix_power(self.A_square, 1), self.A_square)
        # A^2
        result = matrix_power(self.A_square, 2)
        expected = np.dot(self.A_square, self.A_square)
        np.testing.assert_array_almost_equal(result, expected)

    def test_matrix_power_non_square(self):
        with self.assertRaises(ValueError):
            matrix_power(self.rect_3x2, 2)

    def test_matrix_power_invalid_exponent(self):
        with self.assertRaises(ValueError):
            matrix_power(self.A_square, "invalid_power")

    def test_matrix_rank(self):
        # Full rank 2x2
        self.assertEqual(matrix_rank(self.A_square), 2)
        # Rank deficient 2x2 (singular)
        self.assertEqual(matrix_rank(self.singular_matrix), 1)
        # 3x2 matrix rank
        self.assertEqual(matrix_rank(self.rect_3x2), 2)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from ..neural_map import bubble, conical, gaussian, gaussian_cut, mexican_hat, no_neighbourhood

TOLERANCE = 1e-8


def euclidean(f_element, s_element):
    return ((f_element[0] - s_element[0]) ** 2 + (f_element[1] - s_element[1]) ** 2) ** 0.5


cart_coord = np.array([
    [[0.5, 0.], [0., 0.8660254], [0.5, 1.73205081], [0., 2.59807621], [0.5, 3.46410162]],
    [[1.5, 0.], [1., 0.8660254], [1.5, 1.73205081], [1., 2.59807621], [1.5, 3.46410162]],
    [[2.5, 0.], [2., 0.8660254], [2.5, 1.73205081], [2., 2.59807621], [2.5, 3.46410162]],
    [[3.5, 0.], [3., 0.8660254], [3.5, 1.73205081], [3., 2.59807621], [3.5, 3.46410162]],
    [[4.5, 0.], [4., 0.8660254], [4.5, 1.73205081], [4., 2.59807621], [4.5, 3.46410162]],
    [[5.5, 0.], [5., 0.8660254], [5.5, 1.73205081], [5., 2.59807621], [5.5, 3.46410162]],
    [[6.5, 0.], [6., 0.8660254], [6.5, 1.73205081], [6., 2.59807621], [6.5, 3.46410162]],
    [[7.5, 0.], [7., 0.8660254], [7.5, 1.73205081], [7., 2.59807621], [7.5, 3.46410162]]
])
COLUMN = 1
ROW = 1
RADIUS = 2.
LEARNING_RATE = 0.5

bmu = cart_coord[COLUMN, ROW]


class BubbleTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = bubble
        self.update_matrix = self.tested_function(cart_coord, bmu, RADIUS, LEARNING_RATE)

    def test_bmu_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - LEARNING_RATE)
        self.assertLessEqual(error, TOLERANCE, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - self.update_matrix.max())
        self.assertLessEqual(error, TOLERANCE, 'bmu has not the greatest value in g matrix')

    def test_neighbourhood_values(self):
        for i in range(cart_coord.shape[0]):
            for j in range(cart_coord.shape[1]):
                neighbourhood_membership = RADIUS - euclidean(cart_coord[i, j], bmu)
                if neighbourhood_membership > 0:
                    error = abs(self.update_matrix[i, j] - LEARNING_RATE)
                    self.assertLessEqual(error, TOLERANCE, 'g matrix has an incorrect values')
                else:
                    error = abs(self.update_matrix[i, j])
                    self.assertLessEqual(error, TOLERANCE, 'g matrix has an incorrect values')


class ConicalTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = conical
        self.update_matrix = self.tested_function(cart_coord, bmu, RADIUS, LEARNING_RATE)

    def test_bmu_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - LEARNING_RATE)
        self.assertLessEqual(error, TOLERANCE, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - self.update_matrix.max())
        self.assertLessEqual(error, TOLERANCE, 'bmu has not the greatest value in g matrix')

    def test_neighbourhood_values(self):
        for i in range(cart_coord.shape[0]):
            for j in range(cart_coord.shape[1]):
                neighbourhood_membership = RADIUS - euclidean(cart_coord[i, j], bmu)
                if neighbourhood_membership > 0:
                    self.assertGreater(self.update_matrix[i, j], 0,
                                       'g matrix has an incorrect values')
                else:
                    error = abs(self.update_matrix[i, j])
                    self.assertLessEqual(error, TOLERANCE, 'g matrix map has an incorrect values')


class GaussianTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = gaussian
        self.update_matrix = self.tested_function(cart_coord, bmu, RADIUS, LEARNING_RATE)

    def test_bmu_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - LEARNING_RATE)
        self.assertLessEqual(error, TOLERANCE, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - self.update_matrix.max())
        self.assertLessEqual(error, TOLERANCE, 'bmu has not the greatest value in g matrix')

    def test_neighbourhood_values(self):
        for i in range(cart_coord.shape[0]):
            for j in range(cart_coord.shape[1]):
                self.assertGreater(self.update_matrix[i, j], 0, 'g matrix has an incorrect values')


class GaussianCutTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = gaussian_cut
        self.update_matrix = self.tested_function(cart_coord, bmu, RADIUS, LEARNING_RATE)

    def test_bmu_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - LEARNING_RATE)
        self.assertLessEqual(error, TOLERANCE, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - self.update_matrix.max())
        self.assertLessEqual(error, TOLERANCE, 'bmu has not the greatest value in g matrix')

    def test_neighbourhood_values(self):
        for i in range(cart_coord.shape[0]):
            for j in range(cart_coord.shape[1]):
                neighbourhood_membership = RADIUS - euclidean(cart_coord[i, j], bmu)
                if neighbourhood_membership > 0:
                    self.assertGreater(self.update_matrix[i, j], 0,
                                       'g matrix has an incorrect values')
                else:
                    error = abs(self.update_matrix[i, j])
                    self.assertLessEqual(error, TOLERANCE, 'g matrix map has an incorrect values')


class MexicanHatTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = mexican_hat
        self.update_matrix = self.tested_function(cart_coord, bmu, RADIUS, LEARNING_RATE)

    def test_bmu_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - LEARNING_RATE)
        self.assertLessEqual(error, TOLERANCE, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - self.update_matrix.max())
        self.assertLessEqual(error, TOLERANCE, 'bmu has not the greatest value in g matrix')

    def test_min_values(self):
        self.assertLess(self.update_matrix.min(), 0, 'min value is greater or equal than zero')


class NoNeighbourhoodTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = no_neighbourhood
        self.update_matrix = self.tested_function(cart_coord, bmu, RADIUS, LEARNING_RATE)

    def test_bmu_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - LEARNING_RATE)
        self.assertLessEqual(error, TOLERANCE, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.update_matrix[COLUMN, ROW] - self.update_matrix.max())
        self.assertLessEqual(error, TOLERANCE, 'bmu has not the greatest value in g matrix')

    def test_min_values(self):
        g_c = self.update_matrix.copy()
        g_c[COLUMN, ROW] = 0.
        self.assertEqual(g_c.min(), 0, 'min value is not zero')
        self.assertEqual(g_c.max(), 0, 'max value is not zero (excluding bmu)')

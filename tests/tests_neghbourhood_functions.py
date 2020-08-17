import unittest
from ..neural_map import bubble, conical, gaussian, gaussian_cut, mexican_hat, no_neighbourhood
import numpy as np

tolerance = 1e-8


def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


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
x_coord = 1
y_coord = 1
bmu = cart_coord[x_coord, y_coord]
radius = 2.
learning_rate = 0.5


class BubbleTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = bubble
        self.g = self.tested_function(cart_coord, bmu, radius, learning_rate)

    def test_bmu_value(self):
        error = abs(self.g[x_coord, y_coord] - learning_rate)
        self.assertLessEqual(error, tolerance, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.g[x_coord, y_coord] - self.g.max())
        self.assertLessEqual(error, tolerance, 'bmu has not the greatest value in g matrix')

    def test_neighbourhood_values(self):
        for i in range(cart_coord.shape[0]):
            for j in range(cart_coord.shape[1]):
                neighbourhood_membership = radius - euclidean(cart_coord[i, j], bmu)
                if neighbourhood_membership > 0:
                    error = abs(self.g[i, j] - learning_rate)
                    self.assertLessEqual(error, tolerance, 'g matrix has an incorrect values')
                else:
                    error = abs(self.g[i, j])
                    self.assertLessEqual(error, tolerance, 'g matrix has an incorrect values')


class ConicalTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = conical
        self.g = self.tested_function(cart_coord, bmu, radius, learning_rate)

    def test_bmu_value(self):
        error = abs(self.g[x_coord, y_coord] - learning_rate)
        self.assertLessEqual(error, tolerance, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.g[x_coord, y_coord] - self.g.max())
        self.assertLessEqual(error, tolerance, 'bmu has not the greatest value in g matrix')

    def test_neighbourhood_values(self):
        for i in range(cart_coord.shape[0]):
            for j in range(cart_coord.shape[1]):
                neighbourhood_membership = radius - euclidean(cart_coord[i, j], bmu)
                if neighbourhood_membership > 0:
                    self.assertGreater(self.g[i, j], 0, 'g matrix has an incorrect values')
                else:
                    error = abs(self.g[i, j])
                    self.assertLessEqual(error, tolerance, 'g matrix map has an incorrect values')


class GaussianTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = gaussian
        self.g = self.tested_function(cart_coord, bmu, radius, learning_rate)

    def test_bmu_value(self):
        error = abs(self.g[x_coord, y_coord] - learning_rate)
        self.assertLessEqual(error, tolerance, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.g[x_coord, y_coord] - self.g.max())
        self.assertLessEqual(error, tolerance, 'bmu has not the greatest value in g matrix')

    def test_neighbourhood_values(self):
        for i in range(cart_coord.shape[0]):
            for j in range(cart_coord.shape[1]):
                self.assertGreater(self.g[i, j], 0, 'g matrix has an incorrect values')


class GaussianCutTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = gaussian_cut
        self.g = self.tested_function(cart_coord, bmu, radius, learning_rate)

    def test_bmu_value(self):
        error = abs(self.g[x_coord, y_coord] - learning_rate)
        self.assertLessEqual(error, tolerance, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.g[x_coord, y_coord] - self.g.max())
        self.assertLessEqual(error, tolerance, 'bmu has not the greatest value in g matrix')

    def test_neighbourhood_values(self):
        for i in range(cart_coord.shape[0]):
            for j in range(cart_coord.shape[1]):
                neighbourhood_membership = radius - euclidean(cart_coord[i, j], bmu)
                if neighbourhood_membership > 0:
                    self.assertGreater(self.g[i, j], 0, 'g matrix has an incorrect values')
                else:
                    error = abs(self.g[i, j])
                    self.assertLessEqual(error, tolerance, 'g matrix map has an incorrect values')


class MexicanHatTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = mexican_hat
        self.g = self.tested_function(cart_coord, bmu, radius, learning_rate)

    def test_bmu_value(self):
        error = abs(self.g[x_coord, y_coord] - learning_rate)
        self.assertLessEqual(error, tolerance, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.g[x_coord, y_coord] - self.g.max())
        self.assertLessEqual(error, tolerance, 'bmu has not the greatest value in g matrix')

    def test_min_values(self):
        self.assertLess(self.g.min(), 0, 'min value is greater or equal than zero')


class NoNeighbourhoodTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = no_neighbourhood
        self.g = self.tested_function(cart_coord, bmu, radius, learning_rate)

    def test_bmu_value(self):
        error = abs(self.g[x_coord, y_coord] - learning_rate)
        self.assertLessEqual(error, tolerance, 'wrong value in bmu position')

    def test_max_value(self):
        error = abs(self.g[x_coord, y_coord] - self.g.max())
        self.assertLessEqual(error, tolerance, 'bmu has not the greatest value in g matrix')

    def test_min_values(self):
        g_c = self.g.copy()
        g_c[x_coord, y_coord] = 0.
        self.assertEqual(g_c.min(), 0, 'min value is not zero')
        self.assertEqual(g_c.max(), 0, 'max value is not zero (excluding bmu)')


if __name__ == '__main__':
    unittest.main()

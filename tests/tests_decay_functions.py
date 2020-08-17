import unittest
from ..neural_map import linear, exponential, rational, no_decay

tolerance = 1e-8


class LinearTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = linear

    def test_initial_value(self):
        self.assertEqual(self.tested_function(2., 1., 10., 0.), 2., 'wrong first values')

    def test_final_value(self):
        self.assertEqual(self.tested_function(2., 1., 10., 9.), 1., 'wrong last values')

    def test_variations(self):
        first_epoch = self.tested_function(2., 1., 10., 0.)
        second_epoch = self.tested_function(2., 1., 10., 1.)
        third_epoch = self.tested_function(2., 1., 10., 2.)
        delta = abs((second_epoch - first_epoch) - (third_epoch - second_epoch))
        self.assertLessEqual(delta, tolerance, 'wrong incremental values')


class ExponentialTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = exponential

    def test_initial_value(self):
        self.assertEqual(self.tested_function(2., 1., 10., 0.), 2., 'wrong first values')

    def test_final_value(self):
        self.assertEqual(self.tested_function(2., 1., 10., 9.), 1., 'wrong last values')

    def test_variations(self):
        first_epoch = self.tested_function(2., 1., 10., 0.)
        second_epoch = self.tested_function(2., 1., 10., 1.)
        third_epoch = self.tested_function(2., 1., 10., 2.)
        growth = abs((second_epoch / first_epoch) - (third_epoch / second_epoch))
        self.assertLessEqual(growth, tolerance, 'wrong incremental values')


class RationalTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = rational

    def test_initial_value(self):
        self.assertEqual(self.tested_function(2., 1., 10., 0.), 2., 'wrong first values')

    def test_final_value(self):
        self.assertEqual(self.tested_function(2., 1., 10., 9.), 1., 'wrong last values')

    def test_variations(self):
        first_epoch = self.tested_function(2., 1., 10., 0.)
        second_epoch = self.tested_function(2., 1., 10., 1.)
        third_epoch = self.tested_function(2., 1., 10., 2.)
        ratio = abs((1. / second_epoch - 1. / first_epoch) - (1. / third_epoch - 1. / second_epoch))
        self.assertLessEqual(ratio, tolerance, 'wrong incremental values')


class NoDecayTestCase(unittest.TestCase):
    def setUp(self):
        self.tested_function = no_decay

    def test_initial_value(self):
        self.assertEqual(self.tested_function(2., 1., 10., 0.), 2., 'wrong first values')

    def test_final_value(self):
        # even if the first and final values are not the same, this function should always return the first value
        self.assertEqual(self.tested_function(2., 1., 10., 9.), 2., 'wrong last values')

    def test_variations(self):
        first_epoch = self.tested_function(2., 1., 10., 0.)
        second_epoch = self.tested_function(2., 1., 10., 1.)
        delta = abs(second_epoch - first_epoch)
        self.assertLessEqual(delta, tolerance, 'wrong incremental values')


if __name__ == '__main__':
    unittest.main()

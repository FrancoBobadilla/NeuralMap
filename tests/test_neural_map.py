import unittest
from ..neural_map import NeuralMap
from numpy import array, empty
from numba import jit, float64

tolerance = 1e-8

numeric_weights = array([
    [[1.1, 0.], [-10.1, -10.1]],
    [[0.001, -0.3], [-0.1, 0.9]],
])

numeric_reference = array([0.3, -1.1])

binary_weights = array([
    [[0., 0.], [0., 1.]],
    [[1., 0.], [1., 1.]]
])

binary_reference = array([1., 0.])

cases = [
    {
        'metric': 'euclidean',
        'weights': numeric_weights,
        'kwargs': {},
        'reference': numeric_reference,
        'expected': array([
            [1.3601470508735445, 13.753544997563356],
            [0.8540497643580262, 2.039607805437114]
        ])
    },
    {
        'metric': 'minkowski',
        'weights': numeric_weights,
        'kwargs': {'p': 5},
        'reference': numeric_reference,
        'expected': array([
            [1.1415088153682784, 11.25637360875959],
            [0.8011634902446755, 2.0001279836191452]
        ])
    },
    {
        'metric': 'wminkowski',
        'weights': numeric_weights,
        'kwargs': {'p': 2, 'w': [1.5, 0.5]},
        'reference': numeric_reference,
        'expected': array([
            [1.3200378782444087, 16.23607095328177],
            [0.6009594412271099, 1.1661903789690602]
        ])
    },
    {
        'metric': 'jaccard',
        'weights': binary_weights,
        'kwargs': {},
        'reference': binary_reference,
        'expected': array([
            [1.0, 1.0],
            [0.0, 0.5]
        ])
    },
    {
        'metric': 'russellrao',
        'weights': binary_weights,
        'kwargs': {},
        'reference': binary_reference,
        'expected': array([
            [1.0, 1.0],
            [0.5, 0.5]
        ])
    },

]

wrong_cases = [
    {
        'metric': 'ewclidean',
        'weights': numeric_weights,
        'kwargs': {}
    },
    {
        'metric': '',
        'weights': numeric_weights,
        'kwargs': {}
    },
    {
        'metric': 34,
        'weights': numeric_weights,
        'kwargs': {}
    },
]


def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


@jit(float64[:, :](float64[:, :], float64[:, :]), nopython=True, fastmath=True)
def continuous_jaccard(x, w):
    res = empty((x.shape[0], w.shape[0]), dtype=x.dtype)
    for i in range(x.shape[0]):
        for j in range(w.shape[0]):
            num = 0.0
            denum = 0.0
            for k in range(x.shape[1]):
                num += x[i, k] * w[j, k]
                denum += x[i, k] ** 2 + w[j, k] ** 2

            denum -= num

            if denum == 0:
                num = 1. - num
                denum = 1.

            res[i, j] = 1 - num / denum

    return res


class MetricDistanceTestCase(unittest.TestCase):

    def test_string_entered_metric(self):
        for case in cases:
            exception_raised = False
            try:
                som = NeuralMap(z=case['weights'].shape[2], x=case['weights'].shape[0], y=case['weights'].shape[1],
                                weights=case['weights'],
                                metric=case['metric'],
                                **case['kwargs'])
            except:
                exception_raised = True

            self.assertFalse(exception_raised,
                             'exception raised while entering distance metric ' + str(case['metric']))

            if not exception_raised:
                errors = abs(som.generate_activation_map(case['reference']) - case['expected']).flatten()
                for error in errors:
                    self.assertLessEqual(error, tolerance, 'wrong activation value in ' + str(case['metric']))

    def test_wrong_string_entered(self):
        for case in wrong_cases:
            exception_raised = False
            try:
                som = NeuralMap(z=case['weights'].shape[2], x=case['weights'].shape[0], y=case['weights'].shape[1],
                                weights=case['weights'],
                                metric=case['metric'],
                                **case['kwargs'])
            except:
                exception_raised = True

            self.assertTrue(exception_raised,
                            'exception not raised while entering string distance metric ' + str(case['metric']))

    def test_function_entered_metric(self):
        exception_raised = False
        try:
            NeuralMap(z=cases[0]['weights'].shape[2], x=cases[0]['weights'].shape[0],
                            y=cases[0]['weights'].shape[1],
                            weights=cases[0]['weights'],
                            metric=continuous_jaccard,
                            **cases[0]['kwargs'])
        except:
            exception_raised = True

        self.assertFalse(exception_raised,
                         'exception not raised while entering function distance metric')


class PositionsTestCase(unittest.TestCase):
    def setUp(self):
        self.som = NeuralMap(z=5, x=7, y=10, hexagonal=True)

    def test_boundaries(self):
        positions = self.som.positions.reshape(-1, 2)

        left = 0.0
        right = self.som.x
        lower = 0.0
        upper = self.som.y

        if self.som.hexagonal:
            upper *= (3 ** 0.5) * 0.5  # relation between the apothem of an hexagon and its side length

        for position in positions:
            self.assertGreaterEqual(position[0], left, 'left boundary crossed')
            self.assertLessEqual(position[0], right, 'right boundary crossed')
            self.assertGreaterEqual(position[1], lower, 'lower boundary crossed')
            self.assertLessEqual(position[1], upper, 'upper boundary crossed')

    def test_minimum_distance_between_nodes(self):
        positions = self.som.positions.reshape(-1, 2)
        for ref_position in positions:
            for comp_position in positions:
                distance = euclidean(ref_position, comp_position)
                if distance != 0:
                    self.assertLessEqual(1.0, distance + tolerance,
                                         'there are at least two nodes closer than they should')

    def test_index_position_consistency(self):
        positions = self.som.positions // 1
        for ref_i in range(positions.shape[0]):
            for ref_j in range(positions.shape[1]):
                for comp_i in range(positions.shape[0]):
                    for comp_j in range(positions.shape[1]):
                        if ref_i == comp_i:
                            self.assertEqual(positions[ref_i, ref_j, 0], positions[comp_i, comp_j, 0],
                                             'index consistency not preserved')

                        if ref_j == comp_j:
                            self.assertEqual(positions[ref_i, ref_j, 1], positions[comp_i, comp_j, 1],
                                             'index consistency not preserved')

    def test_inner_adjacency(self):
        positions = self.som.positions.reshape(-1, 2)

        expected_inner_nodes = (self.som.x - 2) * (self.som.y - 2)
        if expected_inner_nodes < 0:
            expected_inner_nodes = 0

        actual_inner_nodes = 0

        for ref_position in positions:

            adjacency_count = 0

            for comp_position in positions:
                distance = euclidean(ref_position, comp_position)
                if distance != 0 and distance <= 1.0 + tolerance:
                    adjacency_count += 1

            if self.som.hexagonal and adjacency_count == 6:
                actual_inner_nodes += 1

            if not self.som.hexagonal and adjacency_count == 4:
                actual_inner_nodes += 1

        self.assertEqual(expected_inner_nodes, actual_inner_nodes, 'the net is not properly connected')


if __name__ == '__main__':
    unittest.main()

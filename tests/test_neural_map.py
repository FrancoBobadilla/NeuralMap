import unittest
from ..neural_map import NeuralMap

tolerance = 1e-8
constructor_values = {
    'z': 3,
    'x': 4,
    'y': 5,
    'hexagonal': True
}


def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class CartCoordTestCase(unittest.TestCase):
    def setUp(self):
        self.som = NeuralMap(**constructor_values)

    def test_boundaries(self):
        cart_coord = self.som.cart_coord.reshape(-1, 2)

        left = 0.0
        right = self.som.x
        lower = 0.0
        upper = self.som.y
        if self.som.hexagonal:
            upper *= (3 ** 0.5) * 0.5  # relation between the apothem of an hexagon and its side length

        for position in cart_coord:
            self.assertGreaterEqual(position[0], left, 'left boundary crossed')
            self.assertLessEqual(position[0], right, 'right boundary crossed')
            self.assertGreaterEqual(position[1], lower, 'lower boundary crossed')
            self.assertLessEqual(position[1], upper, 'upper boundary crossed')

    def test_minimum_distance_between_nodes(self):
        cart_coord = self.som.cart_coord.reshape(-1, 2)
        for ref_position in cart_coord:
            for comp_position in cart_coord:
                distance = euclidean(ref_position, comp_position)
                if distance != 0:
                    self.assertLessEqual(1.0, distance + tolerance,
                                         'there are at least two nodes closer than they should')

    def test_index_position_consistency(self):
        cart_coord = self.som.cart_coord // 1
        for ref_i in range(cart_coord.shape[0]):
            for ref_j in range(cart_coord.shape[1]):
                for comp_i in range(cart_coord.shape[0]):
                    for comp_j in range(cart_coord.shape[1]):
                        if ref_i == comp_i:
                            self.assertEqual(cart_coord[ref_i, ref_j, 0], cart_coord[comp_i, comp_j, 0],
                                             'index consistency not preserved')

                        if ref_j == comp_j:
                            self.assertEqual(cart_coord[ref_i, ref_j, 1], cart_coord[comp_i, comp_j, 1],
                                             'index consistency not preserved')

    def test_inner_adjacency(self):
        cart_coord = self.som.cart_coord.reshape(-1, 2)

        expected_inner_nodes = (self.som.x - 2) * (self.som.y - 2)
        if expected_inner_nodes < 0:
            expected_inner_nodes = 0

        actual_inner_nodes = 0

        for ref_position in cart_coord:

            adjacency_count = 0

            for comp_position in cart_coord:
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

"""
NeuralMap, an Open Source implementation of Self Organizing Maps, with some extra features:
 - Hexagonal and toroidal topology
 - Customizable distance metric, neighbourhood function, and decay functions
 - Data analysis and evaluation
 - Relative Positions
 - HDBSCAN clustering
 - K-means or K-medoids clustering
 - Specific visualizations

"""
from collections import Counter

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

from numpy import arange, random, zeros, array, unravel_index, isnan, meshgrid, transpose, ogrid, \
    fill_diagonal, nan, nan_to_num, argmin, where, isin, unique, quantile

from scipy.spatial.distance import cdist

from hdbscan import HDBSCAN

from sklearn_extra.cluster import KMedoids

from neural_map import _plot, _check_inputs, _decay_functions, _neighbourhood_functions, \
    _weight_init_functions


class NeuralMap:
    """
    NueralMap Self Organizing Map initialization.

    Generates a discrete and low dimensional representation of the input data space.

    Parameters
    ----------
    variables: int
        Number of variables of the input data.
        This value is also the dimension of the nodes weights vectors.
    metric: string or function
        The distance metric that apply to the input data space.
        If a string is passed, it must match one of the followings:
            * braycurtis
            * canberra
            * chebyshev
            * cityblock
            * correlation
            * cosine
            * euclidean
            * jensenshannon
            * mahalanobis
            * minkowski
            * seuclidean
            * sqeuclidean
            * wminkowski
            * dice
            * hamming
            * jaccard
            * kulsinski
            * rogerstanimoto
            * russellrao
            * sokalmichener
            * sokalsneath
            * yule
        (see scipy.spatial.distance)
        If a function is passed, it must accept two arrays ``XA`` and ``XB`` of lengths
        ``m`` and ``n`` with observations of the same dimensionality, that returns
        a matrix ``m x n`` with the distances between all pairs of observations in ``XA``
        and ``XB`` (such as scipy.spatial.distance.cdist).
        If the selected ``metric`` (string or function) takes some extra arguments,
        they can be passed as a dictionary through ``**kwargs``.
    columns: int (optional, default 10)
        Number of horizontal nodes.
    rows: int (optional, default 10)
        Number of vertical nodes.
    hexagonal: bool (optional, default True)
        Whether the nodes are arranged in an hexagonal or squared grid.
    toroidal: bool (optional, default False)
        Whether the nodes are arranged in an toroidal or flat space.
        In a toroidal space, opposite edges of the map are connected.
        Recommended when there is a large amount of nodes.
        If toroidal and hexagonal are True, then it's not recommended to
        have an odd number of columns.
    relative_positions: array_like (optional, default None)
        Provided only when it's needed to load (and initialize) an already trained map.
        Cartesian coordinates of the nodes in the relative positions space.
        Dimension should be (columns, rows, 2).
    weights: array_like (optional, default None)
        Provided only when it's needed to load (and initialize) an already trained map.
        Nodes weights vectors.
        Dimension should be (columns, rows, variables).
    **kwargs: dict (optional)
        Extra arguments for the distance ``metric``.

    """

    def __init__(self,
                 variables,
                 metric,
                 columns=10,
                 rows=10,
                 hexagonal=True,
                 toroidal=False,
                 relative_positions=None,
                 weights=None,
                 **kwargs
                 ):

        _check_inputs.value_type(variables, int)
        _check_inputs.positive(variables)
        _check_inputs.value_type(columns, int)
        _check_inputs.positive(columns)
        _check_inputs.value_type(rows, int)
        _check_inputs.positive(rows)
        _check_inputs.value_type(hexagonal, bool)
        _check_inputs.value_type(toroidal, bool)
        self.metric = metric
        self.kwargs = kwargs

        if isinstance(metric, str):
            def distance(first_array, second_array):
                return cdist(first_array, second_array, self.metric, **self.kwargs)

        else:
            def distance(first_array, second_array):
                return self.metric(first_array, second_array, **self.kwargs)

        self.distance = distance
        self.distance(array([[0., 1.]]), array([[1., 2.], [3., 4.]]))
        self.columns = columns
        self.rows = rows
        self.variables = variables
        self.hexagonal = hexagonal
        self.toroidal = toroidal
        self.width = self.columns
        self.height = self.rows

        if weights is not None:
            _check_inputs.ndarray_and_shape(weights, (self.columns, self.rows, self.variables))

        else:
            weights = zeros((self.columns, self.rows, self.variables))

        self.weights = weights
        self.activation_map = zeros((self.columns, self.rows))
        self.adjacent_nodes_relative_positions = [[(1, 0), (0, 1), (-1, 0), (0, -1)],
                                                  [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        self.positions = transpose(meshgrid(arange(self.columns), arange(self.rows)),
                                   axes=[2, 1, 0]).astype(float)

        if self.hexagonal:
            self.positions[:, 0::2, 0] += 0.5
            self.positions[..., 1] *= (3 ** 0.5) * 0.5
            self.height *= (3 ** 0.5) * 0.5
            self.adjacent_nodes_relative_positions = [
                [(1, 0), (1, 1), (0, 1), (-1, 0), (0, -1), (1, -1)],
                [(1, 0), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1)]
            ]

        if relative_positions is not None:
            _check_inputs.ndarray_and_shape(relative_positions, (self.columns, self.rows, 2))

        else:
            relative_positions = self.positions.copy()

        self.relative_positions = relative_positions
        self.current_epoch = None
        self._unified_distance_matrix_cache = None
        self._hdbscan_cache = [(None, None, None)] * self.columns * self.rows

    def train(self,
              data,
              eval_data=None,
              n_epochs=100,
              initial_epoch=0,
              initial_learning_rate=1.0,
              final_learning_rate=0.01,
              initial_radius=None,
              final_radius=1.0,
              learning_rate_decay_function='linear',
              radius_decay_function='exponential',
              neighbourhood_function='gaussian',
              weight_init_function='standard',
              seed=1,
              verbose=True):
        """
        Train the SOM

        Parameters
        ----------
        data: ndarray
            Data to learn by the SOM.
            Dimension should be (n_observations, variables),
            where ``variables`` is the same value passed in the initialization method.
        eval_data: ndarray (optional, default None)
            Data used to evaluate the SOM for each epoch.
            It should be normalize after split form train data.
            Dimension should be (n_observations, variables),
            where ``variables`` is the same value passed in the initialization method.
        n_epochs: int (optional, default 100)
            Number of epochs to train.
            In each epoch, all observations are processed exactly once, in a random sequence.
        initial_epoch: int (optional, default 0)
            Initial training epoch.
            This might be useful when multiple training sessions are applied
            over the same instance.
        initial_learning_rate: float (optional, default 1.0)
            Initial learning rate value. This value is used in the first epoch,
            and then is decremented for the following ones according to the decay function.
        final_learning_rate: float (optional, default 0.1)
            Final learning rate value. This value is used in the last epoch.
        initial_radius: float (optional, default None)
            Initial neighbourhood radius value. This value is used in the first epoch,
            and then is decremented for the following ones according to the decay function.
            If it's not provided a value, it's calculated as half the minimum dimension of the map.
        final_radius: float (optional, default 1.0)
            Final neighbourhood radius value. This value is used in the last epoch.
        learning_rate_decay_function: string or function (optional, default 'linear')
            Function used to decrement the initial learning rate value over
            a number of epochs and reach the final learning rate value in the last one.
            If a string is passed, it must match one of the followings:
                * linear
                * exponential
                * rational
                * no_decay (when it's needed to use the
                initial learning rate value along all epochs)
            It's possible to pass a custom function, but it should accept the following args:
                * Initial learning rate value
                * Final learning rate value
                * Total number of epochs
                * Current epoch number
            And should return the learning rate value for the current epoch.
        radius_decay_function: string or function (optional, default 'linear')
            Function used to decrement the initial neighbourhood radius value over
            a number of epochs and reach the final neighbourhood radius value in the last one.
            If a string is passed, it must match one of the followings:
                * linear
                * exponential
                * rational
                * no_decay (when it's needed to use the
                initial neighbourhood radius value along all epochs)
            It's possible to pass a custom function, but it should accept the following args:
                * Initial neighbourhood radius value
                * Final neighbourhood radius value
                * Total number of epochs
                * Current epoch number
            And should return the neighbourhood radius value for the current epoch.
        neighbourhood_function: string or function (optional, default 'gaussian')
            Function to compute the neighbourhood area surrounding the best matching unit.
            If a string is passed, it must match one of the followings:
                * bubble
                * conical
                * gaussian
                * gaussian_cut
                * mexican_hat
                * no_neighbourhood (when it's needed to update the bmu only)
            It's possible to pass a custom function, but it should accept the following args:
                * Cartesian coordinates of the nodes
                * Cartesian coordinate of the best matching unit
                * Neighbourhood radius
                * Learning rate
            And should return the update value of each node as a matrix (columns, rows).
        weight_init_function: string or function (optional, default 'standard')
            Function to initialize weights at the beginning of training.
            If a string is passed, it must match one of the followings:
                * standard
                * uniform
                * pca
                * pick_from_data
                * no_init (when it's needed to keep previous values)
        seed: int or None (optional, default 1)
            Random seed.
        verbose: bool (optional, default True)
            Verbosity of training process.

        """

        # if it's not provided a value, it's calculated as half the maximum dimension of the map
        if initial_radius is None:
            initial_radius = max(self.width, self.height) / 2

        # if the number of epochs is set to 1, use no decay function
        if n_epochs == 1:
            learning_rate_decay_function = _decay_functions.no_decay
            radius_decay_function = _decay_functions.no_decay

        # restart u-matrix and hdbscan cahces
        self._unified_distance_matrix_cache = None
        self._hdbscan_cache = [(None, None, None)] * self.columns * self.rows

        if seed is None:
            random.seed(None)
            seed = random.randint(low=0, high=1000000)
            print('generated seed: ', seed)

        # get the learning rate decay function
        if learning_rate_decay_function == 'linear':
            learning_rate_decay_function = _decay_functions.linear

        elif learning_rate_decay_function == 'exponential':
            learning_rate_decay_function = _decay_functions.exponential

        elif learning_rate_decay_function == 'rational':
            learning_rate_decay_function = _decay_functions.rational

        elif learning_rate_decay_function == 'no_decay':
            learning_rate_decay_function = _decay_functions.no_decay

        # get the radius decay function
        if radius_decay_function == 'linear':
            radius_decay_function = _decay_functions.linear

        elif radius_decay_function == 'exponential':
            radius_decay_function = _decay_functions.exponential

        elif radius_decay_function == 'rational':
            radius_decay_function = _decay_functions.rational

        elif radius_decay_function == 'no_decay':
            radius_decay_function = _decay_functions.no_decay

        # get the neighbourhood function
        if neighbourhood_function == 'bubble':
            neighbourhood_function = _neighbourhood_functions.bubble

        elif neighbourhood_function == 'conical':
            neighbourhood_function = _neighbourhood_functions.conical

        elif neighbourhood_function == 'gaussian':
            neighbourhood_function = _neighbourhood_functions.gaussian

        elif neighbourhood_function == 'gaussian_cut':
            neighbourhood_function = _neighbourhood_functions.gaussian_cut

        elif neighbourhood_function == 'mexican_hat':
            neighbourhood_function = _neighbourhood_functions.mexican_hat

        elif neighbourhood_function == 'no_neighbourhood':
            neighbourhood_function = _neighbourhood_functions.no_neighbourhood

        # get the weight initialization function
        if weight_init_function == 'standard':
            random.seed(seed)
            weight_init_function = _weight_init_functions.standard

        elif weight_init_function == 'uniform':
            random.seed(seed)
            weight_init_function = _weight_init_functions.uniform

        elif weight_init_function == 'pca':
            random.seed(seed)
            weight_init_function = _weight_init_functions.pca

        elif weight_init_function == 'pick_from_data':
            random.seed(seed)
            weight_init_function = _weight_init_functions.pick_from_data

        elif weight_init_function == 'no_init':
            weight_init_function = _weight_init_functions.no_init

        # check inputs
        _check_inputs.numpy_matrix(data, self.variables)

        if eval_data is not None:
            _check_inputs.numpy_matrix(eval_data, self.variables)

        _check_inputs.value_type(n_epochs, int)
        _check_inputs.positive(n_epochs)
        _check_inputs.value_type(initial_epoch, int)
        _check_inputs.value_type(initial_learning_rate, float)
        _check_inputs.value_type(final_learning_rate, float)
        _check_inputs.value_type(initial_radius, float)
        _check_inputs.value_type(final_radius, float)
        _check_inputs.function(learning_rate_decay_function)
        _check_inputs.function(radius_decay_function)
        _check_inputs.function(neighbourhood_function)
        _check_inputs.function(weight_init_function)
        _check_inputs.value_type(seed, int)
        _check_inputs.value_type(verbose, bool)

        # initialize weight values
        self.weights = weight_init_function(data, self.weights)

        # declare an array for quantization and topographic error of each epoch
        epochs_quantization_error = zeros(n_epochs)
        epochs_topographic_error = zeros(n_epochs)

        # the next variables before the for loop will be used only in case of toroidal topology

        # array with map dimensions, do not confuse with the number of columns and rows
        dimensions = array([self.width, self.height])

        # map center position
        center = (self.columns // 2, self.rows // 2)

        # get indices of the update matrix
        update_matrix_indices = ogrid[[slice(0, self.columns), slice(0, self.rows)]]

        # compute the corrections that each rows must have,
        # if the bmu is in a row offset from the map center
        correction = zeros(self.rows, dtype='int')
        correction[center[1] % 2:: 2] = (center[1] % 2) * 2 - 1

        for epoch in range(initial_epoch, n_epochs):

            # set current epoch
            self.current_epoch = epoch

            # generate observation indices to iterate over
            indices = arange(len(data))

            if self.current_epoch == 0:

                # generate a new random state using the passed seed
                random.seed(seed)

            else:

                # each epoch generate a new random state
                random.seed(epoch)

            # shuffles all observations
            random.shuffle(indices)

            # compute the learning rate of this epoch
            learning_rate = learning_rate_decay_function(initial_learning_rate, final_learning_rate,
                                                         n_epochs, epoch)

            # compute the neighbourhood radius of this epoch
            radius = radius_decay_function(initial_radius, final_radius, n_epochs, epoch)

            # print progress
            if verbose:
                print('\nEpoch: ', epoch + 1, ' of ', n_epochs,
                      '    Learning rate: ', learning_rate,
                      '    Radius: ', radius)

            # plot update matrix and relative positions displacement
            # for the first iteration in each epoch
            plot_update = False

            # plt.scatter(self._rp[..., 0], self._rp[..., 1])
            # plt.show()
            # plt.scatter((self._rp[..., 0] + self._width / 3) % self._width,
            #             (self._rp[..., 1] + self._height / 3) % self._height)
            # plt.show()

            # in case of having a toroidal topology the update matrix is the same for each iteration
            # in every iteration, this matrix is rotated to match the bmu position
            update_matrix_over_center = neighbourhood_function(self.positions,
                                                               self.positions[center],
                                                               radius, learning_rate)

            for i in indices:

                # get best matching unit for a given observation
                ind = data[i]
                bmu = self.get_best_matching_unit(ind)

                if self.toroidal:

                    # The update matrix computed over the map center must be rotated to match the
                    # bmu, and so emulate a toroidal space. For this, the matrix is reindexed,
                    # subtracting to the original indices the corresponding horizontal and vertical
                    # displacement. Also, if the topology is hexagonal and the row of the bmu is
                    # offset from the row of the map center, it must be applied an extra correction
                    # that is computed at the beginning of the training. Finally, the new positions
                    # are rotated to fit the map dimensions.

                    if plot_update and verbose:
                        _plot.update(self.positions, self.hexagonal, update_matrix_over_center,
                                     dimensions, None, None, None)

                    # computes the amount of columns and rows between the map center and the bmu
                    offset = [bmu[0] - center[0], bmu[1] - center[1]]

                    # compute the offset correction for each row
                    offset_correction = (self.hexagonal and offset[1] % 2) * correction

                    # rotate update matrix to match its center with the bmu position
                    update_matrix = update_matrix_over_center[tuple([
                        (update_matrix_indices[0] - (offset[0] + offset_correction)) % self.columns,
                        (update_matrix_indices[1] - offset[1]) % self.rows
                    ])]

                    if plot_update and verbose:
                        _plot.update(self.positions, self.hexagonal, update_matrix, dimensions,
                                     None, None, None)
                        temp_relative_positions = self.relative_positions.copy()
                        previous_relative_positions = self.relative_positions.copy()
                        bmu_displacement = (self.positions[bmu] - temp_relative_positions[bmu]) * \
                            update_matrix[bmu]
                        temp_relative_positions[bmu] += bmu_displacement
                        displacement = (temp_relative_positions[bmu] - temp_relative_positions) * \
                            update_matrix[..., None]
                        displacement[bmu] = 0.
                        temp_relative_positions += displacement
                        displacement[bmu] = bmu_displacement
                        _plot.update(self.positions, self.hexagonal, update_matrix, dimensions,
                                     previous_relative_positions[bmu], previous_relative_positions,
                                     displacement)

                    # compute displacement for bmu relative position

                    # get opposite point of the bmu absolute position
                    position = self.positions[bmu]
                    anti_bmu = (position + dimensions / 2) % dimensions

                    # get the 'quadrant' the bmu is in, with respect to its opposite position
                    quadrant = array([
                        offset[0] > 0 or anti_bmu[0] < 1,
                        offset[1] > 0 or anti_bmu[1] < 1
                    ]) * 2 - 1

                    # compute the matrix with the positions to which the bmu must tend
                    mod = (self.relative_positions[bmu] * quadrant < anti_bmu * quadrant) \
                        * dimensions * quadrant

                    # compute displacement for bmu relative position
                    bmu_displacement = (self.positions[bmu] - mod - self.relative_positions[bmu]) \
                        * update_matrix[bmu]

                    # update relative positions
                    self.relative_positions[bmu] += bmu_displacement

                    # fits new relative positions into the map
                    self.relative_positions[bmu] %= dimensions

                    # computes displacement for all the relative positions of neighbourhood nodes

                    # get opposite point of the bmu relative position
                    anti_bmu = (self.relative_positions[bmu] + dimensions / 2) % dimensions

                    # get the 'quadrant' the bmu is in, with respect to its opposite position
                    quadrant = array([
                        self.relative_positions[bmu][0] - self.width / 2 > 0 or anti_bmu[0] < 1,
                        self.relative_positions[bmu][1] - self.height / 2 > 0 or anti_bmu[1] < 1
                    ]) * 2 - 1

                    # compute the matrix with the positions to which each node must tend
                    mod = (self.relative_positions * quadrant < anti_bmu * quadrant) \
                        * dimensions * quadrant

                    # compute displacement for each node relative position
                    displacement = (self.relative_positions[bmu] - mod - self.relative_positions) \
                        * update_matrix[..., None]

                    displacement[bmu] = 0.

                    # update relative positions
                    self.relative_positions += displacement

                    # fits new relative positions into the map
                    self.relative_positions %= dimensions

                    # plot for easier debugging
                    if plot_update and verbose:
                        plot_update = False
                        displacement[bmu] = bmu_displacement
                        _plot.update(self.positions, self.hexagonal, update_matrix, dimensions,
                                     previous_relative_positions[bmu],
                                     previous_relative_positions, displacement)
                        plt.show()

                else:

                    # compute update matrix, with the proportion of change for each node
                    update_matrix = neighbourhood_function(self.positions, self.positions[bmu],
                                                           radius, learning_rate)

                    # the relative positions of neighbourhood nodes tends towards
                    # the bmu relative position
                    bmu_displacement = (self.positions[bmu] - self.relative_positions[bmu]) \
                        * update_matrix[bmu]

                    # update relative positions
                    self.relative_positions[bmu] += bmu_displacement

                    # nodes in the bmu neighbourhood tends towards the new bmu relative position
                    displacement = (self.relative_positions[bmu] - self.relative_positions) \
                        * update_matrix[..., None]

                    # the bmu relative position is overridden over the displacement matrix
                    displacement[bmu] = 0.

                    # update relative positions
                    self.relative_positions += displacement

                    # plot for easier debugging
                    if plot_update and verbose:
                        plot_update = False
                        print("\nInput vector: ")
                        print("(" + str(i) + "): " + str(ind))
                        print("\nActivation map: ")
                        _plot.update(self.positions, self.hexagonal, self.activation_map,
                                     dimensions,
                                     None, None,
                                     displacement)
                        plt.show()
                        print("\nBest Matching Unit: ")
                        print(str(bmu) + ": " + str(self.weights[bmu]))
                        print("\nUpdate of BMU and its neighbourhood: ")
                        _plot.update(self.positions, self.hexagonal, update_matrix, dimensions,
                                     self.relative_positions[bmu], self.relative_positions,
                                     displacement)
                        plt.show()
                        print("----------------------------------------------------------------\n")

                # update weights
                self.weights += (ind - self.weights) * update_matrix[..., None]

            # evaluate the SOM with respect to the input data each epoch
            if eval_data is not None:
                epochs_quantization_error[epoch], epochs_topographic_error[epoch] = self.evaluate(
                    eval_data)

        # plot quantization error and topographic error evolution
        if eval_data is not None:
            print('\n\nQuantization error per epoch')
            print('first epoch: ', epochs_quantization_error[0])
            print('last epoch: ', epochs_quantization_error[-1])
            plt.plot(epochs_quantization_error)
            plt.grid()
            plt.show()
            print('\n\nTopographic error per epoch')
            print('first epoch: ', epochs_topographic_error[0])
            print('last epoch: ', epochs_topographic_error[-1])
            plt.plot(epochs_topographic_error)
            plt.grid()
            plt.show()

    def generate_activation_map(self, ind):
        """
        Compute the distance between the reference vector (ind) and all nodes

        """
        self.activation_map = self.distance(ind.reshape([1, -1]), self.weights.reshape([
            self.columns * self.rows, self.variables])).reshape([self.columns, self.rows])

        return self.activation_map

    def get_best_matching_unit(self, ind):
        """
        From the activation map, select the node with the minimum distance
        and return its position in the 2D array.

        """
        return unravel_index(self.generate_activation_map(ind).argmin(), self.activation_map.shape)

    def get_unified_distance_matrix(self):
        """
        Compute the u-matrix of the SOM.

        The u-matrix value of a particular node is the distance between
        that node and all its adjacent neighbours, and the mean distance.
        The distance metric is provided in the initialization parameter.

        Returns
        -------
            unified_distance_matrix, adjacency_matrix: ndarray, ndarray
                The first value is the unified distance matrix, including the distance between
                each node and all its neighbours (clockwise), and the mean distance.
                Dimensions are (columns, rows, n_neigh + 1), where n_neigh is 4 in case of a
                square arrangement, and 6 in case of a hexagonal arrangement.
                The second value is the distance matrix between each pair of adjacent nodes.
                Dimensions are (columns * rows, columns * rows).
                The nan value is used to represent that a pair of nodes is not adjacent.

        """
        if self._unified_distance_matrix_cache is not None:
            return self._unified_distance_matrix_cache

        adjacency_matrix = zeros((self.columns * self.rows, self.columns * self.rows)) * nan
        fill_diagonal(adjacency_matrix, 0.)
        adjacency_count = len(self.adjacent_nodes_relative_positions[0])
        unified_distance_matrix = zeros((self.columns, self.rows, 1 + adjacency_count))

        for x_index in range(self.columns):
            for y_index in range(self.rows):
                adjacent_nodes = 0

                for k, (i, j) in enumerate(self.adjacent_nodes_relative_positions[y_index % 2]):
                    if self.toroidal:
                        neighbour = array([[self.weights[
                                                (x_index + i + self.columns) % self.columns,
                                                (y_index + j + self.rows) % self.rows
                                            ]]])

                    elif self.columns > x_index + i >= 0 <= y_index + j < self.rows:
                        neighbour = array([[self.weights[x_index + i, y_index + j]]])

                    else:
                        neighbour = None

                    if neighbour is not None:
                        distance = self.distance(self.weights[x_index, y_index].reshape([1, -1]),
                                                 neighbour[0, 0].reshape([1, -1]))
                        unified_distance_matrix[x_index, y_index, k] = distance
                        unified_distance_matrix[x_index, y_index, adjacency_count] += distance
                        adjacent_nodes += 1
                        adjacency_matrix[x_index * self.rows + y_index,
                                         ((x_index + i + self.columns) % self.columns) * self.rows +
                                         (y_index + j + self.rows) % self.rows] = distance

                    else:
                        unified_distance_matrix[x_index, y_index, k] = nan

                if adjacent_nodes > 0:
                    unified_distance_matrix[x_index, y_index, adjacency_count] /= adjacent_nodes

                else:
                    unified_distance_matrix[x_index, y_index, adjacency_count] = nan

        self._unified_distance_matrix_cache = unified_distance_matrix, adjacency_matrix

        return self._unified_distance_matrix_cache

    def analyse(self, data):
        """
        Analyse a set of observation using the SOM, computing the quantization error,
        activation frequency and mean distance for each node.

        Quantization error is the average distance between each node weights vector and all the
        observations for which it's the BMU.

        Activation frequency is the number of times that each node is BMU.

        Mean distance is the average distance between each node weights vector and all the
        observations.

        Parameters
        ----------
            data: ndarray
                Data to analyse.
                Dimensions should be (n_obs, variables),
                where ``variables`` is the same value passed in the initialization method.

        Returns
        -------
            quantization_error, activation_frequency, mean_distance: ndarray, ndarray, ndarray

        """
        _check_inputs.numpy_matrix(data, self.variables)
        quantization_error = zeros((self.columns, self.rows))
        activation_frequency = zeros((self.columns, self.rows))
        mean_distance = zeros((self.columns, self.rows))

        for ind in data:
            best_matching_unit = self.get_best_matching_unit(ind)
            activation_frequency[best_matching_unit] += 1
            quantization_error[best_matching_unit] += self.activation_map[best_matching_unit]
            mean_distance += self.activation_map

        quantization_error[activation_frequency > 0] /= \
            activation_frequency[activation_frequency > 0]
        quantization_error[activation_frequency == 0] = nan
        mean_distance /= len(data)

        return activation_frequency, quantization_error, mean_distance

    def map_attachments(self, data, attachments, aggregation_function=None):
        """
        Map data attachments to the SOM and then apply
        an aggregation function over each node mapped values.

        Parameters
        ----------
        data: ndarray
            Data to map their attachments.
            Dimensions should be (n_obs, variables),
            where ``variables`` is the same value passed in the initialization method.
        attachments: ndarray
            Data attachments to map to the SOM.
            Dimensions should be (n_obs, ).
        aggregation_function: function (optional, default None)
            After map all data attachments, it's possible to apply a function to
            aggregate them, for each node.
            If nothing is provided, no aggregation function is applied.

        Returns
        -------
            Mapped values: ndarray
                Matrix with mapped values for each node.
                Dimensions are (columns, rows).
                Each value depends on the aggregation function.

        """
        if aggregation_function is None:
            def identity(input_value):
                return input_value

            aggregation_function = identity

        _check_inputs.numpy_matrix(data, self.variables)
        _check_inputs.length(data, attachments)
        _check_inputs.function(aggregation_function)
        dict_map = {(i, j): [] for i in range(self.columns) for j in range(self.rows)}

        for ind, attachment in zip(data, attachments):
            dict_map[tuple(self.get_best_matching_unit(ind))].append(attachment)

        result = [[0.0] * self.rows for _ in range(self.columns)]

        for k, item in dict_map.items():
            result[k[0]][k[1]] = aggregation_function(item)

        return array(result)

    def k_means(self, n_clusters=4):
        """
        Perform k-means++ clustering over the nodes based
        on their weights vectors (see sklearn.cluster.KMeans).

        Returns
        -------
            labels, cluster_centers: ndarray, ndarray
                The first value is the matrix of each node cluster label.
                Dimensions are (columns, rows).
                The second value is an array with the clusters centroids.

        """
        _check_inputs.value_type(n_clusters, int)
        _check_inputs.positive(n_clusters)
        clusters = KMeans(n_clusters=n_clusters, init="k-means++") \
            .fit(self.weights.reshape(self.columns * self.rows, self.variables))

        return clusters.labels_.reshape(self.columns, self.rows), clusters.cluster_centers_

    def k_medoids(self, n_clusters=4):
        """Perform k-medoids++ clustering over the nodes based
        on their weights vectors (see sklearn_extra.cluster.KMedoids).

        Returns
        -------
            labels, cluster_centers: ndarray, ndarray
                The first value is the matrix of each node cluster label.
                Dimensions are (columns, rows).
                The second value is an array with the clusters centroids.

        """
        _check_inputs.value_type(n_clusters, int)
        _check_inputs.positive(n_clusters)

        if isinstance(self.metric, str):
            metric = self.metric

        else:
            def custom_metric(first_set, second_set):
                return self.distance(array([first_set]), array([second_set]))

            metric = custom_metric

        clusters = KMedoids(n_clusters=n_clusters, init="k-medoids++", metric=metric) \
            .fit(self.weights.reshape(self.columns * self.rows, self.variables))

        return clusters.labels_.reshape(self.columns, self.rows), clusters.cluster_centers_

    def hdbscan(self, min_cluster_size=3, plot_condensed_tree=False):
        """
        Perform a clustering operation over the nodes using the HDBSCAN technique.

        Its application is based on the notion that if there is a cluster in the input data space
        it would be reflected in the SOM as a cohesive cluster of adjacent nodes
        separated from the surrounding nodes.

        See The hdbscan Clustering Library

        Parameters
        ----------
            min_cluster_size: int (optional, default 3)
                Minimum valid amount of nodes in each cluster.
                Should be greater than 0.
            plot_condensed_tree: bool (optional, default False)
                Plot the clusters hierarchy.

        Returns
        -------
            labels, probabilities, outlier_score: ndarray, ndarray, ndarray
                The first value is the matrix of each node's cluster label.
                If the node is an outlier, its label is -1.
                Dimensions are (columns, rows).
                The second value is the probability of each node to belong to its cluster.
                The third value is the probability of each node to be an outlier.

        """
        _check_inputs.value_type(min_cluster_size, int)
        _check_inputs.positive(min_cluster_size - 1)
        _check_inputs.value_type(plot_condensed_tree, bool)

        if self._hdbscan_cache[min_cluster_size][0] is not None \
                and self._hdbscan_cache[min_cluster_size][1] is not None \
                and self._hdbscan_cache[min_cluster_size][2] is not None:
            return self._hdbscan_cache[min_cluster_size]

        adjacency_matrix = self.get_unified_distance_matrix()[1]
        clusters = HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=2)
        clusters.fit(nan_to_num(adjacency_matrix, nan=1e8))
        labels = clusters.labels_
        probabilities = clusters.probabilities_
        outlier_scores = clusters.outlier_scores_

        if plot_condensed_tree:
            clusters.condensed_tree_.plot(select_clusters=True, label_clusters=True)

        self._hdbscan_cache[min_cluster_size] = (
            labels.reshape(self.columns, self.rows),
            probabilities.reshape(self.columns, self.rows),
            outlier_scores.reshape(self.columns, self.rows)
        )

        return self._hdbscan_cache[min_cluster_size]

    def evaluate(self, data):
        """
        Evaluate the SOM through quantization and topographic error.

        Quantization error is the average distance between
        each observation and their bmu in the SOM.
        Topographic error is the number of observations that their best matching unit are not
        adjacent to their second best matching unit, divided by the total number of observations.

        Parameters
        ----------
            data: ndarray
                Observations with which the SOM is evaluated.
                Dimensions should be (n_obs, variables),
                where ``variables`` is the same value passed in the initialization method.

        Returns
        -------
            quantization_error, topographic error: ndarray, ndarray
                The first value is the SOM quantization error, given the input data.
                Dimensions are (columns, row).
                The second value is the SOM topographic error, given the input data.
                Dimensions are (columns, row).

        """
        _check_inputs.numpy_matrix(data, self.variables)
        topographic_error = 0
        quantization_error = 0

        for ind in data:
            activation_map = self.generate_activation_map(ind)
            f_bmu = unravel_index(argmin(activation_map), activation_map.shape)
            quantization_error += activation_map[f_bmu]
            activation_map[f_bmu] = activation_map.max()
            s_bmu = unravel_index(argmin(activation_map), activation_map.shape)
            error = 1

            for (i, j) in self.adjacent_nodes_relative_positions[f_bmu[1] % 2]:
                if (self.toroidal and s_bmu[0] == (f_bmu[0] + i + self.columns) % self.columns
                        and s_bmu[1] == (f_bmu[1] + j + self.rows) % self.rows):
                    error = 0

                if (not self.toroidal
                        and self.columns > f_bmu[0] + i >= 0 <= f_bmu[1] + j < self.rows
                        and s_bmu[0] == f_bmu[0] + i and s_bmu[1] == f_bmu[1] + j):
                    error = 0

            topographic_error += error

        return quantization_error / data.shape[0], topographic_error / data.shape[0]

    def get_dict(self):
        """
        Get a SOM representation as a dict (including kwargs).

        Useful to initialize an identical NeuralMap instance, by just passing the
        dictionary to the constructor method (with no need for further training).

        Also useful to save or serialize the SOM.

        """
        return {
            **{
                'variables': self.variables,
                'metric': self.metric,
                'columns': self.columns,
                'rows': self.rows,
                'hexagonal': self.hexagonal,
                'toroidal': self.toroidal,
                'weights': self.weights,
                'relative_positions': self.relative_positions
            },
            **self.kwargs
        }

    def get_connections_and_reverse(self, min_cluster_size=3):
        """
        Get the distance matrix of all adjacent nodes that are in the same HDBSCAN cluster,
        and also get the matrix of connections that are between opposites edges (only for toroidal
        topologies).

        Parameters
        ---------
            min_cluster_size: int (optional, default 3)
                Minimum valid amount of nodes in each cluster.
                Should be greater than 0.

        Returns
        -------
            connections, reverse: ndarray, ndarray
                The first value is the distance matrix of all adjacent nodes
                that are in the same HDBSCAN cluster.
                Dimensions are (columns * rows, columns * rows).
                The second value is the matrix of connections that are between opposites edges.
                Dimensions are (columns * rows, columns * rows).

        """
        clusters_labels = array(self.hdbscan(
            min_cluster_size=min_cluster_size)
        )[0].reshape(self.columns * self.rows)
        adjacency_matrix = self.get_unified_distance_matrix()[1]
        connections = zeros(adjacency_matrix.shape) * nan
        reverse = zeros(adjacency_matrix.shape)
        reshaped_rp = self.relative_positions.reshape([-1, 2])

        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if not isnan(adjacency_matrix[i, j]) and clusters_labels[i] == clusters_labels[j] \
                        and clusters_labels[i] >= 0 and i != j:
                    connections[i, j] = adjacency_matrix[i, j]

                    distance = ((reshaped_rp[i, 0] - reshaped_rp[j, 0]) ** 2 +
                                (reshaped_rp[i, 1] - reshaped_rp[j, 1]) ** 2) ** 0.5
                    if self.toroidal and distance >= min(self.columns, self.rows) / 2:
                        reverse[i, j] = 1
                        reverse[j, i] = 1

        return connections, reverse

    def plot_analysis(self, data, attached_values=None, labels_to_display=None,
                      aggregation_function=None, use_relative_positions=True, cluster=True,
                      min_cluster_size=3, borders=True, title='Activation Frequency',
                      display_value=None, display_empty_nodes=True, size=10):
        """
        Plot the mapped data in each node with diameters according to their activation frequency.

        Parameters
        ----------
        data: ndarray
            Observations to map in the SOM.
            Dimensions should be (n_obs, variables).
        attached_values: array_like (optional, default None)
            Array containing the data to map in the nodes.
            Length should be n_obs.
        labels_to_display: array_like (optional, default None)
            Array with labels to display.
            Useful when attached_values are data labels and it's not necessary to plot all labels
            but it's necessary to maintain the activation frequency with respect to all data.
            Each label to show should be passed only once.
            Ignored if aggregation_function is not None.
        aggregation_function: function (optional, default None)
            Aggregation function to apply after map the attached values in the nodes.
        use_relative_positions: bool (optional, default True)
            Place nodes according to their relative positions.
        cluster: bool (optional, default True)
            Display the connections between adjacent nodes that are in the same HDBSCAN cluster.
        min_cluster_size: int (optional, default 3)
            Minimum valid amount of nodes in each HDBSCAN cluster.
            Should be greater than 0.
            Ignored if cluster is 0.
        borders: bool (optional, default True)
            Draw nodes borders.
        title: string (optional, default 'Activation Frequency')
            Plot title
        display_value: string (optional, default None)
            Value to display for each node.
            Should be one of the followings:
                * index
                * cluster
                * act_freq
        display_empty_nodes: bool (optional, default True)
            Display nodes that have 0 activation frequency.
        size: int (optional, default 10)
            Horizontal and vertical size of the plot.

        """
        _check_inputs.value_type(use_relative_positions, bool)
        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(title, str)
        _check_inputs.value_type(display_value, (str, type(None)))
        _check_inputs.value_type(display_empty_nodes, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)

        analysis = self.analyse(data)
        activation_frequency = analysis[0]
        quantization_error = analysis[1]

        if cluster:
            connections, reverse = self.get_connections_and_reverse(min_cluster_size)

        else:
            connections, reverse = None, None

        if use_relative_positions:
            positions = self.relative_positions

        else:
            positions = self.positions

        if attached_values is not None:
            if aggregation_function is not None:
                unique_labels = None
                map_values = self.map_attachments(data, attached_values, aggregation_function)
                normalize = True

            else:
                unique_labels = unique(attached_values)

                def aggregation_function(item):
                    res = zeros(unique_labels.shape[0])
                    counted = Counter(item)

                    for k, unique_label in enumerate(unique_labels):
                        res[k] = counted[unique_label]

                    return res

                map_values = self.map_attachments(data, attached_values, aggregation_function)
                normalize = False
                if labels_to_display is not None:
                    labels_to_display_indices = where(isin(unique_labels, labels_to_display))[0]
                    unique_labels = unique_labels[labels_to_display_indices]
                    _check_inputs.positive(len(labels_to_display_indices))
                    map_values = map_values[..., labels_to_display_indices].reshape([
                        self.columns,
                        self.rows,
                        len(labels_to_display_indices)
                    ])

        else:
            unique_labels = None
            map_values = quantization_error
            normalize = True

        text = None
        if display_value == 'index':
            text = [[(i, j) for j in range(self.rows)] for i in range(self.columns)]

        if display_value == 'cluster':
            text = self.hdbscan(min_cluster_size=min_cluster_size)[0]

        if display_value == 'act_freq':
            text = activation_frequency.astype(int)

        _plot.bubbles(activation_frequency, positions, map_values, size=size, text=text,
                      borders=borders, norm=normalize, labels=unique_labels, title=title,
                      connections=connections, reverse=reverse,
                      display_empty_nodes=display_empty_nodes)

    def plot_unified_distance_matrix(self, detailed=True, borders=True, title='U-matrix',
                                     min_cluster_size=3, display_value=None, size=10):
        """
        Plot the u-matrix.

        Parameters
        ----------
        detailed: bool (optional, default True)
            Whether it must plot the distance between each node and their adjacent ones
            or just the average.
        borders: bool (optional, default True)
            Draw nodes borders.
        title: string (optional, default 'Weight vectors')
            Plot title
        min_cluster_size: int (optional, default 3)
            Minimum valid amount of nodes in each HDBSCAN cluster.
            Should be greater than 0.
            Ignored if cluster is 0.
        display_value: string (optional, default None)
            Value to display for each node.
            Should be one of the followings:
                * index
                * cluster
        size: int (optional, default 10)
            Horizontal and vertical size of the plot.

        """
        _check_inputs.value_type(detailed, bool)
        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(title, str)
        _check_inputs.value_type(display_value, (str, type(None)))
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)
        unified_distance_matrix = self.get_unified_distance_matrix()[0]

        text = None
        if display_value == 'index':
            text = [[(i, j) for j in range(self.rows)] for i in range(self.columns)]

        if display_value == 'cluster':
            text = self.hdbscan(min_cluster_size=min_cluster_size)[0]

        if detailed:
            _plot.tiles(self.positions, self.hexagonal, unified_distance_matrix,
                        title=title, borders=borders, size=size, text=text)

        else:
            _plot.tiles(self.positions, self.hexagonal, unified_distance_matrix[..., -1],
                        title=title, borders=borders, size=size, text=text)

    def plot_weights(self, scaler=None, weights_to_display=None, headers=None, borders=True,
                     title='Weight vectors', use_relative_positions=True, cluster=True,
                     min_cluster_size=3, display_value=None, size=10):
        """
        Plot the nodes weights vectors.

        Parameters
        ----------
        scaler: scaler (optional, default None)
            sklearn scaler to apply inverse transform to weights.
        weights_to_display: list (optional, default None)
            List of weights to display.
        headers: list (optional, default None)
            List of titles to put to each plot. They could be the original name of the weights.
        borders: bool (optional, default True)
            Draw nodes borders.
        title: string (optional, default 'Weight vectors')
            Plot title
        use_relative_positions: bool (optional, default True)
            Place nodes according to their relative positions.
        cluster: bool (optional, default True)
            Display the connections between adjacent nodes that are in the same HDBSCAN cluster.
        min_cluster_size: int (optional, default 3)
            Minimum valid amount of nodes in each HDBSCAN cluster.
            Should be greater than 0.
            Ignored if cluster is 0.
        display_value: string (optional, default None)
            Value to display for each node.
            Should be one of the followings:
                * index
                * cluster
                * act_freq
        size: int (optional, default 10)
            Horizontal and vertical size of the plot.

        """
        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(title, str)
        _check_inputs.value_type(display_value, (str, type(None)))
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)

        if scaler is None:
            weights = self.weights.copy()

        else:
            weights = scaler.inverse_transform(self.weights.reshape(-1, self.variables)) \
                .reshape(self.weights.shape)

        if cluster:
            connections, reverse = self.get_connections_and_reverse(min_cluster_size)

        else:
            connections, reverse = None, None

        if use_relative_positions:
            positions = self.relative_positions

        else:
            positions = self.positions

        if weights_to_display is None:
            weights_to_display = list(range(self.variables))

        if headers is None:
            headers = list(range(self.variables))

        text = None
        if display_value == 'index':
            text = [[(i, j) for j in range(self.rows)] for i in range(self.columns)]

        if display_value == 'cluster':
            text = self.hdbscan(min_cluster_size=min_cluster_size)[0]

        _plot.bubbles(zeros((self.columns, self.rows)) + 1., positions,
                      self.weights[..., weights_to_display] / self.weights.sum(axis=-1)[..., None],
                      title=title, labels=list(headers[i] for i in weights_to_display), norm=False,
                      borders=borders, size=size, text=text, connections=connections,
                      reverse=reverse)

        for weight in weights_to_display:
            _plot.tiles(self.positions, self.hexagonal, weights[..., weight],
                        title=headers[weight], borders=borders, size=size, text=text)

    def plot_weights_vector(self, node_index=(0, 0), scaler=None, xticks_labels=None, bars=True,
                            line=False, scatter=False, size=10):
        """
        Plot the weights vector of a node.

        Parameters
        ---------
        node_index: tuple (optional, default (0, 0))
            Index of the node (column, row).
        scaler: scaler (optional, default None)
            sklearn scaler to apply inverse transform to weights.
        xticks_labels: list (optional, default None)
            List of weights labels.
        bars: bool (optional, default True)
            Plot the weights vector using bars.
        line: bool (optional, default False)
            Plot the weights vector using a line.
        scatter: bool (optional, default False)
            Plot the weights vector using points.
        size: int (optional, default 10)
            Horizontal and vertical size of the plot.

        """
        _check_inputs.value_type(node_index, tuple)
        _check_inputs.value_type(bars, bool)
        _check_inputs.value_type(line, bool)
        _check_inputs.value_type(scatter, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)
        node = self.weights[node_index]

        if scaler is not None:
            node = scaler.inverse_transform([node])[0]

        plt.figure(figsize=(size, size))

        if bars:
            plt.bar(list(range(self.variables)), node, color='blue', zorder=1)

        if line:
            plt.plot(list(range(self.variables)), node, color='green', zorder=2)

        if scatter:
            plt.scatter(list(range(self.variables)), node, color='red', zorder=3)

        plt.title("Node " + str(node_index) + " weights vector")

        if xticks_labels is None:
            xticks_labels = list(range(self.variables))

        plt.xticks(ticks=list(range(self.variables)), labels=xticks_labels, rotation='vertical')

    def plot_cluster_weights_vectors(self, cluster=None, scaler=None, xticks_labels=None,
                                     min_cluster_size=3, display_median=False, display_mean=True,
                                     display_lines=True, size=10):
        """
        Plot the weights vector of a HDBSCAN cluster of nodes, or for all nodes.
        Also plot the nodes weights vectors median or mean.

        Parameters
        ----------
        cluster: int (optional, default None)
            HDBSCAN cluster number to display.
            If nothing is provided, plot all nodes.
        scaler: scaler (optional, default None)
            sklearn scaler to apply inverse transform to weights.
        xticks_labels: list (optional, default None)
            List of weights labels.
        min_cluster_size: int (optional, default 3)
            Minimum valid amount of nodes in each cluster.
            Should be greater than 0.
        display_median: bool (optional, default False)
            Plot the median of the weights vectors.
        display_mean: bool (optional, default True)
            Plot the mean of the weights vectors.
        display_lines: bool (optional, default True)
            Plot the weights vector using a line.
        size: int (optional, default 10)
            Horizontal and vertical size of the plot.

        """
        _check_inputs.value_type(display_median, bool)
        _check_inputs.value_type(display_mean, bool)
        _check_inputs.value_type(display_lines, bool)
        labels = self.hdbscan(plot_condensed_tree=False, min_cluster_size=min_cluster_size)[0]

        if scaler is None:
            values = self.weights.reshape((-1, self.variables))

        else:
            values = scaler.inverse_transform(self.weights.reshape(-1, self.variables))

        cluster_title = "Weights vectors"
        alpha = 1.

        if cluster is not None:
            _check_inputs.value_type(cluster, int)
            values = values[array(labels == cluster).flatten()]
            cluster_title += ". Cluster " + str(cluster)

        if display_mean:
            alpha = 0.5
            cluster_title += '. Mean, M + 2 std y M - 2 std'

        if display_median:
            alpha = 0.5
            cluster_title += '. Median, Q1 y Q3'

        plt.figure(figsize=(size, size))

        for value in values:
            plt.scatter(list(range(self.variables)), value, alpha=alpha)

            if display_lines:
                plt.plot(value, alpha=alpha / 2, zorder=-1)

        if display_median:
            plt.scatter(list(range(self.variables)), quantile(values, 0.5, axis=0),
                        color='black', s=50)
            plt.scatter(list(range(self.variables)), quantile(values, 0.75, axis=0),
                        color='black', s=50)
            plt.scatter(list(range(self.variables)), quantile(values, 0.25, axis=0),
                        color='black', s=50)

            if display_lines:
                plt.plot(quantile(values, 0.5, axis=0), color='black', lw=2, zorder=-1)
                plt.plot(quantile(values, 0.75, axis=0), color='black', lw=1, ls='--', zorder=-1)
                plt.plot(quantile(values, 0.25, axis=0), color='black', lw=1, ls='--', zorder=-1)

        if display_mean:
            v_mean = values.mean(axis=0)
            v_std = values.std(axis=0)
            plt.scatter(list(range(self.variables)), v_mean, color='black', s=50)
            plt.scatter(list(range(self.variables)), v_mean + 2 * v_std, color='black', s=50)
            plt.scatter(list(range(self.variables)), v_mean - 2 * v_std, color='black', s=50)

            if display_lines:
                plt.plot(v_mean, color='black', lw=2)
                plt.plot(v_mean + 2 * v_std, color='black', lw=1, ls='--')
                plt.plot(v_mean - 2 * v_std, color='black', lw=1, ls='--')

        plt.title(cluster_title)

        if xticks_labels is None:
            xticks_labels = list(range(self.variables))

        plt.xticks(ticks=list(range(self.variables)), labels=xticks_labels, rotation='vertical')

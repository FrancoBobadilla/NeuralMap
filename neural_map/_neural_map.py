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
    cov, argsort, linspace, fill_diagonal, nan, nan_to_num, mean, argmin, where, isin, unique, \
    quantile

from numpy.linalg import norm, eig

from scipy.spatial.distance import cdist

from hdbscan import HDBSCAN

from sklearn_extra.cluster import KMedoids

from neural_map import _plot, _check_inputs, _decay_functions, _neighbourhood_functions


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
    columns: int (optional, default 20)
        Number of horizontal nodes.
    rows: int (optional, default 20)
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
    seed: int (optional, default None)
        Random seed.
    **kwargs: dict (optional)
        Extra arguments for the distance ``metric``.

    """

    def __init__(self,
                 variables,
                 metric,
                 columns=20,
                 rows=20,
                 hexagonal=True,
                 toroidal=False,
                 relative_positions=None,
                 weights=None,
                 seed=None,
                 **kwargs
                 ):
        if seed is None:
            random.seed(None)
            seed = random.randint(low=0, high=10000)
            print('generated seed: ', seed)

        _check_inputs.value_type(variables, int)
        _check_inputs.positive(variables)
        _check_inputs.value_type(columns, int)
        _check_inputs.positive(columns)
        _check_inputs.value_type(rows, int)
        _check_inputs.positive(rows)
        _check_inputs.value_type(hexagonal, bool)
        _check_inputs.value_type(toroidal, bool)
        _check_inputs.value_type(seed, int)
        self._metric = metric
        self._kwargs = kwargs

        if isinstance(metric, str):
            def distance(first_array, second_array):
                return cdist(first_array, second_array, self._metric, **self._kwargs)

        else:
            def distance(first_array, second_array):
                return self._metric(first_array, second_array, **self._kwargs)

        self.distance = distance
        self.distance(array([[0., 1.]]), array([[1., 2.], [3., 4.]]))
        self.columns = columns
        self.rows = rows
        self.variables = variables
        self.hexagonal = hexagonal
        self.toroidal = toroidal
        self.width = self.columns
        self.height = self.rows
        self.seed = seed
        random.seed(self.seed)

        if weights is not None:
            _check_inputs.ndarray_and_shape(weights, (self.columns, self.rows, self.variables))

        else:
            weights = zeros((self.columns, self.rows, self.variables))

        self.weights = weights
        self.activation_map = zeros((self.columns, self.rows))
        self._adjacent_nodes_relative_positions = [[(1, 0), (0, 1), (-1, 0), (0, -1)],
                                                   [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        self.positions = transpose(meshgrid(arange(self.columns), arange(self.rows)),
                                   axes=[2, 1, 0]).astype(float)

        if self.hexagonal:
            self.positions[:, 0::2, 0] += 0.5
            self.positions[..., 1] *= (3 ** 0.5) * 0.5
            self.height *= (3 ** 0.5) * 0.5
            self._adjacent_nodes_relative_positions = [
                [(1, 0), (1, 1), (0, 1), (-1, 0), (0, -1), (1, -1)],
                [(1, 0), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1)]
            ]

        if relative_positions is not None:
            _check_inputs.ndarray_and_shape(
                relative_positions,
                (self.columns, self.rows, self.variables)
            )

        else:
            relative_positions = self.positions.copy()

        self.relative_positions = relative_positions
        self.current_epoch = None
        self._unified_distance_matrix_cache = None
        self._hdbscan_cache = [(None, None, None)] * self.columns * self.rows

    def pca_weights_init(self, data):
        """
        Initialize the weights to span the first two principal components.

        It is recommended to normalize the data before initializing the weights
        and use the same normalization for the training data.

        """
        principal_components_length, principal_components = eig(cov(transpose(data)))
        components_order = argsort(-principal_components_length)

        for i, component_1 in enumerate(
                linspace(-1, 1, self.columns) * principal_components[components_order[0]]):
            for j, component_2 in enumerate(
                    linspace(-1, 1, self.rows) * principal_components[components_order[1]]):
                self.weights[i, j] = component_1 + component_2

    def uniform_weights_init(self):
        """
        Randomly initialize the weights with values between 0 and 1.

        """
        self.weights = random.rand(self.columns, self.rows, self.variables)

    def standard_weights_init(self):
        """
        Randomly initialize the weights with a random distribution with mean of 0 and std of 1.

        """
        self.weights = random.normal(0., 1., (self.columns, self.rows, self.variables))

    def pick_from_data_weights_init(self, data):
        """
        Initialize the weights picking random samples from the input data.

        """
        indices = arange(data.shape[0])

        for i in range(self.columns):
            for j in range(self.rows):
                self.weights[i, j] = data[random.choice(indices)]

    def train(self,
              data,
              eval_data=None,
              n_epochs=100,
              weights_init_method='standard',
              initial_learning_rate=1.0,
              final_learning_rate=0.01,
              initial_radius=None,
              final_radius=1.0,
              learning_rate_decay_function='linear',
              radius_decay_function='linear',
              neighbourhood_function='gaussian',
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
        weights_init_method: string (optional, default 'standard')
            Method to initialize weights. It should be one of the followings:
                * pca_weights_init
                * uniform_weights_init
                * standard_weights_init
                * pick_from_data_weights_init
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
        verbose: bool (optional, default True)
            Verbosity of training process.

        """

        # if it's not provided a value, it's calculated as half the minimum dimension of the map
        if initial_radius is None:
            initial_radius = min(self.width, self.height) / 2

        # if the number of epochs is set to 1, use no decay function
        if n_epochs == 1:
            learning_rate_decay_function = _decay_functions.no_decay
            radius_decay_function = _decay_functions.no_decay

        # restart u-matrix and hdbscan cahces
        self._unified_distance_matrix_cache = None
        self._hdbscan_cache = [(None, None, None)] * self.columns * self.rows

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

        # check inputs
        _check_inputs.numpy_matrix(data, self.variables)

        if eval_data is not None:
            _check_inputs.numpy_matrix(eval_data, self.variables)

        _check_inputs.value_type(n_epochs, int)
        _check_inputs.positive(n_epochs)
        _check_inputs.value_type(initial_learning_rate, float)
        _check_inputs.positive(initial_learning_rate)
        _check_inputs.value_type(final_learning_rate, float)
        _check_inputs.positive(final_learning_rate)
        _check_inputs.value_type(initial_radius, float)
        _check_inputs.positive(initial_radius)
        _check_inputs.value_type(final_radius, float)
        _check_inputs.positive(final_radius)
        _check_inputs.function(learning_rate_decay_function)
        _check_inputs.function(radius_decay_function)
        _check_inputs.function(neighbourhood_function)
        _check_inputs.value_type(verbose, bool)

        # get the weights initialization method
        if weights_init_method == 'standard':
            self.standard_weights_init()

        elif weights_init_method == 'uniform':
            self.uniform_weights_init()

        elif weights_init_method == 'pca':
            self.pca_weights_init(data)

        elif weights_init_method == 'pick_from_data':
            self.pick_from_data_weights_init(data)

        # declare an array for quantization and topographic error of each epoch
        epochs_quantization_error = zeros(n_epochs)
        epochs_topographic_error = zeros(n_epochs)

        # generate observation indices to iterate over
        indices = arange(len(data))

        # set random state
        random.seed(self.seed)

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

        for epoch in range(n_epochs):

            # set current epoch
            self.current_epoch = epoch

            # shuffles all observations
            random.shuffle(indices)

            # get a new random state, for the next epoch
            random.seed(epoch)

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
                    # subtracting to the original indices the corresponding horizontal and vertival
                    # displacement. Also, if the topology is hexagonal and the row of the bmu is
                    # offset from the row of the map center, it must be applied an extra correction
                    # that is computed at the beginning of the training. Finally, the new positions
                    # are rotated to fit the map dimensions.

                    # computes the amount of columns and rows between the map center and the bmu
                    offset = [bmu[0] - center[0], bmu[1] - center[1]]

                    # compute the offset correction for each row
                    offset_correction = (self.hexagonal and offset[1] % 2) * correction

                    # rotate update matrix to match its center with the bmu position
                    update_matrix = update_matrix_over_center[tuple([
                        (update_matrix_indices[0] - (offset[0] + offset_correction)) % self.columns,
                        (update_matrix_indices[1] - offset[1][:, None]) % self.rows
                    ])]

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

                    # compute displacement for bmu relative position

                    # get opposite point of the bmu relative position
                    anti_bmu = (self.positions[bmu] + dimensions / 2) % dimensions

                    # get the 'quadrant' the bmu is in, with respect to its opposite position
                    quadrant = array([
                        offset[0] > 0 or anti_bmu[0] < 1,
                        offset[1] > 0 or anti_bmu[1] < 1
                    ]) * 2 - 1

                    # compute the matrix with the positions to which each node must tend
                    mod = (self.relative_positions[bmu] * quadrant < anti_bmu * quadrant) \
                        * dimensions * quadrant

                    # compute displacement for each node relative position
                    displacement[bmu] = (self.positions[bmu] - mod - self.relative_positions[bmu]) \
                        * update_matrix[bmu]

                    # plot for easier debugging
                    if plot_update:
                        plot_update = False
                        _plot.update(self.positions, self.hexagonal, update_matrix, dimensions,
                                     self.relative_positions[bmu], self.relative_positions,
                                     displacement)
                        plt.show()

                    # update relative positions
                    self.relative_positions += displacement

                    # fits new relative positions into the map
                    self.relative_positions %= dimensions

                else:

                    # compute update matrix, with the proportion of change for each node
                    update_matrix = neighbourhood_function(self.positions, self.positions[bmu],
                                                           radius, learning_rate)

                    # the relative positions of neighbourhood nodes tends towards
                    # the bmu relative position
                    displacement = (self.relative_positions[bmu] - self.relative_positions) \
                        * update_matrix[..., None]

                    # the bmu relative positions tends towards its absolute position
                    displacement[bmu] = (self.positions[bmu] - self.relative_positions[bmu]) \
                        * update_matrix[bmu]

                    # plot for easier debugging
                    if plot_update:
                        plot_update = False
                        _plot.update(self.positions, self.hexagonal, update_matrix, dimensions,
                                     self.positions[bmu], self.relative_positions, displacement)
                        plt.show()

                    # update relative positions
                    self.relative_positions += displacement

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
            self.weights.shape[0] * self.weights.shape[1],
            self.weights.shape[2]
        ])).reshape([self.weights.shape[0], self.weights.shape[1]])

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
        adjacency_count = len(self._adjacent_nodes_relative_positions[0])
        unified_distance_matrix = zeros((self.columns, self.rows, 1 + adjacency_count))

        for x_index in range(self.columns):
            for y_index in range(self.rows):
                adjacent_nodes = 0

                for k, (i, j) in enumerate(self._adjacent_nodes_relative_positions[y_index % 2]):
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
        dict_map = {(i, j): [] for i in range(self.rows) for j in range(self.columns)}

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

        if isinstance(self._metric, str):
            metric = self._metric

        else:
            def custom_metric(first_set, second_set):
                return self.distance(array([first_set]), array([second_set]))

            metric = custom_metric

        clusters = KMedoids(n_clusters=n_clusters, init="k-medoids++", metric=metric) \
            .fit(self.weights.reshape(self.columns * self.rows, self.variables))

        return clusters.labels_.reshape(self.columns, self.rows), clusters.cluster_centers_

    def hdbscan(self, min_cluster_size=3, plot_condensed_tree=True):
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
            plot_condensed_tree: bool (optional, default True)
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

            for (i, j) in self._adjacent_nodes_relative_positions[f_bmu[1] % 2]:
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
                'metric': self._metric,
                'columns': self.columns,
                'rows': self.rows,
                'hexagonal': self.hexagonal,
                'toroidal': self.toroidal,
                'seed': self.seed,
                'weights': self.weights,
                'relative_positions': self.relative_positions
            },
            **self._kwargs
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
            min_cluster_size=min_cluster_size,
            plot_condensed_tree=False)
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

                    if self.toroidal and norm(reshaped_rp[i] - reshaped_rp[j]) >= \
                            min(self.columns, self.rows) / 2:
                        reverse[i, j] = 1
                        reverse[j, i] = 1

        return connections, reverse

    def plot_analysis(self, data, cluster=True, min_cluster_size=3,
                      borders=True, display_empty_nodes=True, size=10):
        """
        Plot the quantization error and activation frequency of each node placed according to their
        relative positions and draw their connections if they are in the same HDBSCAN cluster.

        Parameters
        ----------
        data: ndarray
            Observations to map in the SOM.
            Dimensions should be (n_obs, variables).
        cluster: bool (optional, default True)
            Display the connections between adjacent nodes that are in the same HDBSCAN cluster.
        min_cluster_size: int (optional, default True) (optional, default 3)
            Minimum valid amount of nodes in each HDBSCAN cluster.
            Should be greater than 0.
            Ignored if ``cluster`` is False.
        borders: bool (optional, default True)
            Draw nodes borders.
        display_empty_nodes: bool (optional, default True)
            Display nodes that have 0 activation frequency.
        size: int (optional, default 10)
            Horizontal and vertical size of the plot.

        """
        connections, reverse = None, None

        if cluster:
            connections, reverse = self.get_connections_and_reverse(min_cluster_size)

        analysis = self.analyse(data)
        activation_frequency = analysis[0]
        quantization_error = analysis[1]
        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(display_empty_nodes, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)
        _plot.bubbles(activation_frequency, self.relative_positions, quantization_error, size=size,
                      borders=borders, title='RP-HDBSCAN quantization error',
                      connections=connections, reverse=reverse,
                      display_empty_nodes=display_empty_nodes)

    def plot_labels(self, data, labels, labels_to_display=None, cluster=True,
                    min_cluster_size=3, borders=True, display_empty_nodes=True, size=10):
        """
        Plot the labels of the mapped data in each node as pie charts,
        with diameters according to their activation frequency,
        placed according to their relative positions
        and draw their connections if they are in the same HDBSCAN cluster.

        Parameters
        ----------
        data: ndarray
            Observations to map in the SOM.
            Dimensions should be (n_obs, variables).
        labels: array_like
            Array containing the data labels to map in the nodes.
            Length should be n_obs.
        labels_to_display: array_like (optional, default None)
            Array with labels to display.
            Useful when it's not necessary to plot all labels but
            it's necessary to maintain the activation frequency with respect to all observations.
            Each label to show should be passed only once.
        cluster: bool (optional, default True)
            Display the connections between adjacent nodes that are in the same HDBSCAN cluster.
        min_cluster_size: int (optional, default True) (optional, default 3)
            Minimum valid amount of nodes in each HDBSCAN cluster.
            Should be greater than 0.
            Ignored if cluster is 0.
        borders: bool (optional, default True)
            Draw nodes borders.
        display_empty_nodes: bool (optional, default True)
            Display nodes that have 0 activation frequency.
        size: int (optional, default 10)
            Horizontal and vertical size of the plot.

        """
        activation_frequency = self.analyse(data)[0]
        connections, reverse = None, None

        if cluster:
            connections, reverse = self.get_connections_and_reverse(min_cluster_size)

        unique_labels = unique(labels)
        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(display_empty_nodes, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)

        def aggregation_function(item):
            res = zeros(unique_labels.shape[0])
            counted = Counter(item)

            for k, unique_label in enumerate(unique_labels):
                res[k] = counted[unique_label]

            return res

        map_labels = self.map_attachments(data, labels, aggregation_function)

        if labels_to_display is None:
            _plot.bubbles(activation_frequency, self.relative_positions, map_labels,
                          color_map=plt.cm.get_cmap('hsv', len(unique_labels) + 1), size=size,
                          borders=borders, norm=False, labels=unique_labels,
                          title='RP-HDBSCAN labels', connections=connections, reverse=reverse,
                          display_empty_nodes=display_empty_nodes)

        else:
            labels_to_display_indices = where(isin(unique_labels, labels_to_display))[0]
            _check_inputs.positive(len(labels_to_display_indices))
            map_labels = map_labels[..., labels_to_display_indices].reshape([
                activation_frequency.shape[0],
                activation_frequency.shape[1],
                len(labels_to_display_indices)
            ])
            _plot.bubbles(activation_frequency, self.relative_positions, map_labels,
                          color_map=plt.cm
                          .get_cmap('hsv', len(unique_labels[labels_to_display_indices]) + 1),
                          size=size, borders=borders, norm=False,
                          labels=unique_labels[labels_to_display_indices],
                          title='RP-HDBSCAN labels', connections=connections, reverse=reverse,
                          display_empty_nodes=display_empty_nodes)

    def plot_map_value(self, data, attached_values, func=mean, cluster=True, min_cluster_size=3,
                       borders=True, display_empty_nodes=True, size=10):
        """
        Plot data related values mapped in each node,
        with diameters according to their activation frequency,
        placed according to their relative positions
        and draw their connections if they are in the same HDBSCAN cluster.

        Parameters
        ----------
        data: ndarray
            Observations to map in the SOM.
            Dimensions should be (n_obs, variables).
        attached_values: array_like
            Array containing the values associated to each observation.
            Length should be n_obs.
        func: function (optional, default mean)
            Aggregation function to apply after map the attached values in the nodes.
        cluster: bool (optional, default True)
            Display the connections between adjacent nodes that are in the same HDBSCAN cluster.
        min_cluster_size: int (optional, default True) (optional, default 3)
            Minimum valid amount of nodes in each HDBSCAN cluster.
            Should be greater than 0.
            Ignored if cluster is 0.
        borders: bool (optional, default True)
            Draw nodes borders.
        display_empty_nodes: bool (optional, default True)
            Display nodes that have 0 activation frequency.
        size: int (optional, default 10)
            Horizontal and vertical size of the plot.

        """
        activation_frequency = self.analyse(data)[0]
        connections, reverse = None, None

        if cluster:
            connections, reverse = self.get_connections_and_reverse(min_cluster_size)

        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(display_empty_nodes, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)
        nodes_values = self.map_attachments(data, attached_values, func)
        _plot.bubbles(activation_frequency, self.relative_positions, nodes_values, size=size,
                      borders=borders, norm=True, title='RP-HDBSCAN attached values',
                      connections=connections, reverse=reverse,
                      display_empty_nodes=display_empty_nodes)

    def plot_unified_distance_matrix(self, detailed=True, borders=True, size=10):
        """
        Plot the u-matrix.

        Parameters
        ----------
        detailed: bool (optional, default True)
            Whether it must plot the distance between each node and their adjacent ones
            or just the average.
        borders: bool (optional, default True)
            Draw nodes borders.
        size: int (optional, default 10)
            Horizontal and vertical size of the plot.

        """
        _check_inputs.value_type(detailed, bool)
        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)
        unified_distance_matrix = self.get_unified_distance_matrix()[0]

        if detailed:
            _plot.tiles(self.positions, self.hexagonal, unified_distance_matrix,
                        title='Distance between nodes', borders=borders, size=size)

        else:
            _plot.tiles(self.positions, self.hexagonal, unified_distance_matrix[..., -1],
                        title='Distance between nodes', borders=borders, size=size)

    def plot_weights(self, scaler=None, weights_to_display=None, headers=None, borders=True,
                     size=10):
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
        size: int (optional, default 10)
            Horizontal and vertical size of the plot.

        """
        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)

        if scaler is None:
            weights = self.weights.copy()

        else:
            weights = scaler.inverse_transform(self.weights.reshape(-1, self.weights.shape[2])) \
                .reshape(self.weights.shape)

        if weights_to_display is None:
            weights_to_display = list(range(weights.shape[2]))

        if headers is None:
            headers = weights_to_display

        _check_inputs.length(weights_to_display, headers)

        for weight in weights_to_display:
            _plot.tiles(self.positions, self.hexagonal, weights[..., weight],
                        title=headers[weight], borders=borders, size=size)

    def plot_weights_vector(self, node_index=(0, 0), xticks_labels=None, bars=True,
                            scatter=False, line=False):
        """
        Plot the weights vector of a node.

        Parameters
        ---------
        node_index: tuple (optional, default (0, 0))
            Index of the node (column, row).
        xticks_labels: list (optional, default None)
            List of weights labels.
        bars: bool (optional, default True)
            Plot the weights vector using bars.
        scatter: bool (optional, default False)
            Plot the weights vector using points.
        line: bool (optional, default False)
            Plot the weights vector using a line.

        """
        _check_inputs.value_type(node_index, tuple)
        _check_inputs.value_type(bars, bool)
        _check_inputs.value_type(scatter, bool)
        _check_inputs.value_type(line, bool)
        node = self.weights[node_index]
        plt.figure(figsize=(20, 5))

        if bars:
            plt.bar(list(range(node.shape[0])), node)

        if scatter:
            plt.scatter(list(range(node.shape[0])), node)

        if line:
            plt.plot(list(range(node.shape[0])), node)

        plt.title("Node " + str(node_index) + " weights vector")

        if xticks_labels is None:
            xticks_labels = list(range(node.shape[0]))

        plt.xticks(ticks=list(range(node.shape[0])), labels=xticks_labels, rotation='vertical')

    def plot_cluster_weights_vectors(self, cluster=None, xticks_labels=None, min_cluster_size=3,
                                     display_median=False, display_mean=True, display_lines=True):
        """
        Plot the weights vector of a HDBSCAN cluster of nodes, or for all nodes.
        Also plot the nodes weights vectors median or mean.

        Parameters
        ----------
        cluster: int (optional, default None)
            HDBSCAN cluster number to display.
            If nothing is provided, plot all nodes.
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

        """
        _check_inputs.value_type(display_median, bool)
        _check_inputs.value_type(display_mean, bool)
        _check_inputs.value_type(display_lines, bool)
        labels = self.hdbscan(plot_condensed_tree=False, min_cluster_size=min_cluster_size)[0]
        values = self.weights.reshape((-1, self.variables))
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

        plt.figure(figsize=(20, 5))

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

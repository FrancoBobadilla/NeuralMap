from collections import Counter

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

from numpy import arange, random, zeros, array, unravel_index, isnan, meshgrid, transpose, ogrid, \
    cov, argsort, \
    linspace, fill_diagonal, nan, nan_to_num, mean, argmin, where, isin, unique, quantile

from numpy.linalg import norm, eig

from scipy.spatial.distance import cdist

from hdbscan import HDBSCAN

from sklearn_extra.cluster import KMedoids

from . import _plot, _check_inputs, _decay_functions, _neighbourhood_functions


def _identity(item):
    return item


class NeuralMap:
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

        # si no se ingresa una semilla aleatoria...
        if seed is None:
            # ... inicializa un generador aleatorio
            random.seed(None)

            # se crea una semilla, de forma aleatoria
            seed = random.randint(low=0, high=10000)

            # muestra al usuario la semilla generada
            print('generated seed: ', seed)

        # se checkean todos los datos ingresados
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

        # check if input metric is valid
        self.distance(array([[0., 1.]]), array([[1., 2.], [3., 4.]]))

        # se configura la cantidad horizontal de nodos
        self.columns = columns

        # se configura la cantidad vertical de nodos
        self.rows = rows

        # se configura la cantidad de elementos de los vectores de pesos de los nodos
        self.variables = variables

        # configura si la topología del mapa es hexagonal (o cuadrada)
        self.hexagonal = hexagonal

        # configura si la topología del mapa es toroidal (o plana)
        self.toroidal = toroidal

        # configura el ancho del mapa
        self.width = self.columns

        # configura el alto del mapa
        self.height = self.rows

        # configura la semilla aleatoria
        self.seed = seed

        # se configura el estado aleatorio en base a la semilla calculada en el constructor
        random.seed(self.seed)

        if weights is not None:
            _check_inputs.ndarray_and_shape(weights, (self.columns, self.rows, self.variables))
        else:
            weights = zeros((self.columns, self.rows, self.variables))

        self.weights = weights

        # crea el mapa de activación, con valores iniciales de 0
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
                [(1, 0), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1)]]

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
        principal_components_length, principal_components = eig(cov(transpose(data)))
        components_order = argsort(-principal_components_length)
        for i, component_1 in enumerate(
                linspace(-1, 1, self.columns) * principal_components[components_order[0]]):
            for j, component_2 in enumerate(
                    linspace(-1, 1, self.rows) * principal_components[components_order[1]]):
                self.weights[i, j] = component_1 + component_2

    def uniform_weights_init(self):
        self.weights = random.rand(self.columns, self.rows, self.variables)

    def standard_weights_init(self):
        self.weights = random.normal(0., 1., (self.columns, self.rows, self.variables))

    def pick_from_data_weights_init(self, data):
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

        # si no se ingresó un rado inicial...
        if initial_radius is None:
            # ... se calcula como la mitad de la mínima dimensión del mapa
            initial_radius = min(self.width, self.height) / 2

        # si se ingresó solo una época de entrenamiento...
        if n_epochs == 1:
            # ... no se utiliza función de disminución para learning rate ni radius
            learning_rate_decay_function = _decay_functions.no_decay
            radius_decay_function = _decay_functions.no_decay

        self._unified_distance_matrix_cache = None
        self._hdbscan_cache = [None] * self.columns * self.rows

        if learning_rate_decay_function == 'linear':
            learning_rate_decay_function = _decay_functions.linear
        elif learning_rate_decay_function == 'exponential':
            learning_rate_decay_function = _decay_functions.exponential
        elif learning_rate_decay_function == 'rational':
            learning_rate_decay_function = _decay_functions.rational
        elif learning_rate_decay_function == 'no_decay':
            learning_rate_decay_function = _decay_functions.no_decay

        if radius_decay_function == 'linear':
            radius_decay_function = _decay_functions.linear
        elif radius_decay_function == 'exponential':
            radius_decay_function = _decay_functions.exponential
        elif radius_decay_function == 'rational':
            radius_decay_function = _decay_functions.rational
        elif radius_decay_function == 'no_decay':
            radius_decay_function = _decay_functions.no_decay

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

        # se checkean los datos ingresados
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

        if weights_init_method == 'standard':
            self.standard_weights_init()

        elif weights_init_method == 'uniform':
            self.uniform_weights_init()

        elif weights_init_method == 'pca':
            self.pca_weights_init(data)

        elif weights_init_method == 'pick_from_data':
            self.pick_from_data_weights_init(data)

        # se declara la lista que se va a utilizar para almacenar la distancia promedio de los
        # bmu por cada época
        epochs_quantization_error = zeros(n_epochs)
        epochs_topographic_error = zeros(n_epochs)

        # se genera la lista de índices de los datos de entrada sobre el que se va a iterar
        indices = arange(len(data))

        # se configura el estado aleatorio en base a la semilla calculada en el constructor
        random.seed(self.seed)

        #############################

        # se declara el arreglo que va a contener las dimensiones del mapa
        dimensions = array([self.width, self.height])

        # se declara la posición del centro del mapa
        center = (self.columns // 2, self.rows // 2)

        # se declara la lista sobre la que se va a almacenar el desplazamiento horizontal de cada
        # itereación
        y_displacement = zeros(self.rows, dtype='int')

        # se declara la lista sobre la que se va a almacenar el desplazamiento vertical de cada
        # itereación
        x_displacement = zeros(self.columns, dtype='int')

        # se almcenan en una variable los índices de la mariz g
        updata_matrix_indices = ogrid[[slice(0, self.columns), slice(0, self.rows)]]

        # se inicializa con ceros la lista de correcciones en las posiciones de los nodos
        correction = zeros(self.rows, dtype='int')

        # se calcula la corrección que debe tener cada fila cuando el bmu está defasado del centro
        correction[center[1] % 2:: 2] = (center[1] % 2) * 2 - 1

        # por cada época...
        for epoch in range(n_epochs):

            # ... se mezclan nuevamente las iteraciones
            random.shuffle(indices)

            # se genera un nuevo estado aleatorio
            random.seed(epoch)

            # se calcula la Learning Rate de la época
            learning_rate = learning_rate_decay_function(initial_learning_rate, final_learning_rate,
                                                         n_epochs, epoch)

            # se calcula el Radius de la época
            radius = radius_decay_function(initial_radius, final_radius, n_epochs, epoch)

            self.current_epoch = epoch

            if verbose:
                # se imprime el progreso
                print('\nEpoch: ', epoch + 1, ' of ', n_epochs,
                      '    Learning rate: ', learning_rate,
                      '    Radius: ', radius)

            plot_update = False

            # plt.scatter(self._rp[..., 0], self._rp[..., 1])
            # plt.show()
            # plt.scatter((self._rp[..., 0] + self._width / 3) % self._width,
            #             (self._rp[..., 1] + self._height / 3) % self._height)
            # plt.show()

            update_matrix_over_center = neighbourhood_function(self.positions,
                                                               self.positions[center],
                                                               radius, learning_rate)

            # para cada iteración...
            for i in indices:
                # ... se toma el dato correspondiente a la iteración
                ind = data[i]

                # se calcula el nodo ganador, y su distancia con respecto al dato
                bmu = self.get_best_matching_unit(ind)

                if self.toroidal:

                    # se calcula la cantidad de nodos horizontal y vertical
                    # que hay desde el centro del mapa al nodo ganador
                    relative_indices = [bmu[0] - center[0], bmu[1] - center[1]]

                    # se llena con el valur u la lista de desplazamientos horizontales
                    y_displacement.fill(relative_indices[0])

                    # se llena con el valur v la lista de desplazamientos verticales
                    x_displacement.fill(relative_indices[1])

                    # La matriz de actualizaciones (g) calculada sobre el centro del mapa debe
                    # ser desplazada para que coincida con la bmu. De esta manera se logra emular
                    # un espacio toroidal. Para eso se reindexan los elementos del arreglo,
                    # restando de los índices originales (all_idcs) el desplazamiento horizontal y
                    # vertical correspondiente (hor_disp, ver_disp). Además, cuando se trata de
                    # topología hexagonal y el bmu está en una fila "defasada" del centro del mapa,
                    # se debe hacer una corrección adicional, que es calculada al principio del
                    # entrenamiento. Finalmente se calcula el resto con respecto a la cantidad de
                    # nodos horizontal y verticalmente, para asegurar que los nuevos índices se
                    # encuentren dentro de los límites.

                    exceptions = (self.hexagonal and relative_indices[1] % 2) * correction

                    # se reindexa la matriz g para que coincida con el nodo ganador
                    update_matrix = update_matrix_over_center[tuple([
                        (updata_matrix_indices[0] - (y_displacement + exceptions)) % self.columns,
                        (updata_matrix_indices[1] - x_displacement[:, None]) % self.rows
                    ])]

                    # se calcula la ubicación opuesta al nodo ganador
                    anti_bmu = (self.relative_positions[bmu] + dimensions / 2) % dimensions

                    # se calcula el cuadrante en el que está el nodo ganador con respecto a su
                    # ubicación opuesta
                    cuad = array(
                        [self.relative_positions[bmu][0] - self.width / 2 > 0 or anti_bmu[0] < 1,
                         self.relative_positions[bmu][1] - self.height / 2 > 0 or anti_bmu[
                             1] < 1]) * 2 - 1

                    # se calcula la matriz donde se indica la ubicación del nodo ganador al que
                    # debe tender cada RP
                    mod = (self.relative_positions * cuad < anti_bmu * cuad) * dimensions * cuad

                    # se calcula el desplazamiento que debe aplicarse a cada RP
                    # einsum('ij, ijk->ijk', g, (self._cart_coord[winner_node] - mod - self._rp))
                    displacement = (self.relative_positions[bmu] - mod -
                                    self.relative_positions) * update_matrix[..., None]

                    anti_bmu = (self.positions[bmu] + dimensions / 2) % dimensions
                    cuad = array([
                        relative_indices[0] > 0 or anti_bmu[0] < 1,
                        relative_indices[1] > 0 or anti_bmu[1] < 1
                    ]) * 2 - 1
                    mod = (self.relative_positions[
                               bmu] * cuad < anti_bmu * cuad) * dimensions * cuad
                    displacement[bmu] = (self.positions[bmu] -
                                         mod - self.relative_positions[bmu]) * update_matrix[bmu]

                    if plot_update:
                        plot_update = False
                        _plot.update(self.positions, self.hexagonal, update_matrix, dimensions,
                                     self.relative_positions[bmu], self.relative_positions,
                                     displacement)
                        plt.show()
                        # print((self._width, self._height, winner_node, self._cart_coord,
                        # center, g, self._rp, '\n\n'))

                    # se actualiza la matriz de RP
                    self.relative_positions += displacement

                    # se corrige para que entre en las dimensiones del mapa
                    self.relative_positions %= dimensions

                else:

                    # se calcula la matriz de actualizaciones
                    update_matrix = neighbourhood_function(self.positions,
                                                           self.positions[bmu],
                                                           radius,
                                                           learning_rate)

                    # se calcula el desplazamiento que debe aplicarse a cada RP
                    # despl = einsum('ij, ijk->ijk', g, winner_node - self._rp)

                    # displacement = (self._positions[best_matching_unit] - self._rp) *
                    # update_matrix[..., None]
                    displacement = (self.relative_positions[bmu] -
                                    self.relative_positions) * update_matrix[..., None]
                    displacement[bmu] = (self.positions[bmu] -
                                         self.relative_positions[bmu]) * update_matrix[bmu]

                    if plot_update:
                        plot_update = False
                        _plot.update(self.positions, self.hexagonal, update_matrix, dimensions,
                                     self.positions[bmu],
                                     self.relative_positions, displacement)
                        plt.show()

                    # se actualiza la matriz de RP
                    self.relative_positions += displacement

                # se actualizan los pesos de la red
                self.weights += (ind - self.weights) * update_matrix[..., None]

            if eval_data is not None:
                epochs_quantization_error[epoch], epochs_topographic_error[epoch] = self.evaluate(
                    eval_data)

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

        # se calcula el mapa de activación con el dato ingresado
        self.activation_map = self.distance(ind.reshape([1, -1]), self.weights.reshape(
            [self.weights.shape[0] * self.weights.shape[1], self.weights.shape[2]])).reshape(
            [self.weights.shape[0], self.weights.shape[1]])

        # devuelve el mapa de activación
        return self.activation_map

    def get_best_matching_unit(self, ind):

        # se devuelve la posición bidimensional del nodo ganador
        return unravel_index(self.generate_activation_map(ind).argmin(), self.activation_map.shape)

    def get_unified_distance_matrix(self):

        if self._unified_distance_matrix_cache is not None:
            return self._unified_distance_matrix_cache

        adjacency_matrix = zeros((self.columns * self.rows, self.columns * self.rows)) * nan
        fill_diagonal(adjacency_matrix, 0.)

        # se calcula la cantidad de nodos adyacentes que tiene cada nodo
        adjacency_length = len(self._adjacent_nodes_relative_positions[0])

        # se inicializa la matriz de distancias unificadas con ceros
        unified_distance_matrix = zeros((self.columns, self.rows, 1 + adjacency_length))

        # para cada posición en x del mapa...
        for x_index in range(self.columns):

            # ... para cada posición en y del mapa...
            for y_index in range(self.rows):

                # ... se configura en 0 el contador de la cantidad de nodos adyacentes válidos
                adjacent_nodes = 0

                # para cada tupla de coordenadas relativas al nodo...
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

                        # se calcula la distancia entre el nodo y su vecino
                        distance = self.distance(self.weights[x_index, y_index].reshape([1, -1]),
                                                 neighbour[0, 0].reshape([1, -1]))

                        # se lo suma la distancia al valor que tiene la matriz para esa posición
                        unified_distance_matrix[x_index, y_index, k] = distance

                        # se asigna la distancia calcular al valor que tiene la matriz para esa
                        # posición
                        unified_distance_matrix[x_index, y_index, adjacency_length] += distance

                        # se incrementa en 1 la cuenta de nodos adyacentes, para calcular el
                        # promedio
                        adjacent_nodes += 1

                        adjacency_matrix[x_index * self.rows + y_index,
                                         ((x_index + i + self.columns) % self.columns) * self.rows +
                                         (y_index + j + self.rows) % self.rows] = distance

                    else:
                        # ... se asigna nan a esa posición de la matriz
                        unified_distance_matrix[x_index, y_index, k] = nan

                # si el nodo tiene al menos un nodo adyacente...
                if adjacent_nodes > 0:

                    # ... se aclcula el promedio de las distancias
                    new_value = unified_distance_matrix[x_index, y_index, adjacency_length] / \
                                adjacent_nodes
                    unified_distance_matrix[x_index, y_index, adjacency_length] = new_value

                # si no tiene ningún vecino...
                else:

                    # ... se asigna nan
                    unified_distance_matrix[x_index, y_index, adjacency_length] = nan

        self._unified_distance_matrix_cache = unified_distance_matrix, adjacency_matrix
        return self._unified_distance_matrix_cache

    def analyse(self, data):

        # se checkea el conjunto de datos ingresado
        _check_inputs.numpy_matrix(data, self.variables)

        # se inicializa con ceros la matriz que representa la frecuencia de activación
        activation_frequency = zeros((self.columns, self.rows))

        # se inicializa con ceros la matriz que representa la distancia entre un nodo y los datos
        # con los que es ganador
        bmu_distance = zeros((self.columns, self.rows))

        # se inicializa con ceros la matriz que representa la distancia media entre un nodo y
        # todos los datos
        mean_distance = zeros((self.columns, self.rows))

        # para cada dato...
        for ind in data:
            # ... se determina el nodo ganador
            best_matching_unit = self.get_best_matching_unit(ind)

            # se agrega 1 al contador de frecuencia de activación de ese nodo
            activation_frequency[best_matching_unit] += 1

            # se suma la distancia que tuvo con el dato con el que ganó
            bmu_distance[best_matching_unit] += self.activation_map[best_matching_unit]

            # a la matriz de destancias medias, se le agrega el mapa de activación para x
            mean_distance += self.activation_map

        # s calcula la distancia promedio de los nodos con respecto a los datos con los que ganó
        bmu_distance[activation_frequency > 0] /= activation_frequency[activation_frequency > 0]

        # se asigna nan a los nodos que nunca fueron ganadores
        bmu_distance[activation_frequency == 0] = nan

        # si calcula el promedio de la distancia entre el nodo y todos los datos
        mean_distance /= len(data)

        # se retorna las tres matrices
        return activation_frequency, bmu_distance, mean_distance

    def map_attachments(self, data, attachments, aggregation_function=None):

        if aggregation_function is None:
            aggregation_function = _identity

        # se checkean el conjunto ded atos ingresados y los attachments
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

        # se checkea la cantidad de clusters ingresada
        _check_inputs.value_type(n_clusters, int)
        _check_inputs.positive(n_clusters)

        # inicializa la instancia de KMeans
        clusters = KMeans(n_clusters=n_clusters, init="k-means++").fit(
            self.weights.reshape(self.columns * self.rows, self.variables))

        # retorna los clusters y los centros
        return clusters.labels_.reshape(self.columns, self.rows), clusters.cluster_centers_

    def k_medoids(self, n_clusters=4):

        # se checkea la cantidad de clusters ingresada
        _check_inputs.value_type(n_clusters, int)
        _check_inputs.positive(n_clusters)

        if isinstance(self._metric, str):
            metric = self._metric

        else:
            def custom_metric(first_set, second_set):
                return self.distance(array([first_set]), array([second_set]))

            metric = custom_metric

        # inicializa la instancia de KMedoids
        clusters = KMedoids(n_clusters=n_clusters, init="k-medoids++", metric=metric).fit(
            self.weights.reshape(self.columns * self.rows, self.variables))

        return clusters.labels_.reshape(self.columns, self.rows), clusters.cluster_centers_

    def hdbscan(self, min_cluster_size=3, plot_condensed_tree=True):

        _check_inputs.value_type(min_cluster_size, int)
        _check_inputs.positive(min_cluster_size - 1)
        _check_inputs.value_type(plot_condensed_tree, bool)

        if self._hdbscan_cache[min_cluster_size] is not None:
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

                if (
                        self.toroidal
                        and s_bmu[0] == (f_bmu[0] + i + self.columns) % self.columns
                        and s_bmu[1] == (f_bmu[1] + j + self.rows) % self.rows
                ):
                    error = 0

                if (
                        not self.toroidal
                        and self.columns > f_bmu[0] + i >= 0 <= f_bmu[1] + j < self.rows
                        and s_bmu[0] == f_bmu[0] + i
                        and s_bmu[1] == f_bmu[1] + j
                ):
                    error = 0

            topographic_error += error

        return quantization_error / data.shape[0], topographic_error / data.shape[0]

    def get_dict(self, custom_metric=None):
        if custom_metric is None:
            metric = self._metric
        else:
            metric = custom_metric

        return {
            **{
                'z': self.variables,
                'x': self.columns,
                'y': self.rows,
                'metric': metric,
                'hexagonal': self.hexagonal,
                'toroidal': self.toroidal,
                'seed': self.seed,
                'weights': self.weights.tolist(),
                'rp': self.relative_positions.tolist()
            },
            **self._kwargs
        }

    def get_connections_and_reverse(self, cluster, min_cluster_size):

        _check_inputs.value_type(cluster, bool)

        if cluster is False:
            return None, None

        clusters_labels = \
            array(self.hdbscan(min_cluster_size=min_cluster_size, plot_condensed_tree=False))[
                0
            ].reshape(self.columns * self.rows)

        # if restrictive_conf and self.toroidal == False: for i in [[0, 1, self.y], [self.y -
        # 1, 2 * self.y - 2, 2 * self.y - 1], [-self.y, -self.y + 1, -2 * self.y], [-1, -2,
        # -self.y - 1]]: if clusters_labels[i[0]] == -1 and clusters_labels[i[1]] ==
        # clusters_labels[i[2]]: clusters_labels[i[0]] = clusters_labels[i[1]]

        adjacency_matrix = self.get_unified_distance_matrix()[1]

        connections = zeros(adjacency_matrix.shape) * nan
        reverse = zeros(adjacency_matrix.shape)

        reshaped_rp = self.relative_positions.reshape([-1, 2])
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if not isnan(adjacency_matrix[i, j]) and \
                        clusters_labels[i] == clusters_labels[j] and \
                        clusters_labels[i] >= 0 and i != j:
                    connections[i, j] = adjacency_matrix[i, j]
                    if self.toroidal and \
                            norm(reshaped_rp[i] - reshaped_rp[j]) >= min(self.columns,
                                                                         self.rows) / 2:
                        reverse[i, j] = 1
                        reverse[j, i] = 1

        return connections, reverse

    def plot_analysis(self, data, cluster=True, min_cluster_size=3,
                      borders=True, display_empty_nodes=True, size=10):

        analysis = self.analyse(data)
        activation_frequency = analysis[0]
        quantization_error = analysis[1]
        connections, reverse = self.get_connections_and_reverse(cluster, min_cluster_size)

        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(display_empty_nodes, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)

        _plot.bubbles(activation_frequency, self.relative_positions, quantization_error,
                      connections=connections, reverse=reverse,
                      title='RP-HDSCAN   quantization error', borders=borders,
                      display_empty_nodes=display_empty_nodes,
                      size=size)

    def plot_labels(self, data, labels=None, labels_to_display=None, cluster=True,
                    min_cluster_size=3, borders=True, display_empty_nodes=True, size=10):

        activation_frequency = self.analyse(data)[0]
        connections, reverse = self.get_connections_and_reverse(cluster, min_cluster_size)
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
                          connections=connections,
                          norm=False,
                          labels=unique_labels, title='RP-HDBSCAN   labels',
                          color_map=plt.cm.get_cmap('hsv', len(unique_labels) + 1),
                          reverse=reverse, borders=borders,
                          display_empty_nodes=display_empty_nodes,
                          size=size)

        else:
            labels_to_display_indices = where(isin(unique_labels, labels_to_display))[0]

            _check_inputs.positive(len(labels_to_display_indices))

            _plot.bubbles(activation_frequency, self.relative_positions,
                          map_labels[..., labels_to_display_indices].reshape(
                              [activation_frequency.shape[0], activation_frequency.shape[1],
                               len(labels_to_display_indices)]),
                          connections=connections,
                          norm=False, labels=unique_labels[labels_to_display_indices],
                          title='RP-HDBSCAN   labels',
                          color_map=plt.cm.get_cmap('hsv', len(
                              unique_labels[labels_to_display_indices]) + 1),
                          reverse=reverse,
                          borders=borders, display_empty_nodes=display_empty_nodes, size=size)

    def plot_map_value(self, data, attached_values, func=mean, cluster=True, min_cluster_size=3,
                       borders=True, display_empty_nodes=True, size=10):

        activation_frequency = self.analyse(data)[0]
        connections, reverse = self.get_connections_and_reverse(cluster, min_cluster_size)

        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(display_empty_nodes, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)
        nodes_values = self.map_attachments(data, attached_values, func)
        _plot.bubbles(activation_frequency, self.relative_positions, nodes_values,
                      connections=connections,
                      norm=True,
                      title='RP-HDBSCAN   attached values', reverse=reverse, borders=borders,
                      display_empty_nodes=display_empty_nodes, size=size)

    def plot_unified_distance_matrix(self, detailed=True, borders=True, size=10):

        _check_inputs.value_type(detailed, bool)
        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)

        unified_distance_matrix = self.get_unified_distance_matrix()[0]

        if detailed:
            _plot.tiles(self.positions, self.hexagonal, unified_distance_matrix,
                        title='Distance between nodes',
                        borders=borders,
                        size=size)

        else:
            _plot.tiles(self.positions, self.hexagonal, unified_distance_matrix[..., -1],
                        title='Distance between nodes',
                        borders=borders, size=size)

    def plot_weights(self, scaler=None, weights_to_display=None, headers=None, borders=True,
                     size=10):

        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)

        if scaler is None:
            weights = self.weights.copy()
        else:
            weights = scaler.inverse_transform(
                self.weights.reshape(-1, self.weights.shape[2])).reshape(
                self.weights.shape)

        if weights_to_display is None:
            weights_to_display = list(range(weights.shape[2]))

        if headers is None:
            headers = weights_to_display

        _check_inputs.length(weights_to_display, headers)

        for weight in weights_to_display:
            _plot.tiles(self.positions, self.hexagonal, weights[..., weight],
                        title=headers[weight], borders=borders,
                        size=size)

    def plot_weights_vector(self, node_index=(0, 0), xticks_labels=None, lateral_bar=True,
                            scatter=False, line=False):

        _check_inputs.value_type(node_index, tuple)

        _check_inputs.value_type(lateral_bar, bool)
        _check_inputs.value_type(scatter, bool)
        _check_inputs.value_type(line, bool)

        node = self.weights[node_index]
        plt.figure(figsize=(20, 5))
        if lateral_bar:
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
                                     display_median=False,
                                     display_mean=True,
                                     display_lines=True):

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
            plt.scatter(list(range(self.variables)), quantile(values, 0.5, axis=0), color='black',
                        s=50)
            plt.scatter(list(range(self.variables)), quantile(values, 0.75, axis=0), color='black',
                        s=50)
            plt.scatter(list(range(self.variables)), quantile(values, 0.25, axis=0), color='black',
                        s=50)
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

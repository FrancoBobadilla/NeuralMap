from matplotlib import pyplot as plt

from collections import Counter

from sklearn.cluster import KMeans

from numpy import arange, random, zeros, array, unravel_index, isnan, meshgrid, transpose, ogrid, cov, argsort, \
    linspace, fill_diagonal, nan, nan_to_num, mean, argmin, where, isin, unique, quantile

from numpy.linalg import norm, eig

from scipy.spatial.distance import cdist

from hdbscan import HDBSCAN

from sklearn_extra.cluster import KMedoids

from . import _plot, _check_inputs, _decay_functions, _neighbourhood_functions


def _identity(x):
    return x


class NeuralMap:
    def __init__(self,
                 z,
                 x=20,
                 y=20,
                 hexagonal=True,
                 toroidal=False,
                 rp=None,
                 weights=None,
                 seed=None,
                 metric='euclidean',
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
        _check_inputs.value_type(z, int)
        _check_inputs.positive(z)
        _check_inputs.value_type(x, int)
        _check_inputs.positive(x)
        _check_inputs.value_type(y, int)
        _check_inputs.positive(y)
        _check_inputs.value_type(hexagonal, bool)
        _check_inputs.value_type(toroidal, bool)
        _check_inputs.value_type(seed, int)

        self._metric = metric
        self._kwargs = kwargs

        if isinstance(metric, str):
            self._distance = lambda first_array, second_array: \
                cdist(first_array, second_array, self._metric, **self._kwargs)
        else:
            self._distance = lambda first_array, second_array: \
                self._metric(first_array, second_array, **self._kwargs)

        # check if input metric is valid
        self._distance(array([[0., 1.]]), array([[1., 2.], [3., 4.]]))

        # se configura la cantidad horizontal de nodos
        self._x = x

        # se configura la cantidad vertical de nodos
        self._y = y

        # se configura la cantidad de elementos de los vectores de pesos de los nodos
        self._z = z

        if weights is not None:
            weights = array(weights).astype(float)
            _check_inputs.ndarray_and_shape(weights, (self._x, self._y, self._z))

        if rp is not None:
            rp = array(rp).astype(float)
            _check_inputs.ndarray_and_shape(rp, (self._x, self._y, self._z))

        # configura si la topología del mapa es hexagonal (o cuadrada)
        self._hexagonal = hexagonal

        # configura si la topología del mapa es toroidal (o plana)
        self._toroidal = toroidal

        # configura la semilla aleatoria
        self._seed = seed

        # configura el ancho del mapa
        self._width = self._x

        # configura el alto del mapa
        self._height = self._y

        # se configura el estado aleatorio en base a la semilla calculada en el constructor
        random.seed(self._seed)

        if weights is None:
            # inicia los pesos en 0
            self._weights = zeros((self._x, self._y, self._z))
        else:
            self._weights = weights

        # crea el mapa de activación, con valores iniciales de 0
        self._activation_map = zeros((self._x, self._y))

        self._adjacent_nodes_relative_positions = [[(1, 0), (0, 1), (-1, 0), (0, -1)],
                                                   [(1, 0), (0, 1), (-1, 0), (0, -1)]]

        self._positions = transpose(meshgrid(arange(self._x), arange(self._y)), axes=[2, 1, 0]).astype(float)

        if self._hexagonal:
            self._positions[:, 0::2, 0] += 0.5
            self._positions[..., 1] *= (3 ** 0.5) * 0.5
            self._height *= (3 ** 0.5) * 0.5
            self._adjacent_nodes_relative_positions = [[(1, 0), (1, 1), (0, 1), (-1, 0), (0, -1), (1, -1)],
                                                       [(1, 0), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1)]]

        # xx, yy = array(meshgrid(arange(self._x), arange(self._y)), dtype=float)
        # if self._hexagonal:
        #     xx[::2] += 0.5
        #     yy *= (3 ** 0.5) * 0.5
        #     self._height *= (3 ** 0.5) * 0.5
        # self._positions = transpose(array([xx.T, yy.T]), axes=[1, 2, 0])

        if rp is None:
            # se inicializan las pocisiones relativas de los nodos en la posición "fija" que cada uno tiene
            self._rp = self._positions.copy()
        else:
            self._rp = rp

        self._current_epoch = None

        self._unified_distance_matrix_cache = None

        self._hdbscan_cache = [(None, None, None)] * self._x * self._y

    def pca_weights_init(self, data):
        principal_components_length, principal_components = eig(cov(transpose(data)))
        principal_component_order = argsort(-principal_components_length)
        for i, component_1 in enumerate(linspace(-1, 1, self._x) * principal_components[principal_component_order[0]]):
            for j, component_2 in enumerate(linspace(-1, 1, self._y) * principal_components[principal_component_order[1]]):
                self._weights[i, j] = component_1 + component_2

    def uniform_weights_init(self):
        self._weights = random.rand(self._x, self._y, self._z)

    def standard_weights_init(self):
        self._weights = random.normal(0., 1., (self._x, self._y, self._z))

    def pick_from_data_weights_init(self, data):
        indices = arange(data.shape[0])
        for i in range(self._x):
            for j in range(self._y):
                self._weights[i, j] = data[random.choice(indices)]

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
            initial_radius = min(self._width, self._height) / 2

        # si se ingresó solo una época de entrenamiento...
        if n_epochs == 1:
            # ... no se utiliza función de disminución para learning rate ni radius
            learning_rate_decay_function = _decay_functions.no_decay
            radius_decay_function = _decay_functions.no_decay

        self._unified_distance_matrix_cache = None
        self._hdbscan_cache = [None] * self._x * self._y

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
        _check_inputs.numpy_matrix(data, self._z)
        if eval_data is not None:
            _check_inputs.numpy_matrix(eval_data, self._z)

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

        # se declara la lista que se va a utilizar para almacenar la distancia promedio de los bmu por cada época
        epochs_quantization_error = zeros(n_epochs)
        epochs_topographic_error = zeros(n_epochs)

        # se genera la lista de índices de los datos de entrada sobre el que se va a iterar
        indices = arange(len(data))

        # se configura el estado aleatorio en base a la semilla calculada en el constructor
        random.seed(self._seed)

        #############################

        # se declara el arreglo que va a contener las dimensiones del mapa
        dimensions = array([self._width, self._height])

        # se declara la posición del centro del mapa
        center = (self._x // 2, self._y // 2)

        # se declara la lista sobre la que se va a almacenar el desplazamiento horizontal de cada itereación
        y_displacement = zeros(self._y, dtype='int')

        # se declara la lista sobre la que se va a almacenar el desplazamiento vertical de cada itereación
        x_displacement = zeros(self._x, dtype='int')

        # se almcenan en una variable los índices de la mariz g
        updata_matrix_indices = ogrid[[slice(0, self._x), slice(0, self._y)]]

        # se inicializa con ceros la lista de correcciones en las posiciones de los nodos
        correction = zeros(self._y, dtype='int')

        # se calcula la corrección que debe tener cada fila cuando el bmu está defasado del centro
        correction[center[1] % 2:: 2] = (center[1] % 2) * 2 - 1

        #############################

        plot_update = False

        # por cada época...
        for epoch in range(n_epochs):

            # ... se mezclan nuevamente las iteraciones
            random.shuffle(indices)

            # se genera un nuevo estado aleatorio
            random.seed(epoch)

            # se calcula la Learning Rate de la época
            learning_rate = learning_rate_decay_function(initial_learning_rate, final_learning_rate, n_epochs,
                                                         epoch)

            # se calcula el Radius de la época
            radius = radius_decay_function(initial_radius, final_radius, n_epochs, epoch)

            self._current_epoch = epoch

            if verbose:
                # se imprime el progreso
                print('\nEpoch: ', epoch + 1, ' of ', n_epochs,
                      '    Learning rate: ', learning_rate,
                      '    Radius: ', radius)

            pr = plot_update

            # plt.scatter(self._rp[..., 0], self._rp[..., 1])
            # plt.show()
            # plt.scatter((self._rp[..., 0] + self._width / 3) % self._width,
            #             (self._rp[..., 1] + self._height / 3) % self._height)
            # plt.show()

            update_matrix_over_center = neighbourhood_function(self._positions, self._positions[center], radius, learning_rate)

            # para cada iteración...
            for i in indices:
                # ... se toma el dato correspondiente a la iteración
                d = data[i]

                # se calcula el nodo ganador, y su distancia con respecto al dato
                best_matching_unit = self.get_best_matching_unit(d)

                if self._toroidal:

                    # se calcula la cantidad de nodos horizontal que hay desde el centro del mapa al nodo ganador
                    u = best_matching_unit[0] - center[0]

                    # se llena con el valur u la lista de desplazamientos horizontales
                    y_displacement.fill(u)

                    # se calcula la distancia vertical que hay desde el centro del mapa al nodo ganador
                    v = best_matching_unit[1] - center[1]

                    # se llena con el valur v la lista de desplazamientos verticales
                    x_displacement.fill(v)

                    '''La matriz de actualizaciones (g) calculada sobre el centro del mapa debe ser desplazada para 
                    que coincida con la bmu. De esta manera se logra emular un espacio toroidal. Para eso se 
                    reindexan los elementos del arreglo, restando de los índices originales (all_idcs) el 
                    desplazamiento horizontal y vertical correspondiente (hor_disp, ver_disp). Además, 
                    cuando se trata de topología hexagonal y el bmu está en una fila "defasada" del centro del mapa, 
                    se debe hacer una corrección adicional, que es calculada al principio del entrenamiento. 
                    Finalmente se calcula el resto con respecto a la cantidad de nodos horizontal y verticalmente, 
                    para asegurar que los nuevos índices se encuentren dentro de los límites.'''

                    # se reindexa la matriz g para que coincida con el nodo ganador
                    update_matrix = update_matrix_over_center[tuple([
                        (updata_matrix_indices[0] - (y_displacement + (self._hexagonal and v % 2) * correction)) % self._x,
                        (updata_matrix_indices[1] - x_displacement[:, None]) % self._y
                    ])]

                    # se calcula la ubicación opuesta al nodo ganador
                    anti_bmu = (self._positions[best_matching_unit] + dimensions / 2) % dimensions

                    # se calcula el cuadrante en el que está el nodo ganador con respecto a su ubicación opuesta
                    cuad = array([u > 0 or anti_bmu[0] < 1, v > 0 or anti_bmu[1] < 1]) * 2 - 1

                    # se calcula la matriz donde se indica la ubicación del nodo ganador al que debe tender cada RP
                    mod = (self._rp * cuad < anti_bmu * cuad) * dimensions * cuad

                    # se calcula el desplazamiento que debe aplicarse a cada RP
                    # despl = einsum('ij, ijk->ijk', g, (self._cart_coord[winner_node] - mod - self._rp))
                    displacement = (self._positions[best_matching_unit] - mod - self._rp) * update_matrix[..., None]

                    if pr:
                        pr = False
                        _plot.update(self._positions, self._hexagonal, update_matrix, dimensions, self._positions[best_matching_unit], self._rp,
                                     displacement)
                        plt.show()
                        # print((self._width, self._height, winner_node, self._cart_coord, center, g, self._rp, '\n\n'))

                    # se actualiza la matriz de RP
                    self._rp += displacement

                    # se corrige para que entre en las dimensiones del mapa
                    self._rp %= dimensions

                else:

                    # se calcula la matriz de actualizaciones
                    update_matrix = neighbourhood_function(self._positions, self._positions[best_matching_unit], radius,
                                               learning_rate)

                    # se calcula el desplazamiento que debe aplicarse a cada RP
                    # despl = einsum('ij, ijk->ijk', g, winner_node - self._rp)

                    displacement = (self._positions[best_matching_unit] - self._rp) * update_matrix[..., None]
                    # despl = (self._rp[winner_node] - self._rp) * g[..., None]
                    # despl[winner_node] = (self._cart_coord[winner_node] - self._rp[winner_node]) * g[winner_node]

                    if pr:
                        pr = False
                        _plot.update(self._positions, self._hexagonal, update_matrix, dimensions, self._positions[best_matching_unit],
                                     self._rp, displacement)
                        plt.show()

                    # se actualiza la matriz de RP
                    self._rp += displacement

                # se actualizan los pesos de la red
                self._weights += (d - self._weights) * update_matrix[..., None]

            if eval_data is not None:
                epochs_quantization_error[epoch], epochs_topographic_error[epoch] = self.evaluate(eval_data)

        if eval_data is not None:
            # se muestra el gráfico de la distancia promedio de los nodos ganadores por cada época
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

    # def _distance(self, x, y):
    #     if self._string_metric:
    #         return cdist(x, y, self._metric, **self._kwargs)
    #     else:
    #         return self._metric(x, y, **self._kwargs)

    def generate_activation_map(self, x):

        # se calcula el mapa de activación con el dato ingresado
        self._activation_map = self._distance(x.reshape([1, -1]), self._weights.reshape(
            [self._weights.shape[0] * self._weights.shape[1], self._weights.shape[2]])).reshape(
            [self._weights.shape[0], self._weights.shape[1]])

        # devuelve el mapa de activación
        return self._activation_map

    def get_best_matching_unit(self, x):

        # se devuelve la posición bidimensional del nodo ganador
        return unravel_index(self.generate_activation_map(x).argmin(), self._activation_map.shape)

    def get_unified_distance_matrix(self):

        if self._unified_distance_matrix_cache is not None:
            return self._unified_distance_matrix_cache

        adjacency_matrix = zeros((self._x * self._y, self._x * self._y)) * nan
        fill_diagonal(adjacency_matrix, 0.)

        # se calcula la cantidad de nodos adyacentes que tiene cada nodo
        adjacency_length = len(self._adjacent_nodes_relative_positions[0])

        # se inicializa la matriz de distancias unificadas con ceros
        unified_distance_matrix = zeros((self._x, self._y, 1 + adjacency_length))

        # para cada posición en x del mapa...
        for x in range(self._x):

            # ... para cada posición en y del mapa...
            for y in range(self._y):

                # ... se configura en 0 el contador de la cantidad de nodos adyacentes válidos
                c = 0

                # para cada tupla de coordenadas relativas al nodo...
                for k, (i, j) in enumerate(self._adjacent_nodes_relative_positions[y % 2]):

                    if self._toroidal:
                        neighbour = array([[self._weights[(x + i + self._x) % self._x, (y + j + self._y) % self._y]]])

                    elif self._x > x + i >= 0 <= y + j < self._y:
                        neighbour = array([[self._weights[x + i, y + j]]])

                    else:
                        neighbour = None

                    if neighbour is not None:

                        # se calcula la distancia entre el nodo y su vecino
                        distance = self._distance(self._weights[x, y].reshape([1, -1]),
                                                  neighbour[0, 0].reshape([1, -1]))

                        # se lo suma la distancia al valor que tiene la matriz para esa posición
                        unified_distance_matrix[x, y, k] = distance

                        # se asigna la distancia calcular al valor que tiene la matriz para esa posición
                        unified_distance_matrix[x, y, adjacency_length] += distance

                        # se incrementa en 1 la cuenta de nodos adyacentes, para calcular el promedio
                        c += 1

                        adjacency_matrix[x * self._y + y, ((x + i + self._x) % self._x) * self._y + (
                                y + j + self._y) % self._y] = distance

                    else:
                        # ... se asigna nan a esa posición de la matriz
                        unified_distance_matrix[x, y, k] = nan

                # si el nodo tiene al menos un nodo adyacente...
                if c > 0:

                    # ... se aclcula el promedio de las distancias
                    unified_distance_matrix[x, y, adjacency_length] = unified_distance_matrix[x, y, adjacency_length] / c

                # si no tiene ningún vecino...
                else:

                    # ... se asigna nan
                    unified_distance_matrix[x, y, adjacency_length] = nan

        self._unified_distance_matrix_cache = unified_distance_matrix, adjacency_matrix
        return self._unified_distance_matrix_cache

    def analyse(self, data):

        # se checkea el conjunto de datos ingresado
        _check_inputs.numpy_matrix(data, self._z)

        # se inicializa con ceros la matriz que representa la frecuencia de activación
        activation_frequency = zeros((self._x, self._y))

        # se inicializa con ceros la matriz que representa la distancia entre un nodo y los datos con los que es ganador
        bmu_distance = zeros((self._x, self._y))

        # se inicializa con ceros la matriz que representa la distancia media entre un nodo y todos los datos
        mean_distance = zeros((self._x, self._y))

        # para cada dato...
        for x in data:
            # ... se determina el nodo ganador
            best_matching_unit = self.get_best_matching_unit(x)

            # se agrega 1 al contador de frecuencia de activación de ese nodo
            activation_frequency[best_matching_unit] += 1

            # se suma la distancia que tuvo con el dato con el que ganó
            bmu_distance[best_matching_unit] += self._activation_map[best_matching_unit]

            # a la matriz de destancias medias, se le agrega el mapa de activación para x
            mean_distance += self._activation_map

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
        _check_inputs.numpy_matrix(data, self._z)
        _check_inputs.length(data, attachments)
        _check_inputs.function(aggregation_function)

        result = zeros((self._x, self._y)).tolist()
        # result = [[0.0] * self._y] * self._x

        dict_map = {}

        for i in range(self._x):
            for j in range(self._y):
                dict_map[(i, j)] = []

        for d, a in zip(data, attachments):
            dict_map[tuple(self.get_best_matching_unit(d))].append(a)

        for k, item in dict_map.items():
            result[k[0]][k[1]] = aggregation_function(item)

        return array(result)

    def k_means(self, n_clusters=4):

        # se checkea la cantidad de clusters ingresada
        _check_inputs.value_type(n_clusters, int)
        _check_inputs.positive(n_clusters)

        # inicializa la instancia de KMeans
        clusters = KMeans(n_clusters=n_clusters, init="k-means++").fit(
            self._weights.reshape(self._x * self._y, self._z))

        # retorna los clusters y los centros
        return clusters.labels_.reshape(self._x, self._y), clusters.cluster_centers_

    def _custom_metric(self, x, y):
        return self._distance(array([x]), array([y]))

    def k_medoids(self, n_clusters=4):

        # se checkea la cantidad de clusters ingresada
        _check_inputs.value_type(n_clusters, int)
        _check_inputs.positive(n_clusters)

        if isinstance(self._metric, str):
            metric = self._metric

        else:
            metric = self._custom_metric

        # inicializa la instancia de KMedoids
        clusters = KMedoids(n_clusters=n_clusters, init="k-medoids++", metric=metric).fit(
            self._weights.reshape(self._x * self._y, self._z))

        # retorna los clusters y los centros
        return clusters.labels_.reshape(self._x, self._y), clusters.cluster_centers_

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

        # if restrictive_conf and self._toroidal == False: for i in [[0, 1, self._y], [self._y - 1, 2 * self._y - 2,
        # 2 * self._y - 1], [-self._y, -self._y + 1, -2 * self._y], [-1, -2, -self._y - 1]]: if labels[i[0]] == -1
        # and labels[i[1]] == labels[i[2]]: labels[i[0]] = labels[i[1]] probabilities[i[0]] = (probabilities[i[1]] +
        # probabilities[i[2]]) / 2 outlier_scores[i[0]] = (outlier_scores[i[1]] + outlier_scores[i[2]]) / 2

        if plot_condensed_tree:
            clusters.condensed_tree_.plot(select_clusters=True, label_clusters=True)

        self._hdbscan_cache[min_cluster_size] = (
            labels.reshape(self._x, self._y),
            probabilities.reshape(self._x, self._y),
            outlier_scores.reshape(self._x, self._y)
        )

        return self._hdbscan_cache[min_cluster_size]

    def evaluate(self, data):

        # se checkea el conjunto de datos ingresado
        _check_inputs.numpy_matrix(data, self._z)

        topographic_error = 0
        quantization_error = 0

        # para cada dato...
        for index, d in enumerate(data):

            activation_map = self.generate_activation_map(d)

            f_bmu = unravel_index(argmin(activation_map), activation_map.shape)
            quantization_error += activation_map[f_bmu]

            activation_map[f_bmu] = activation_map.max()
            s_bmu = unravel_index(argmin(activation_map), activation_map.shape)
            x = f_bmu[0]
            y = f_bmu[1]
            error = 1

            # para cada tupla de coordenadas relativas al nodo...
            for k, (i, j) in enumerate(self._adjacent_nodes_relative_positions[y % 2]):

                if (
                        self._toroidal
                        and s_bmu[0] == (x + i + self._x) % self._x
                        and s_bmu[1] == (y + j + self._y) % self._y
                ) or (
                        not self._toroidal
                        and self._x > x + i >= 0 <= y + j < self._y
                        and s_bmu[0] == x + i
                        and s_bmu[1] == y + j
                ):
                    error = 0

            topographic_error += error

            # if error:
            #     activation_map[f_bmu] = nan
            #     activation_map[s_bmu] = nan
            #     print('\n\n', index)
            #     _plot.tiles(cart_coord=self._cart_coord, hexagonal=self._hexagonal, data=activation_map, size=5)
            #     plt.show()

        # retorna la matriz de distancia unificada
        return quantization_error / data.shape[0], topographic_error / data.shape[0]

    def get_dict(self, custom_metric=None):
        if custom_metric is None:
            metric = self._metric
        else:
            metric = custom_metric

        return {
            **{
                'z': self._z,
                'x': self._x,
                'y': self._y,
                'metric': metric,
                'hexagonal': self._hexagonal,
                'toroidal': self._toroidal,
                'seed': self._seed,
                'weights': self._weights.tolist(),
                'rp': self._rp.tolist()
            },
            **self._kwargs
        }

    def plot_analysis(self, data, labels=None, labels_to_display=None, attached_values=None, func=mean,
                      display_quantization_error=True, cluster=True, min_cluster_size=3, borders=True,
                      display_empty_nodes=True, size=10):

        _check_inputs.value_type(display_quantization_error, bool)
        _check_inputs.value_type(cluster, bool)
        _check_inputs.value_type(min_cluster_size, int)
        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(display_empty_nodes, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)

        activation_frequency, quantization_error, mean_distance = self.analyse(data)
        adjacency_matrix = self.get_unified_distance_matrix()[1]

        if cluster:
            clusters_labels = array(self.hdbscan(min_cluster_size=min_cluster_size, plot_condensed_tree=False))[
                0
            ].reshape(self._x * self._y)

            # if restrictive_conf and self.toroidal == False: for i in [[0, 1, self.y], [self.y - 1, 2 * self.y - 2,
            # 2 * self.y - 1], [-self.y, -self.y + 1, -2 * self.y], [-1, -2, -self.y - 1]]: if clusters_labels[i[0]]
            # == -1 and clusters_labels[i[1]] == clusters_labels[i[2]]: clusters_labels[i[0]] = clusters_labels[i[1]]

            connections = zeros(adjacency_matrix.shape) * nan
            reverse = zeros(adjacency_matrix.shape)

            for i in range(adjacency_matrix.shape[0]):
                for j in range(adjacency_matrix.shape[1]):
                    if not isnan(adjacency_matrix[i, j]) and clusters_labels[i] == clusters_labels[j] and \
                            clusters_labels[i] >= 0 and i != j:
                        connections[i, j] = adjacency_matrix[i, j]
                        if self.toroidal and norm(
                                self.rp.reshape([-1, 2])[i] - self.rp.reshape([-1, 2])[j]) >= min(self.x, self.y) / 2:
                            reverse[i, j] = 1
                            reverse[j, i] = 1
        else:
            connections = None
            reverse = None

        if display_quantization_error:
            _plot.bubbles(activation_frequency, self.rp, quantization_error, connections=connections, reverse=reverse,
                          title='RP-HDSCAN   quantization error', borders=borders, display_empty_nodes=display_empty_nodes,
                          size=size)

        if labels is not None:
            unique_labels = unique(labels)

            def aggregation_function(item):
                res = zeros(unique_labels.shape[0])
                c = Counter(item)
                for k, Type in enumerate(unique_labels):
                    res[k] = c[Type]
                return res

            map_labels = self.map_attachments(data, labels, aggregation_function)

            if labels_to_display is None:
                _plot.bubbles(activation_frequency, self.rp, map_labels, connections=connections, norm=False,
                              labels=unique_labels, title='RP-HDBSCAN   labels',
                              color_map=plt.cm.get_cmap('hsv', len(unique_labels) + 1),
                              reverse=reverse, borders=borders, display_empty_nodes=display_empty_nodes,
                              size=size)

            else:
                labels_to_display_indices = where(isin(unique_labels, labels_to_display))[0]

                _check_inputs.positive(len(labels_to_display_indices))

                _plot.bubbles(activation_frequency, self.rp, map_labels[..., labels_to_display_indices].reshape(
                    [activation_frequency.shape[0], activation_frequency.shape[1], len(labels_to_display_indices)]), connections=connections,
                              norm=False, labels=unique_labels[labels_to_display_indices], title='RP-HDBSCAN   labels',
                              color_map=plt.cm.get_cmap('hsv', len(unique_labels[labels_to_display_indices]) + 1),
                              reverse=reverse,
                              borders=borders, display_empty_nodes=display_empty_nodes, size=size)

        if attached_values is not None:
            nodes_values = self.map_attachments(data, attached_values, func)
            _plot.bubbles(activation_frequency, self.rp, nodes_values, connections=connections, norm=True,
                          title='RP-HDBSCAN   attached values', reverse=reverse, borders=borders,
                          display_empty_nodes=display_empty_nodes, size=size)

    def plot_unified_distance_matrix(self, detailed=True, borders=True, size=10):

        _check_inputs.value_type(detailed, bool)
        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)

        unified_distance_matrix, adjacency_matrix = self.get_unified_distance_matrix()

        if detailed:
            _plot.tiles(self._positions, self._hexagonal, unified_distance_matrix, title='Distance between nodes', borders=borders,
                        size=size)

        else:
            _plot.tiles(self._positions, self._hexagonal, unified_distance_matrix[..., -1], title='Distance between nodes',
                        borders=borders, size=size)

    def plot_weights(self, scaler=None, weights_to_display=None, headers=None, borders=True, size=10):

        _check_inputs.value_type(borders, bool)
        _check_inputs.value_type(size, int)
        _check_inputs.positive(size)

        if scaler is None:
            weights = self._weights.copy()
        else:
            weights = scaler.inverse_transform(self._weights.reshape(-1, self._weights.shape[2])).reshape(
                self._weights.shape)

        if weights_to_display is None:
            weights_to_display = list(range(weights.shape[2]))

        if headers is None:
            headers = weights_to_display

        _check_inputs.length(weights_to_display, headers)

        for weight in weights_to_display:
            _plot.tiles(self._positions, self._hexagonal, weights[..., weight], title=headers[weight], borders=borders,
                        size=size)

    def plot_weights_vector(self, node_index=(0, 0), xticks_labels=None, bar=True, scatter=False, line=False):

        _check_inputs.value_type(node_index, tuple)

        _check_inputs.value_type(bar, bool)
        _check_inputs.value_type(scatter, bool)
        _check_inputs.value_type(line, bool)

        node = self._weights[node_index]
        plt.figure(figsize=(20, 5))
        if bar:
            plt.bar(list(range(node.shape[0])), node)

        if scatter:
            plt.scatter(list(range(node.shape[0])), node)

        if line:
            plt.plot(list(range(node.shape[0])), node)

        plt.title("Node " + str(node_index), " weights vector")

        if xticks_labels is None:
            xticks_labels = list(range(node.shape[0]))

        plt.xticks(ticks=list(range(node.shape[0])), labels=xticks_labels, rotation='vertical')

    def plot_cluster_weights_vectors(self, cluster=None, xticks_labels=None, min_cluster_size=3, display_median=False, display_mean=True,
                                     display_lines=True):

        _check_inputs.value_type(display_median, bool)
        _check_inputs.value_type(display_mean, bool)
        _check_inputs.value_type(display_lines, bool)

        labels = self.hdbscan(plot_condensed_tree=False, min_cluster_size=min_cluster_size)[0]

        values = self._weights.reshape((-1, self._z))
        cluster_title = "Weights vectors"
        alpha = 1.

        if cluster is not None:
            _check_inputs.value_type(cluster, int)
            values = values[(labels == cluster).flatten()]
            cluster_title += ". Cluster " + str(cluster)

        if display_mean:
            alpha = 0.5
            cluster_title += '. Mean, M + 2 std y M - 2 std'

        if display_median:
            alpha = 0.5
            cluster_title += '. Median, Q1 y Q3'

        plt.figure(figsize=(20, 5))

        for value in values:
            plt.scatter(list(range(self._z)), value, alpha=alpha)
            if display_lines:
                plt.plot(value, alpha=alpha / 2, zorder=-1)

        if display_median:
            v_q1 = quantile(values, 0.25, axis=0)
            v_q2 = quantile(values, 0.5, axis=0)
            v_q3 = quantile(values, 0.75, axis=0)
            plt.scatter(list(range(self._z)), v_q2, color='black', s=50)
            plt.scatter(list(range(self._z)), v_q3, color='black', s=50)
            plt.scatter(list(range(self._z)), v_q1, color='black', s=50)
            if display_lines:
                plt.plot(v_q2, color='black', lw=2, zorder=-1)
                plt.plot(v_q3, color='black', lw=1, ls='--', zorder=-1)
                plt.plot(v_q1, color='black', lw=1, ls='--', zorder=-1)

        if display_mean:
            v_mean = values.mean(axis=0)
            v_std = values.std(axis=0)
            plt.scatter(list(range(self._z)), v_mean, color='black', s=50)
            plt.scatter(list(range(self._z)), v_mean + 2 * v_std, color='black', s=50)
            plt.scatter(list(range(self._z)), v_mean - 2 * v_std, color='black', s=50)
            if display_lines:
                plt.plot(v_mean, color='black', lw=2)
                plt.plot(v_mean + 2 * v_std, color='black', lw=1, ls='--')
                plt.plot(v_mean - 2 * v_std, color='black', lw=1, ls='--')

        plt.title(cluster_title)

        if xticks_labels is None:
            xticks_labels = list(range(self._z))

        plt.xticks(ticks=list(range(self._z)), labels=xticks_labels, rotation='vertical')

        # plt.figure(figsize=(20, 5))
        # plt.boxplot(values)
        # plt.xticks(ticks=list(range(1, 1 + self._z)), labels=df.drop(columns=l_columns).columns, rotation='vertical')

    '''
    getters y setters
    '''

    def get_current_epoch(self):
        return self._current_epoch

    def get_x(self):
        return self._x

    def get_toroidal(self):
        return self._toroidal

    def get_y(self):
        return self._y

    def get_z(self):
        return self._z

    def get_hexagonal(self):
        return self._hexagonal

    def get_seed(self):
        return self._seed

    def get_metric(self):
        return self._metric

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_weights(self):
        return self._weights

    def get_cart_coord(self):
        return self._positions

    def get_rp(self):
        return self._rp

    def get_activation_map(self):
        return self._activation_map

    @property
    def x(self):
        return self._x

    @property
    def toroidal(self):
        return self._toroidal

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def hexagonal(self):
        return self._hexagonal

    @property
    def seed(self):
        return self._seed

    @property
    def metric(self):
        return self._metric

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def rp(self):
        return self._rp

    @property
    def positions(self):
        return self._positions

    @property
    def activation_map(self):
        return self._activation_map

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        if _check_inputs.ndarray_and_shape(weights, self._weights.shape):
            self._unified_distance_matrix_cache = None
            self._hdbscan_cache = [None] * self._x * self._y
            self._weights = weights

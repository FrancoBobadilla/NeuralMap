from numba import jit, float64
from numpy import empty, exp


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def bubble(positions, bmu, radius, learning_rate):
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            if ((positions[i, j, 0] - bmu[0]) ** 2 + (
                    positions[i, j, 1] - bmu[1]) ** 2) / radius ** 2 < 1:
                res[i, j] = learning_rate
            else:
                res[i, j] = 0.
    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def conical(positions, bmu, radius, learning_rate):
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            res[i, j] = max(0., 1. - (((positions[i, j, 0] - bmu[0]) ** 2 + (
                        positions[i, j, 1] - bmu[1]) ** 2) ** 0.5) / radius) * learning_rate
    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def gaussian(positions, bmu, radius, learning_rate):
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            res[i, j] = exp(-((positions[i, j, 0] - bmu[0]) ** 2 +
                              (positions[i, j, 1] - bmu[1]) ** 2) / radius ** 2) * learning_rate
    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def gaussian_cut(positions, bmu, radius, learning_rate):
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            val = ((positions[i, j, 0] - bmu[0]) ** 2 + (positions[i, j, 1] - bmu[1]) ** 2) / \
                  radius ** 2
            if val >= 1.:
                res[i, j] = 0.
            else:
                res[i, j] = exp(-val) * learning_rate
    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def mexican_hat(positions, bmu, radius, learning_rate):
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            value = -((positions[i, j, 0] - bmu[0]) ** 2 + (positions[i, j, 1] - bmu[1]) ** 2) / \
                    radius ** 2
            res[i, j] = exp(value) * (1. + value) * learning_rate
    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def no_neighbourhood(positions, bmu, radius, learning_rate):
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            if positions[i, j, 0] == bmu[0] and positions[i, j, 1] == bmu[1]:
                res[i, j] = learning_rate
            else:
                res[i, j] = 0.
    return res

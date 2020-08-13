from numba import jit, float64
from numpy import empty, exp

neighbourhood_functions = ['bubble', 'conical', 'gaussian', 'gaussian_cut', 'mexican_hat', 'no_neighborhood']


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def bubble(cart_coord, bmu, radius, learning_rate):
    res = empty((cart_coord.shape[0], cart_coord.shape[1]), dtype=cart_coord.dtype)
    for i in range(cart_coord.shape[0]):
        for j in range(cart_coord.shape[1]):
            if ((cart_coord[i, j, 0] - bmu[0]) ** 2 + (cart_coord[i, j, 1] - bmu[1]) ** 2) / radius ** 2 < 1:
                res[i, j] = learning_rate
            else:
                res[i, j] = 0.
    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def conical(cart_coord, bmu, radius, learning_rate):
    res = empty((cart_coord.shape[0], cart_coord.shape[1]), dtype=cart_coord.dtype)
    for i in range(cart_coord.shape[0]):
        for j in range(cart_coord.shape[1]):
            res[i, j] = max(0., 1. - (((cart_coord[i, j, 0] - bmu[0]) ** 2 + (
                    cart_coord[i, j, 1] - bmu[1]) ** 2) ** 0.5) / radius) * learning_rate
    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def gaussian(cart_coord, bmu, radius, learning_rate):
    res = empty((cart_coord.shape[0], cart_coord.shape[1]), dtype=cart_coord.dtype)
    for i in range(cart_coord.shape[0]):
        for j in range(cart_coord.shape[1]):
            res[i, j] = exp(-((cart_coord[i, j, 0] - bmu[0]) ** 2 + (
                    cart_coord[i, j, 1] - bmu[1]) ** 2) / radius ** 2) * learning_rate
    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def gaussian_cut(cart_coord, bmu, radius, learning_rate):
    res = empty((cart_coord.shape[0], cart_coord.shape[1]), dtype=cart_coord.dtype)
    for i in range(cart_coord.shape[0]):
        for j in range(cart_coord.shape[1]):
            val = ((cart_coord[i, j, 0] - bmu[0]) ** 2 + (cart_coord[i, j, 1] - bmu[1]) ** 2) / radius ** 2
            if val > 1.:
                res[i, j] = 0.
            else:
                res[i, j] = exp(-val) * learning_rate
    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def mexican_hat(cart_coord, bmu, radius, learning_rate):
    res = empty((cart_coord.shape[0], cart_coord.shape[1]), dtype=cart_coord.dtype)
    for i in range(cart_coord.shape[0]):
        for j in range(cart_coord.shape[1]):
            p = -((cart_coord[i, j, 0] - bmu[0]) ** 2 + (cart_coord[i, j, 1] - bmu[1]) ** 2) / radius ** 2
            res[i, j] = exp(p) * (1. + p) * learning_rate
    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def no_neighborhood(cart_coord, bmu, radius, learning_rate):
    res = empty((cart_coord.shape[0], cart_coord.shape[1]), dtype=cart_coord.dtype)
    for i in range(cart_coord.shape[0]):
        for j in range(cart_coord.shape[1]):
            if cart_coord[i, j, 0] == bmu[0] and cart_coord[i, j, 1] == bmu[1]:
                res[i, j] = learning_rate
            else:
                res[i, j] = 0.
    return res

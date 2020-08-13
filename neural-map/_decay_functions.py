from numba import jit, float64

decay_functions = ['linear', 'exponential', 'rational', 'no_decay']


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def linear(init, final, epochs, t):
    return init + t * (final - init) / (epochs - 1)


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def exponential(init, final, epochs, t):
    return init * (final / init) ** (t / (epochs - 1))


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def rational(init, final, epochs, t):
    b = (epochs - 1) / ((init / final) - 1)
    return init * b / (b + t)


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def no_decay(init, final, epochs, t):
    return init

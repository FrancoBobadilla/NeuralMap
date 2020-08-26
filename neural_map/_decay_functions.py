from numba import jit, float64


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def linear(init, final, total_steps, step):
    return init + step * (final - init) / (total_steps - 1)


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def exponential(init, final, total_steps, step):
    return init * (final / init) ** (step / (total_steps - 1))


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def rational(init, final, total_steps, step):
    b = (total_steps - 1) / ((init / final) - 1)
    return init * b / (b + step)


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def no_decay(init, final, total_steps, step):
    return init

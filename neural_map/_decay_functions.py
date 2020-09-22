"""
Utility functions for computing the decay of some variable over time (steps).

Functions
--------------
.. autosummary::
   :toctree: generated/
   linear
   exponential
   rational
   no_decay

"""
from numba import jit, float64


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def linear(init, final, total_steps, step):
    """
    Compute the linear interpolation between two given values
    and return the required intermediate value.

    Parameters
    ----------
    init: float
        First value.
    final: float
        Last value.
    total_steps: float
        Number of intermediate points.
    step: float
        Intermediate point to compute.

    Raises
    ------
    ZeroDivisionError
        Raised when ``total_steps`` is equal to 1.
        Instead use the ``no_decay`` function.

    Returns
    -------
    interpolation: float
        Value of the ``step`` point in the interpolation.

    Example
    -------
    >>> from neural_map import linear
    >>> linear(1.0, 0.1, 10, 5)
    0.5

    """
    return init + step * (final - init) / (total_steps - 1)


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def exponential(init, final, total_steps, step):
    """
    Compute the exponential interpolation between two given values
    and return the required intermediate value.

    Parameters
    ----------
    init: float
        First value.
    final: float
        Last value.
    total_steps: float
        Number of intermediate points.
    step: float
        Intermediate point to compute.

    Raises
    ------
    ZeroDivisionError
        Raised when ``total_steps`` is equal to 1 or ``init`` is equal to 0.
        Instead use the ``no_decay`` function.

    Returns
    -------
    interpolation: float
        Value of the ``step`` point in the interpolation.

    Notes
    -----
    The exponential interpolation is given by an exponential function
    that matches both ``(0, init)`` and ``(total_steps - 1, final)`` points.

    .. math::

    f(x) = M * e ^ (t * x)
    f(0) = init
    f(total_steps - 1) = final

    Example
    -------
    >>> from neural_map import exponential
    >>> exponential(1.0, 0.1, 10, 5)
    0.2782559402207124

    """
    return init * (final / init) ** (step / (total_steps - 1))


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def rational(init, final, total_steps, step):
    """
    Compute the rational interpolation between two given values
    and return the required intermediate value.

    Parameters
    ----------
    init: float
        First value.
    final: float
        Last value
    total_steps: float.
        Number of intermediate points
    step: float
        Intermediate point to compute.

    Raises
    ------
    ZeroDivisionError
        Raised when ``steps + (total_steps - 1) / ((init / final) - 1)`` is equal to 0
        or when ``final`` is equal to 0.
        Instead use the ``no_decay`` function.

    Returns
    -------
    interpolation: float
        Value of the ``step`` point in the interpolation.

    Notes
    -----
    The rational interpolation is given by a simple rational function
    that matches both ``(0, init)`` and ``(total_steps - 1, final)`` points.

    .. math::

    f(x) = a / (b + x)
    f(0) = init
    f(total_steps - 1) = final

    Example
    -------
    >>> from neural_map import rational
    >>> rational(1.0, 0.1, 10, 5)
    0.16666666666666666

    """
    value = (total_steps - 1) / ((init / final) - 1)
    return init * value / (value + step)


@jit(float64(float64, float64, float64, float64), nopython=True, fastmath=True)
def no_decay(init, _final, _total_steps, _step):
    """
    Return the init value, regardless of all others.

    Parameters
    ----------
    init: float
        First value.
    _final: Any
        Unused value.
    _total_steps: Any
        Unused value.
    _step: Any
        Unused value.

    Returns
    -------
    init: float
        Value entered in the `init` parameter.

    Notes
    -----
    Useful when it is needed a function that accepts four parameters as all others
    decay functions, but it is no needed to compute any decay.

    """
    return init

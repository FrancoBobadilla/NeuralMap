"""
Utility functions for computing the update matrix for the neighbourhood of the best matching unit.

Functions
--------------
.. autosummary::
   :toctree: generated/
   bubble
   conical
   gaussian
   gaussian_cut
   mexican_hat
   no_neighbourhood

"""
from numba import jit, float64
from numpy import empty, exp


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def bubble(positions, bmu, radius, learning_rate):
    """
    Compute a circular area of radius ``radius`` centered at ``bmu`` over the
    nodes at positions ``positions`` and then multiply all by ``learning_rate``.

    Parameters
    ----------
    positions: ndarray
        Cartesian coordinates of the nodes in the 2D virtual space.
        Dimensions should be (x, y, 2).
    bmu: ndarray
        Cartesian coordinate of the best matching unit in the 2D virtual space.
        Dimensions should be (2).
    radius: float
        Radius of neighbourhood.
    learning_rate: float
        Multiplication factor.

    Raises
    ------
    ZeroDivisionError
        Raised when ``radius`` is equal to 0.
        Instead use the ``no_neighbourhood`` function.

    Returns
    -------
    update_map: ndarray
        Array with each node update.
        Dimensions are (x, y).

    Example
    -------
    >>> from numpy import array
    >>> from neural_map import bubble
    >>> bubble(array([[[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]]]), array([1., 1.]), 1.4, 0.5)
    array([[0., 0.5], [0.5, 0.5]])

    """
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)

    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            if (positions[i, j, 0] - bmu[0]) ** 2 + (positions[i, j, 1] - bmu[1]) ** 2 - \
                    radius ** 2 <= 0:
                res[i, j] = learning_rate

            else:
                res[i, j] = 0.

    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def conical(positions, bmu, radius, learning_rate):
    """
    Compute a conical area of radius ``radius`` centered at ``bmu`` over the
    nodes at positions ``positions`` and then multiply all by ``learning_rate``.

    Parameters
    ----------
    positions: ndarray
        Cartesian coordinates of the nodes in the 2D virtual space.
        Dimensions should be (x, y, 2).
    bmu: ndarray
        Cartesian coordinate of the best matching unit in the 2D virtual space.
        Dimensions should be (2).
    radius: float
        Radius of neighbourhood.
    learning_rate: float
        Multiplication factor.

    Raises
    ------
    ZeroDivisionError
        Raised when ``radius`` is equal to 0.
        Instead use the ``no_neighbourhood`` function.

    Returns
    -------
    update_map: ndarray
        Array with each node update.
        Dimensions are (x, y)

    Example
    -------
    >>> from numpy import array
    >>> from neural_map import bubble
    >>> bubble(array([[[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]]]), array([1., 1.]), 1.4, 0.5)
    array([[0., 0.14285714], [0.14285714, 0.5]])

    """
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)

    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            res[i, j] = max(0., 1. - (((positions[i, j, 0] - bmu[0]) ** 2 + (
                        positions[i, j, 1] - bmu[1]) ** 2) ** 0.5) / radius) * learning_rate

    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def gaussian(positions, bmu, radius, learning_rate):
    """
    Compute a gaussian area of standard deviation ``radius`` centered at ``bmu`` over the
    nodes at positions ``positions`` and then multiply all by ``learning_rate``.

    Parameters
    ----------
    positions: ndarray
        Cartesian coordinates of the nodes in the 2D virtual space.
        Dimensions should be (x, y, 2).
    bmu: ndarray
        Cartesian coordinate of the best matching unit in the 2D virtual space.
        Dimensions should be (2).
    radius: float
        Standard deviation.
    learning_rate: float
        Multiplication factor.

    Raises
    ------
    ZeroDivisionError
        Raised when ``radius`` is equal to 0.
        Instead use the ``no_neighbourhood`` function.

    Returns
    -------
    update_map: ndarray
        Array with each node update.
        Dimensions are (x, y).

    Example
    -------
    >>> from numpy import array
    >>> from neural_map import gaussian
    >>> gaussian(array([[[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]]]), array([1., 1.]), 1.4, 0.5)
    array([[0.18022389, 0.30018652], [0.30018652, 0.5]])

    """
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)

    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            res[i, j] = exp(-((positions[i, j, 0] - bmu[0]) ** 2 +
                              (positions[i, j, 1] - bmu[1]) ** 2) / radius ** 2) * learning_rate

    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def gaussian_cut(positions, bmu, radius, learning_rate):
    """
    Compute a gaussian area of standard deviation ``radius`` centered at ``bmu`` over the
    nodes at positions ``positions``, then multiply all by ``learning_rate``,
    and set to 0 all nodes further than ``radius`` from ``bmu``.

    Parameters
    ----------
    positions: ndarray
        Cartesian coordinates of the nodes in the 2D virtual space.
        Dimensions should be (x, y, 2).
    bmu: ndarray
        Cartesian coordinate of the best matching unit in the 2D virtual space.
        Dimensions should be (2).
    radius: float
        Radius of neighbourhood and standard deviation.
    learning_rate: float
        Multiplication factor.

    Raises
    ------
    ZeroDivisionError
        Raised when ``radius`` is equal to 0.
        Instead use the ``no_neighbourhood`` function.

    Returns
    -------
    update_map: ndarray
        Array with each node update.
        Dimensions are (x, y).

    Example
    -------
    >>> from numpy import array
    >>> from neural_map import gaussian_cut
    >>> gaussian_cut(array([[[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]]]), array([1., 1.]), 1.4, 0.5)
    array([[0., 0.30018652], [0.30018652, 0.5]])

    """
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
    """
    Compute a mexican hat wavelet area of standard deviation ``radius`` centered at ``bmu``
    over the nodes at positions ``positions``, then multiply all by ``learning_rate``.

    Parameters
    ----------
    positions: ndarray
        Cartesian coordinates of the nodes in the 2D virtual space.
        Dimensions should be (x, y, 2).
    bmu: ndarray
        Cartesian coordinate of the best matching unit in the 2D virtual space.
        Dimensions should be (2).
    radius: float
        Radius of neighbourhood and standard deviation.
    learning_rate: float
        Multiplication factor.

    Raises
    ------
    ZeroDivisionError
        Raised when ``radius`` is equal to 0.
        Instead use the ``no_neighbourhood`` function.

    Returns
    -------
    update_map: ndarray
        Array with each node update.
        Dimensions are (x, y).

    Example
    -------
    >>> from numpy import array
    >>> from neural_map import mexican_hat
    >>> mexican_hat(array([[[0., 0.], [0., 1.]], [[1., 0.], [1., 1.]]]), array([1., 1.]), 1.4, 0.5)
    array([[-0.00367804, 0.14703013], [0.14703013, 0.5]])

    """
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)

    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            value = -((positions[i, j, 0] - bmu[0]) ** 2 + (positions[i, j, 1] - bmu[1]) ** 2) / \
                    radius ** 2
            res[i, j] = exp(value) * (1. + value) * learning_rate

    return res


@jit(float64[:, :](float64[:, :, :], float64[:], float64, float64), nopython=True, fastmath=True)
def no_neighbourhood(positions, bmu, _radius, learning_rate):
    """
    Set the ``learning_rate`` value to the node with the ``bmu`` position,
    and set to 0 all others.

    Parameters
    ----------
    positions: ndarray
        Cartesian coordinates of the nodes in the 2D virtual space.
        Dimensions should be (x, y, 2).
    bmu: ndarray
        Cartesian coordinate of the best matching unit in the 2D virtual space.
        Dimensions should be (2).
    _radius: Any
        Unused value.
    learning_rate: float
        Multiplication factor.

    Returns
    -------
    update_map: ndarray
        Array with each node update.
        Dimensions are (x, y).

    Notes
    -----
    Useful when it is needed a function that has the same firm as all others
    neighbourhood functions, but it is no needed to compute any real neighbourhood.

    """
    res = empty((positions.shape[0], positions.shape[1]), dtype=positions.dtype)

    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            if positions[i, j, 0] == bmu[0] and positions[i, j, 1] == bmu[1]:
                res[i, j] = learning_rate

            else:
                res[i, j] = 0.

    return res

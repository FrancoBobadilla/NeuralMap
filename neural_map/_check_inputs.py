"""
Utility functions for checking some conditions over data
and raise an Error when some expected condition is not satisfied

Functions
--------------
.. autosummary::
   :toctree: generated/
   length
   ndarray_and_shape
   value_type
   positive
   function
   numpy_matrix

"""
from numpy import isinf, isnan, ndarray


def length(first_array, second_array):
    """
    Check that the length of two arrays are the same.

    Parameters
    ----------
    first_array: array_like
        First array to compare
    second_array: array_like
        Second array to compare

    Raises
    ------
    ValueError
        Raised when input arrays don't have the same length.

    Returns
    -------
    success: bool
        Return true

    """
    if len(first_array) != len(second_array):
        raise ValueError(
            'Input data and labels must have the same length.'
        )

    return True


def ndarray_and_shape(data, expected_shape):
    """
    Check that ``data`` is a ndarray with the ``expected_shape``, and doesn't any nan or inf value.

    Parameters
    ----------
    data: array_like
        Data to check, should be a ndarray
    expected_shape: tuple
        Expected shape of ``data``

    Raises
    ------
    ValueError
        Raised when data is not a ndarray,
        or doesn't have the expected shape,
        or contains a nan or inf value

    Returns
    -------
    success: bool
        Return true

    """
    if not isinstance(data, ndarray):
        raise ValueError(
            'Input data is {data_type}, but should be {types}!'
            .format(data_type=type(data), types=ndarray)
        )

    if not data.shape == expected_shape:
        raise ValueError(
            'Input data has shape {data_shape}, but should be {shape}!'
            .format(data_shape=data.shape, shape=expected_shape)
        )

    if isnan(data).any():
        raise ValueError(
            'Input data contains nan values!'
        )

    if isinf(data).any():
        raise ValueError(
            'Input data contains inf values!'
        )

    return True


def value_type(value, types):
    """
    Check that the ``value`` type is one of ``types``

    Parameters
    ----------
    value: Any
        Variable to check its type
    types: type or tuple or array
        Acceptable types
        Could be one type, or a tuple or array of types

    Raises
    ------
    ValueError
        Raised when ``value`` is not any of the specified ``types``

    Returns
    -------
    success: bool
        Return true

    """
    if not isinstance(value, types):
        raise ValueError(
            'Value {value} is {value_type}, but should be {types}!'
            .format(value=value, value_type=type(value), types=types)
        )

    return True


def positive(value):
    """
    Check that the ``value`` is greater than zero

    Parameters
    ----------
    value: number
        Value to check

    Raises
    ------
    ValueError
        Raised when ``value`` is not greater than zero

    Returns
    -------
    success: bool
        Return true

    """
    if not value > 0:
        raise ValueError(
            'Value {value} should be positive!'
            .format(value=value)
        )

    return True


def function(func):
    """
    Check that ``func`` is callable

    Parameters
    ----------
    func: callable
        Value to check

    Raises
    ------
    ValueError
        Raised when ``func`` is not callable

    Returns
    -------
    success: bool
        Return true

    """
    if not callable(func):
        raise ValueError(
            'Value {func} is {value_type}, but should be callable!'
            .format(func=func, value_type=type(func)))

    return True


def numpy_matrix(data, expected_len):
    """
    Check that ``data`` is a ndarray,
    has two dimensions,
    its first dimension is greater than 0,
    its second dimension has the `expected_len`,
    and it doesn't contain any nan or inf value

    Parameters
    ----------
    data: ndarray
        2D ndarray to check
    expected_len: int
        Expected value of the data second dimension

    Raises
    ------
    ValueError
        Raised when ``data`` is not a ndarray,
        or has an amount of dimensions different than 2,
        or its first dimension is equal to 0,
        or its second dimension is different than ``expected_len``
        or contains any nan or inf value

    Returns
    -------
    success: bool
        Return true

    """
    if not isinstance(data, ndarray):
        raise ValueError(
            'Input data is {data_type}, but should be {types}!'
            .format(data_type=type(data), types=ndarray)
        )

    if not len(data.shape) == 2:
        raise ValueError(
            'Input data has {data_axis} axis, but should have {axis}!'
            .format(data_axis=len(data.shape), axis=2)
        )

    if not data.shape[0] > 0:
        raise ValueError(
            'Input data has {data_rows} rows, but should have more than {rows}!'
            .format(data_rows=data.shape[0], rows=0)
        )

    if not data.shape[1] == expected_len:
        raise ValueError(
            'Received {data_length} features, expected {length}.'
            .format(data_length=data.shape[1], length=expected_len)
        )

    if isnan(data).any():
        raise ValueError(
            'Input data contains nan values!'
        )

    if isinf(data).any():
        raise ValueError(
            'Input data contains inf values!'
        )

    return True

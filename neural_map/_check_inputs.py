from numpy import isinf, isnan, ndarray


def length(first_array, second_array):
    if len(first_array) != len(second_array):
        raise ValueError(
            'Input data and labels must have the same length.'
        )

    return True


def ndarray_and_shape(data, expected_shape):
    if not isinstance(data, ndarray):
        raise TypeError(
            'Input data is {data_type}, but should be {types}!'
            .format(data_type=type(data), types=ndarray)
        )

    if not data.shape == expected_shape:
        raise TypeError(
            'Input data has shape {data_shape}, but should be {shape}!'
            .format(data_shape=data.shape, shape=expected_shape)
        )

    if isnan(data).any():
        raise TypeError(
            'Input data contains nan values!'
        )

    if isinf(data).any():
        raise TypeError(
            'Input data contains inf values!'
        )

    return True


def value_type(value, types):
    if not isinstance(value, types):
        raise TypeError(
            'Value {value} is {value_type}, but should be {types}!'
            .format(value=value, value_type=type(value), types=types)
        )

    return True


def positive(value):
    if not value > 0:
        raise TypeError(
            'Value {value} should be positive!'
            .format(value=value)
        )

    return True


def function(func):
    if not callable(func):
        raise TypeError(
            'Value {func} is {value_type}, but should be callable!'
            .format(func=func, value_type=type(func)))

    return True


def numpy_matrix(data, expected_len):
    if not isinstance(data, ndarray):
        raise TypeError(
            'Input data is {data_type}, but should be {types}!'
            .format(data_type=type(data), types=ndarray)
        )

    if not len(data.shape) == 2:
        raise TypeError(
            'Input data has {data_axis} axis, but should have {axis}!'
            .format(data_axis=len(data.shape), axis=2)
        )

    if not data.shape[0] > 0:
        raise TypeError(
            'Input data has {data_rows} rows, but should have more than {rows}!'
            .format(data_rows=data.shape[0], rows=0)
        )

    if not data.shape[1] == expected_len:
        raise ValueError(
            'Received {data_length} features, expected {length}.'
            .format(data_length=data.shape[1], length=expected_len)
        )

    if isnan(data).any():
        raise TypeError(
            'Input data contains nan values!'
        )

    if isinf(data).any():
        raise TypeError(
            'Input data contains inf values!'
        )

    return True

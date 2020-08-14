from numpy import isinf, isnan, ndarray
from inspect import signature


def data_to_analyze(data, expected_len):
    if expected_len != len(data[0]):
        raise ValueError(
            'Received {data_length} features, expected {length}.'
                .format(data_length=len(data[0]), length=expected_len)
        )

    return True


def attachments(data, attachments):
    if len(data) != len(attachments):
        raise ValueError(
            'Input data and labels must have the same length.'
        )

    return True


def shape(data, expected_shape):
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
                .format(value=value, value_type=type(value))
        )

    return True


def function(func, num_params):
    if not isinstance(num_params, int):
        raise TypeError(
            'num_params should be int!'
        )

    if not callable(func):
        raise TypeError(
            'Value {func} is {value_type}, but should be a function with {num_params} params!'
                .format(func=func, value_type=type(func), num_params=num_params))

    if not len(signature(func).parameters) is num_params:
        raise TypeError(
            'Function {func} has {funct_num_params} params, but should have {num_params} params!'
                .format(func=func, funct_num_params=len(signature(func).parameters), num_params=num_params)
        )

    return True


def np_matrix(data):
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

    if not data.shape[1] > 0:
        raise TypeError(
            'Input data has {data_columns} columns, but should have more than {columns}!'
                .format(data_columns=data.shape[1], columns=0)
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

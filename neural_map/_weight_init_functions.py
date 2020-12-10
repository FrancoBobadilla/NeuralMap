"""
Utility functions for initializing weights vector of each node at the beginning of training.

Functions
--------------
.. autosummary::
   :toctree: generated/
   standard
   uniform
   pca
   pick_from_data
   no_init

"""
from numpy import random, cov, transpose, argsort, linspace, arange
from numpy.linalg import eig


def standard(_data, weights):
    """
    Randomly initialize the weights with a random distribution with mean of 0 and std of 1.

    Parameters
    ----------
    _data: ndarray
        Data to pick to initialize weights.
    weights: ndarray
        Previous weight values.

    Returns
    -------
    weights: ndarray
        New weight values

    """
    return random.normal(0., 1., weights.shape)


def uniform(_data, weights):
    """
    Randomly initialize the weights with values between 0 and 1.

    Parameters
    ----------
    _data: ndarray
        Data to pick to initialize weights.
    weights: ndarray
        Previous weight values.

    Returns
    -------
    weights: ndarray
        New weight values

    """
    return random.rand(weights.shape)


def pca(data, weights):
    """
    Initialize the weights to span the first two principal components.

    It is recommended to normalize the data before initializing the weights
    and use the same normalization for the training data.

    Parameters
    ----------
    data: ndarray
        Data to pick to initialize weights.
    weights: ndarray
        Previous weight values.

    Returns
    -------
    weights: ndarray
        New weight values

    """
    principal_components_length, principal_components = eig(cov(transpose(data)))
    components_order = argsort(-principal_components_length)

    for i, component_1 in enumerate(
            linspace(-1, 1, weights.shape[0]) * principal_components[components_order[0]]):
        for j, component_2 in enumerate(
                linspace(-1, 1, weights.shape[1]) * principal_components[components_order[1]]):
            weights[i, j] = component_1 + component_2

    return weights


def pick_from_data(data, weights):
    """
    Initialize the weights picking random samples from the input data.

    Parameters
    ----------
    data: ndarray
        Data to pick to initialize weights.
    weights: ndarray
        Previous weight values.

    Returns
    -------
    weights: ndarray
        New weight values

    """
    indices = arange(data.shape[0])

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = data[random.choice(indices)]

    return weights


def no_init(_data, weights):
    """
    Return the entered weights.

    Parameters
    ----------
    _data: ndarray
        Data to pick to initialize weights.
    weights: ndarray
        Previous weight values.

    Returns
    -------
    weights: ndarray
        New weight values

    Notes
    -----
    Useful when it is needed a function that accepts two parameters as all others
    weight init functions, but it is no needed to calculate any new value.

    """
    return weights

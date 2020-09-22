"""
NeuralMap is a data analysis tool that generates discrete, low dimensional representation of the
input data space using the Self Organizing Maps algorithm.

Self Organizing Maps is a neural network that applies a non-supervised training algorithm to get a
low-dimensional discrete representation of the input space, preserving topology in both local and
global scales.

This implementation also adds further functionality for clustering and visualization, such
as Relative Positions, HDBSCAN and several other techniques.

This package includes:
 - The main class neural_map
 - Customizable SOM-specific plots
 - Decay functions for configuring SOM training
 - Neighbourhood functions for configuring SOM training

"""
from ._neural_map import *
from ._plot import *
from ._decay_functions import *
from ._neighbourhood_functions import *

#!/usr/bin/env python

from distutils.core import setup

setup(
    name='neural_map',
    packages=['neural_map'],
    version='0.1',
    license='MIT',
    description='NeuralMap is a data analysis tool based on Self-Organizing Maps',
    author='Franco José Bobadilla',
    author_email='1709673@ucc.edu.ar',
    url='https://github.com/FrancoBobadilla/NeuralMap',
    download_url='https://github.com/FrancoBobadilla/NeuralMap',
    keywords=['som', 'self-organizing maps', 'dimensionality reduction', 'machine learning', 'clustering', 'hdbscan'],
    install_requires=[
        'matplotlib',
        'mpl_toolkits',
        'collections',
        'inspect',
        'sklearn',
        'numpy',
        'scipy',
        'numba',
        'hdbscan',
        'scikit-learn-extra'
    ],
)

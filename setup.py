import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='neural_map',
    version="0.0.7",
    author="Franco José Bobadilla",
    author_email='1709673@ucc.edu.ar',
    description='NeuralMap is a data analysis tool based on Self-Organizing Maps',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/FrancoBobadilla/NeuralMap',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    license='MIT',
    keywords=[
        'self-organizing-maps',
        'dimensionality-reduction',
        'machine-learning',
        'clustering',
        'hdbscan',
        'kmeans-clustering',
        'kmedoids-clustering',
        'data-analysis',
        'unsupervised-learning'
    ],
    install_requires=[
        'hdbscan',
        'scipy',
        'matplotlib',
        'numpy',
        'numba',
        'scikit_learn',
        'scikit_learn_extra'
    ],
)

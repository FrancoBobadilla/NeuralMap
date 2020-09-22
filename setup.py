import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='neural_map',
    version="0.0.3",
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
        'hdbscan~=0.8.26',
        'scipy~=1.4.1',
        'matplotlib~=3.2.1',
        'scikit_learn_extra~=0.1.0b2',
        'numpy~=1.18.3',
        'numba~=0.50.1',
        'scikit_learn~=0.23.2'
    ],
)

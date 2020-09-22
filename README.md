# NeuralMap

NeuralMap is a **data analysis** tool to generate a **discrete**, **low-dimensional** representation of the input data 
space using the **[Self Organizing Maps algorithm](https://en.wikipedia.org/wiki/Self-organizing_map)**. 

This tool supports a **wide range of configurations**:
 - Custom **distance metric**
 - **Hexagonal or square** arrangement
 - **Toroidal or flat** topology
 - Custom radius or learning rate **decay functions**
 - Custom **neighbourhood functions**
 
After training a **NeuralMap instance**, you will be able to get **useful information** about your data, by mapping 
observations to the SOM, watching the **features distribution** over the map, analysing a dataset to get the
**quantization error**, **activation frequency** and **mean distance** for each node, and **evaluating** the SOM.

Since the use of SOM for **clustering** is very spread, this tool employs some common clustering algorithms:
 - [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/index.html)
 - [K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
 - [K-medoids](https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html)

And also includes several highly customizable **visualization methods**, some of them based on the 
**[Relative Positions technique](https://www.researchgate.net/publication/6292810_Improving_cluster_visualization_in_self-organizing_maps_Application_in_gene_expression_data_analysis)** 
to improve the interpretability of results and clusters recognition.

## Installation

NeuralMap has the following requirements:
 - python >= 3.6
 - numpy 1.18.3
 - numba 0.50.1
 - scikit_learn_extra 0.1.0b2
 - matplotlib 3.2.1
 - scipy 1.4.1
 - hdbscan 0.8.26
 - scikit_learn 0.23.2

To install NeuralMap use:
```bash
pip install neural-map
```

## Get started

After installing NeuralMap, **obtain** and **prepare** your dataset. For this example we will create a dataset with 10 
[blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) in a five 
dimensional space and scale it to fit the observations to the range [0, 1]:
```python
from sklearn import datasets, preprocessing

blobs = datasets.make_blobs(n_samples=1000, n_features=5, centers=10)
scaler = preprocessing.MinMaxScaler()
data = scaler.fit_transform(blobs[0])
```

Then, **create** a NeuralMap instance and **train it** with the data:
```python
from neural_map import NeuralMap

nm = NeuralMap(5, 'euclidean', columns=10, rows=10)
nm.train(data, n_epochs=20)
```

Now you can obtain a **discrete representation**:
```python
nm.plot_analysis(data)
```
<img src="https://github.com/FrancoBobadilla/NeuralMap/raw/master/examples/images/RP-HDBSCAN.png" alt="RP-HDBSCAN">

As you can see, NeuralMap was able to successfully **discover** and cluster all the **original blobs**.

It's also possible to get the **U-matrix**:
```python
nm.plot_unified_distane_matrix(data)
```
<img src="https://github.com/FrancoBobadilla/NeuralMap/raw/master/examples/images/U-matrix.png" alt="U-matrix">

Here is also possible to recognize the 10 original blobs.

## Documentation

For more details, see the **[NeuralMap documentation]()**.

If you have a **question**, please open an **issue**.

## Contributing

To **contribute**, do the following:
 - Open an **issue** to discuss possible changes or ask questions
 - Create a **fork** of the project
 - Create a new **branch** and push your changes
 - Create a **pull request**
 
## License

NeuralMap is licensed under [MIT](https://choosealicense.com/licenses/mit/) License.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

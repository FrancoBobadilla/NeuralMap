# NeuralMap

NeuralMap is a data analysis tool that generates discrete, low dimensional representation of the input data space using the Self Organizing Maps algorithm.
What does this project do?
Why is this project useful?
How do I get started?
Where can I get more help, if I need it?

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install NeuralMap.

```bash
pip install neural-map
```

## Usage

First, obtain and prepare your dataset:

```python
from sklearn import datasets, preprocessing

blobs = datasets.make_blobs(n_samples=1000, n_features=5, centers=10)
scaler = preprocessing.MinMaxScaler()
data = scaler.fit_transform(blobs[0])
```

Then, create a NeuralMap instance and train it with the data:
```python
from neural_map import NeuralMap

nm = NeuralMap(4, 'euclidean', columns=10, rows=10)
nm.train(data, n_epochs=20)
```

Now you can obtain the discrete representation:
```python
nm.plot_analysis(data)
```
<img src="https://github.com/FrancoBobadilla/NeuralMap/raw/master/examples/images/RP-HDBSCAN.png" alt="RP-HDBSCAN" width=450>

It's also possible to get the U-matrix
```python
nm.plot_unified_distane_matrix(data)
```
<img src="https://github.com/FrancoBobadilla/NeuralMap/raw/master/examples/images/U-matrix.png" alt="U-matrix" width=450>

## Contributing
To contribute, open an issue to discuss what you would like to change, 
and propose changes with a pull request (that includes tests).

## License
[MIT](https://choosealicense.com/licenses/mit/)
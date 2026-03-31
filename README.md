# Summary

This project is an implementation of a neural network framework built from scratch using NumPy. My goal was to explore and better understand how neural networks work under the hood. The code provides modular implementations of common components such as densely-connected layers, activation and loss functions, optimizers, and a simple sequential model interface. The repository also includes notebooks that walk through the underlying ideas step by step, from gradient computation to linear regression and nonlinear classification, together with documentation covering both the API and the mathematical background.


# Installation

```bash
# set-up environment
source ./0_setup.sh

```bash
# run integration tests
./scripts/run_tests.sh
```

# Documentation

```bash
# build documentation
mkdocs build

# start development server (to test updates to documentation)
mkdocs serve
```

# Features

* Sequential network model
* Layer: Dense 
* Activation functions:
	* Sigmoid
	* Softmax
* Optimizer: SGD with momentum
* Batch generator
* Loss functions:
	* MSE (Mean Squared Error)
	* BCE (Binary cross entropy)
	* CCE (Categorical Cross Entropy)
* Examples (notebooks):
  * WIP
* Documentation
* Testing coverage

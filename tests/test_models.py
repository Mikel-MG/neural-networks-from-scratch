import pytest

from nnfs.activations import Sigmoid
from nnfs.datasets.data_generators import generate_linear_data, generate_XOR_gate
from nnfs.layers import Dense
from nnfs.losses import BCE, MSE
from nnfs.model import Sequential
from nnfs.optimizers import SGD


def test_linear_model_training_step():
    # generate data
    X_data, y_true = generate_linear_data(3, -7, 100)

    # define the model
    list_layers = [Dense(1, 1)]
    loss_func = MSE()
    optimizer = SGD(lr=0.01)
    model = Sequential(list_layers, loss_func, optimizer)

    # compute initial loss
    y_pred = model.forward(X_data)
    loss_start = model.loss.forward(y_pred, y_true)

    # run a training step
    model.run_training_epoch(X_data, y_true)

    # compute loss after training step
    y_pred = model.forward(X_data)
    loss_end = model.loss.forward(y_pred, y_true)

    # check that loss value has decreased
    assert loss_end < loss_start


def test_linear_model_training_optimization():
    # generate data
    X_data, y_true = generate_linear_data(3, -7, 5000)

    # define the model
    list_layers = [Dense(1, 1)]
    loss_func = MSE()
    optimizer = SGD(lr=0.01)
    model = Sequential(list_layers, loss_func, optimizer)

    # train the model
    model.fit(X_data, y_true, N_epochs=5000)

    # extract layer parameters
    layer = model.layers[0]
    assert isinstance(layer, Dense)
    opt_W = layer.W.item()
    opt_b = layer.b.item()

    # check that optimized parameters match ideal values
    assert 3 == pytest.approx(opt_W, abs=0.1)
    assert -7 == pytest.approx(-7, opt_b, abs=0.1)


def test_nonlinear_model_training_step():
    # generate data
    X_data, y_true = generate_XOR_gate()

    # define the model
    list_layers = [Dense(2, 2), Sigmoid(), Dense(2, 1), Sigmoid()]
    loss_func = BCE()
    optimizer = SGD()
    model = Sequential(list_layers, loss_func, optimizer)

    # compute initial loss
    y_pred = model.forward(X_data)
    loss_start = model.loss.forward(y_pred, y_true)

    # run a training step
    model.run_training_epoch(X_data, y_true)

    # compute loss after training step
    y_pred = model.forward(X_data)
    loss_end = model.loss.forward(y_pred, y_true)

    # check that loss value has decreased
    assert loss_end < loss_start

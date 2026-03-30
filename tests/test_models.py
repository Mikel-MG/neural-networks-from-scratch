import pytest

from nnfs.activations import Sigmoid, Softmax
from nnfs.datasets.data_generators import (
    generate_concentric_circles,
    generate_linear_data,
    generate_XOR_gate,
)
from nnfs.layers import Dense
from nnfs.losses import BCE, CCE, MSE
from nnfs.model import Sequential
from nnfs.optimizers import SGD
from nnfs.utils import class_to_onehot, shuffle


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
    X_data_test, y_true_test = generate_linear_data(3, -7, 1000)

    # define the model
    list_layers = [Dense(1, 1)]
    loss_func = MSE()
    optimizer = SGD(lr=0.01)
    model = Sequential(list_layers, loss_func, optimizer)

    # train the model
    N_epochs = 5000
    history = model.fit(
        X_data,
        y_true,
        N_epochs=N_epochs,
        X_test_inputs=X_data_test,
        y_test_outputs=y_true_test,
    )

    # extract layer parameters
    layer = model.layers[0]
    assert isinstance(layer, Dense)
    opt_W = layer.W.item()
    opt_b = layer.b.item()

    # check that optimized parameters match ideal values
    assert 3 == pytest.approx(opt_W, abs=0.1)
    assert -7 == pytest.approx(-7, opt_b, abs=0.1)

    # check that recorded losses feature expected dimensions
    train_loss = history["loss"]
    test_loss = history["test_loss"]

    assert train_loss.shape[0] == N_epochs
    assert test_loss.shape[0] == N_epochs


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


def test_multilabel_classification_model():
    # generate data
    X_data, y_true = generate_concentric_circles(500, 3)
    X_data, y_true = shuffle([X_data, y_true])
    y_true_onehot = class_to_onehot(y_true, 3)

    # define the model
    list_layers = [
        Dense(2, 16),
        Sigmoid(),
        Dense(16, 3),
        Softmax(),
    ]
    loss = CCE()
    optimizer = SGD(lr=0.001, momentum=0.95)
    model = Sequential(list_layers, loss, optimizer)

    # check that the loss has been significantly lowered
    history = model.fit(X_data, y_true_onehot, 10000, debug_flag=True)
    loss_train = history["loss"]
    assert loss_train[-1].mean() < 0.5


def test_model_summary():
    # define the model
    list_layers = [Dense(2, 2), Sigmoid(), Dense(2, 1), Sigmoid()]
    loss_func = BCE()
    optimizer = SGD()
    model = Sequential(list_layers, loss_func, optimizer)

    # check that all layers have a unique name (after model initialization)
    list_names = [layer.name for layer in model.layers]
    assert len(set(list_names)) == len(list_names)

    # interactive check (run script directly to print summary)
    model.summary()


test_model_summary()

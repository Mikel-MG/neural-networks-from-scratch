from nnfs.datasets.data_generators import generate_linear_data
from nnfs.layers import Dense
from nnfs.losses import MSE
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

    assert loss_end < loss_start

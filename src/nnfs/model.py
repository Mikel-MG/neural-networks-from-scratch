import numpy as np

from .layers import BaseLayer
from .losses import Loss
from .optimizers import BaseOptimizer


class Sequential:
    def __init__(
        self,
        list_layers: list,
        loss: Loss,
        optimizer: BaseOptimizer,
    ):
        self.layers: list[BaseLayer] = list_layers
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X_inputs: np.ndarray):
        """
        Runs the forward pass for the entire model.

        Each layer takes the input of the previous layer (the first layer takes the input data) and passes its output to the next layer.
        """
        x = X_inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_loss):
        # this is dL_d_output
        grad = d_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_weights(self) -> None:
        """
        Update all layer weights and biases using stored gradients.
        """
        # gradient descent update
        for layer in self.layers:
            if layer.trainable is not None:
                for param, grad in layer.trainable:
                    self.optimizer.update(param, grad)

    def run_training_epoch(
        self,
        X_inputs: np.ndarray,
        y_true: np.ndarray,
    ):
        # run forward pass
        y_pred = self.forward(X_inputs)
        loss_value = self.loss.forward(y_pred, y_true)

        # compute the dL_d_ytrue gradient
        d_loss = self.loss.backward()

        # run backward pass
        _grad = self.backward(d_loss)

        self.update_weights()

        # re-compute loss
        y_pred = self.forward(X_inputs)
        loss_value = self.loss.forward(y_pred, y_true)
        return loss_value

    def fit(
        self,
        X_inputs: np.ndarray,
        y_outputs: np.ndarray,
        N_epochs: int,
        lr: float = 0.01,
        debug_flag: bool = False,
    ):
        prev_print_epoch = 0
        print_freq = N_epochs / 10

        for i_epoch in range(N_epochs):
            loss = self.run_training_epoch(X_inputs, y_outputs)
            if i_epoch >= prev_print_epoch + print_freq:
                prev_print_epoch += print_freq
                print(f"Epoch {i_epoch} - Loss: {loss}")
        print(f"Epoch {i_epoch} - Loss: {loss}")

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.trainable)
        return weights

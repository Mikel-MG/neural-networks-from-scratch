import numpy as np

from .layers import BaseLayer
from .losses import Loss
from .optimizers import BaseOptimizer


class Sequential:
    """Sequential neural network model.

    Represents a neural network composed of a sequence of layers, where the
    output of each layer is used as the input to the next one.

    Args:
        list_layers (list): Ordered list of layers defining the model architecture.
        loss (Loss): Loss object that specifies loss function.
        optimizer (BaseOptimizer): Optimizer object that updates the layer parameters during training.

    Attributes:
        layers (list): Stored `list_layers` argument.
        loss (Loss): Stored `loss` argument.
        optimizer (BaseOptimizer): Stored `optimizer` argument.
    """

    def __init__(
        self,
        list_layers: list,
        loss: Loss,
        optimizer: BaseOptimizer,
    ):
        self.layers: list[BaseLayer] = list_layers
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X_inputs: np.ndarray) -> np.ndarray:
        """Runs the forward pass for the entire model.

        Each layer takes the input of the previous layer (the first layer
        takes the input data) and passes its output to the next layer.

        Args:
            X_inputs (np.ndarray): Input data to the model, with shape=(M samples, N features)

        Returns:
            Predictions produced by the model.
        """
        x = X_inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_loss: np.ndarray) -> np.ndarray:
        """Runs the backward pass for the entire model.

        Each layer takes the gradient of the next layer (the last layer
        takes it from the loss function) and passes its own gradient to
        the previous layer.

        The gradient that the next layer represent dL / d_input_data (to the next layer)

        Args:
            d_loss (np.ndarray): Gradient provided by the next layer.

        Returns:
            dL / d_input_data (to the current layer), which is passed to the previous layer.
        """
        # this is dL_d_output
        grad = d_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_weights(self) -> None:
        """
        Update all layer weights and biases using stored gradients.
        """
        # TODO: Remove this method after implementing testing

        # gradient descent update
        for layer in self.layers:
            if layer.trainable is not None:
                for param, grad in layer.trainable:
                    self.optimizer.update(param, grad)

    def run_training_epoch(
        self,
        X_inputs: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Runs a training epoch, optimizing the model parameters.

        It uses all the available data.

        Args:
            X_inputs (np.ndarray): Input data to the model, with shape=(M samples, N features)
            y_true (np.ndarray): Ground-truth values for the prediction task.

        Returns:
            The computed loss value.
        """
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

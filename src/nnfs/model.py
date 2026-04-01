import numpy as np

from .layers import BaseLayer, Dense
from .losses import Loss
from .optimizers import BaseOptimizer
from .utils import BatchGenerator


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

        # add suffixes to layer names to ensure uniqueness
        for i, layer in enumerate(self.layers):
            layer.index = i

    def forward(self, X_inputs: np.ndarray) -> np.ndarray:
        """Runs the forward pass for the entire model.

        Each layer takes the input of the previous layer (the first layer
        takes the input data) and passes its output to the next layer.

        Args:
            X_inputs (np.ndarray): Input data to the model, with shape=(M samples, N features).

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

        The update mechanism depends on the optimizer.
        """
        for layer in self.layers:
            if layer.trainable is not None:
                layer_name = layer.name
                for param_name, param, grad in layer.trainable:
                    param_id = f"{layer_name}.{param_name}"
                    self.optimizer.update(param_id, param, grad)

    def run_training_epoch(
        self,
        X_inputs: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Runs a training epoch, optimizing the model parameters.

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

        # update parameters
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
        batch_size: int = -1,
        X_test_inputs: None | np.ndarray = None,
        y_test_outputs: None | np.ndarray = None,
        debug_flag: bool = True,
    ) -> dict:
        """Runs the training loop for the model.

        In this loop, the forward and backward passes are computed to produce
        optimal gradients, and then the model parameters are updated according
        to those gradients. This loop runs for the specified number of epochs.

        Each returned training metric is an array of shape=`(N_epochs, N_batches)`.

        Args:
            X_inputs (np.ndarray): Input data to the model, with shape=(M samples, N features).
            y_outputs (np.ndarray): Ground-truth values for the prediction task.
            N_epochs (int): Number of epochs to run during training.
            batch_size (int): Number of samples per batch. By default, the entire dataset is used.
            X_test_inputs (None | np.ndarray): Input data for validation.
            y_test_outputs (None | np.ndarray): Ground-truth values for validation.
            debug_flag (bool): Flag which specifies whether to output debugging logs.

        Returns:
            A dictionary containing training metrics (such as `loss`).
        """
        # store training metrics
        list_losses = []
        list_test_losses = []

        # generate array of epochs to report
        epoch_to_report = np.arange(0, N_epochs + 1, N_epochs // 10)
        epoch_to_report[0] = 1  # start reporting at epoch=1

        for i_epoch in range(1, N_epochs + 1):
            # store losses for each batch (for this epoch)
            epoch_losses = []

            # run training epoch on each batch
            batch_generator = BatchGenerator(X_inputs, y_outputs, batch_size)
            for batch_counter in range(batch_generator.num_batches):
                X_data_batch, y_data_batch = batch_generator.next()
                loss = self.run_training_epoch(X_data_batch, y_data_batch)
                # store loss
                epoch_losses.append(loss)

            # report progress each epoch
            if debug_flag is True and i_epoch in epoch_to_report:
                print(f"Epoch {i_epoch} - Loss: {loss}")

            # store losses for this epoch
            list_losses.append(epoch_losses)

            # store loss for testing dataset
            if X_test_inputs is not None and y_test_outputs is not None:
                y_test_pred = self.forward(X_test_inputs)
                val_loss = self.loss.forward(y_test_pred, y_test_outputs)
                list_test_losses.append(val_loss)

        # cast each metric as an appropriately shaped np.ndarray
        dict_history = {}

        # store history of losses
        array_losses = np.array(list_losses)
        assert array_losses.shape[0] == N_epochs
        dict_history["loss"] = array_losses

        # add the list of testing losses if applicable
        if len(list_test_losses) > 0:
            array_test_losses = np.array(list_test_losses)
            dict_history["test_loss"] = array_test_losses

        return dict_history

    def summary(self):
        """Prints a text summary of the model architecture."""

        N_total_params = 0
        sep = " " * 4
        print("Model layers:")
        for layer in self.layers:
            print(f"{sep}* {layer.name} ", end="")

            # get input and output dimensions, if applicable
            if isinstance(layer, Dense):
                input_size, output_size = layer.W.shape
                print(f" | Dimensions: {input_size} x {output_size}", end="")

            # get number of parameters
            if layer.trainable is not None:
                N_params = sum(
                    [np.prod(param.shape) for name, param, grads in layer.trainable]
                )
                N_total_params += N_params
                print(f" | Parameters: {N_params}", end="")

            print()
        print(f"{sep}{'-' * 20}")
        print(f"{sep}Total parameters: {N_total_params}")

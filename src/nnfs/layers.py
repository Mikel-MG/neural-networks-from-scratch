from abc import ABC, abstractmethod

import numpy as np


class BaseLayer(ABC):
    """Template for neural network layers.

    Layers have to define a `trainable` property that returns a
    list of (param, grad) tuples.

    Layers also have to define a 'name' property which specifies
    the base name of the layer.
    """

    @abstractmethod
    def forward(self, X_input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_next: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def trainable(self):
        return None

    @property
    @abstractmethod
    def name(self):
        return None


class Dense(BaseLayer):
    """
    A fully connected neural network layer.

    Args:
        input_size (int): Dimensionality of the input to the layer.
        output_size (int): Dimensionality of the output of the layer.

    Attributes:
        W (np.ndarray): weight parameters.
        b (np.ndarray): bias parameters.
        dW (np.ndarray): gradient of loss w.r.t weight parameters.
        db (np.ndarray): gradient of loss w.r.t bias parameters.
        X_input (np.ndarray): cached input to the layer.

    """

    def __init__(self, input_size: int, output_size: int):
        # layer parameters
        self.W = np.random.uniform(size=(input_size, output_size))
        self.b = np.zeros(shape=(1, output_size))

        # cache for gradients and inputs
        self.dW = np.zeros(shape=self.W.shape)
        self.db = np.zeros(shape=self.b.shape)
        self.X_input = np.zeros(1)

    def forward(self, X_input: np.ndarray) -> np.ndarray:
        """Computes the forward pass for the layer.

        Args:
            X_input (np.ndarray): Input data to be transformed by the layer.

        Returns:
            np.ndarray: Output of the layer.

        """
        self.X_input = X_input
        output = np.matmul(X_input, self.W) + self.b
        return output

    def backward(self, grad_next: np.ndarray) -> np.ndarray:
        """Computes the backward pass for the layer.

        This function updates the `dW` and `db` attributes of the layer.

        Args:
            grad_next (np.ndarray): Gradients fed back from the next layer during backpropagation.

        Returns:
            np.ndarray: Gradient of the loss w.r.t the input to the layer.
        """
        # Gradients w.r.t parameters
        self.dW = np.matmul(self.X_input.T, grad_next)  # shape: (input_dim, output_dim)
        self.db = np.sum(grad_next, axis=0, keepdims=True)  # shape: (1, output_dim)

        # Gradient w.r.t input (to propagate backward)
        d_input = np.matmul(grad_next, self.W.T)  # shape: (batch_size, input_dim)
        return d_input

    @property
    def trainable(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns the layer's trainable parameters and their gradients.

        Each element is a tuple (param, grad) representing a parameter array
        and its corresponding gradient, which are used by the optimizer during training.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: A list of tuples containing `(parameter, grad_parameters)` for each trainable parameter.
        """
        return [(self.W, self.dW), (self.b, self.db)]

    @property
    def name(self) -> str:
        """Returns the layer's name.

        The name can be arbitrary but it has to be unique for each of the layer types.

        It is used by the model to summarize layer architecture, as well as to cache
        layer-specific gradients (for example, to implement momentum).
        """
        return "Dense"

from abc import ABC, abstractmethod

import numpy as np


class BaseLayer(ABC):
    """
    Base class of a layer

    Layers may optionally define a `trainable` property returning
    a list of (param, grad) tuples. Layers without trainable parameters
    do not need to implement it.
    """

    @abstractmethod
    def forward(self, X_input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_next: np.ndarray) -> np.ndarray:
        pass

    @property
    def trainable(self):
        return None


class Dense(BaseLayer):
    def __init__(self, input_size, output_size):
        # layer parameters
        self.W = np.random.uniform(size=(input_size, output_size))
        self.b = np.zeros(shape=(1, output_size))

        # cache for gradients and inputs
        self.dW = np.zeros(shape=self.W.shape)
        self.db = np.zeros(shape=self.b.shape)
        self.X_input = np.zeros(1)

    def forward(self, X_input: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for this layer

        For example, for the first layer:
            * X_input: M_samples x N_features
            * self.W: N_features x output_size
            * self.b: 1 x output_size
            * output: M_samples x output_size
        """
        self.X_input = X_input
        output = np.matmul(X_input, self.W) + self.b
        return output

    def backward(self, grad_next: np.ndarray):
        """
        Computes the backward pass for this layer
        """
        # Gradients w.r.t parameters
        self.dW = np.matmul(self.X_input.T, grad_next)  # shape: (input_dim, output_dim)
        self.db = np.sum(grad_next, axis=0, keepdims=True)  # shape: (1, output_dim)

        # Gradient w.r.t input (to propagate backward)
        d_input = np.matmul(grad_next, self.W.T)  # shape: (batch_size, input_dim)
        return d_input

    @property
    def trainable(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Returns the layer's trainable parameters and their gradients.

        Each element is a tuple (param, grad) representing a parameter array
        and its corresponding gradient, which are usedfor updates by an optimizer.
        """
        return [(self.W, self.dW), (self.b, self.db)]


a = Dense(2, 3)

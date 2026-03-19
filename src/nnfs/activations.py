import numpy as np

from .layers import BaseLayer


class BaseActivation(BaseLayer):
    """Template for activation layers.

    Activation functions do not need to update or expose their weights,
    since they do not have trainable parameters
    """

    @property
    def trainable(self):
        return None


def func_sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


class Sigmoid(BaseActivation):
    """Sigmoid activation layer.

    Applies the sigmoid function element-wise to the input during forward propagation,
    and computes its derivative during backpropagation.

    Attributes:
        output (np.ndarray): Stores the output from the forward pass for use in the backward pass.
    """

    def __init__(self):
        self.output: np.ndarray = np.zeros(1)

    def forward(self, X_input: np.ndarray) -> np.ndarray:
        """Performs the forward pass using the sigmoid activation.

        Args:
            X_input (np.ndarray): Input array to the layer.

        Returns:
            np.ndarray: Activated output after applying sigmoid.
        """
        output = func_sigmoid(X_input)
        self.output = output
        return output

    def backward(self, grad_next: np.ndarray) -> np.ndarray:
        """Performs the backward pass for the sigmoid layer.

        Computes the gradient of the loss with respect to the input,
        reusing the cached output from the forward pass.

        Args:
            grad_next (np.ndarray): Gradient from the next layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to this layer's input.
        """
        return grad_next * self.output * (1 - self.output)

    def name(self) -> str:
        """Returns the layer's name.

        The name can be arbitrary but it has to be unique for each of the layer types.

        It is used by the model to summarize layer architecture.
        """
        return "Sigmoid"

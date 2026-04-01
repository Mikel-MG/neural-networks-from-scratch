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
        layer_name (str): Short name for the layer type.
        index (int): Position of the layer within the full model (initializes at 0).
    """

    def __init__(self):
        self.output: np.ndarray = np.zeros(1)

        # attributes for layer navigation
        self.layer_name: str = "Sigmoid"
        self.index: int = 0

    def forward(self, X_input: np.ndarray) -> np.ndarray:
        """Performs the forward pass using the sigmoid activation.

        Args:
            X_input (np.ndarray): Input array to the layer.

        Returns:
            Activated output after applying sigmoid.
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
            Gradient of the loss with respect to this layer's input.
        """
        return grad_next * self.output * (1 - self.output)


def func_softmax(x: np.ndarray):
    # numerical stability trick -> max(logit) = 0
    z_stable = x - np.max(x, axis=1, keepdims=True)

    # compute softmax function
    exp_z = np.exp(z_stable)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return probs


class Softmax(BaseActivation):
    """Softmax activation layer.

    Applies the softmax function the input during forward propagation,
    and computes its derivative during backpropagation.

    Attributes:
        output (np.ndarray): Stores the output from the forward pass for use in the backward pass.
        layer_name (str): Short name for the layer type.
        index (int): Position of the layer within the full model (initializes at 0).
    """

    def __init__(self):
        self.output: np.ndarray = np.zeros(1)

        # attributes for layer navigation
        self.layer_name: str = "Softmax"
        self.index: int = 0

    def forward(self, X_input: np.ndarray) -> np.ndarray:
        """Performs the forward pass using the softmax activation.

        Args:
            X_input (np.ndarray): Input array to the layer.

        Returns:
            Activated output after applying softmax.
        """
        output = func_softmax(X_input)
        self.output = output
        return output

    def backward(self, grad_next: np.ndarray) -> np.ndarray:
        """Performs the backward pass for the softmax layer.

        Computes the gradient of the loss with respect to the input,
        reusing the cached output from the forward pass.

        Args:
            grad_next (np.ndarray): Gradient from the next layer.

        Returns:
            Gradient of the loss with respect to this layer's input.
        """
        # softmax probabilities
        s = self.output  # shape (batch_size, num_classes)

        # compute the dot product of s * grad_next for each sample
        s_dot_grad = np.sum(s * grad_next, axis=1, keepdims=True)

        # Jacobian-vector product: dx = s * (grad_next - s_dot_grad)
        dx = s * (grad_next - s_dot_grad)
        return dx

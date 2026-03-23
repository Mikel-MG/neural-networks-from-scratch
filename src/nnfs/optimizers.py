from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):
    """Template for optimizers.

    Optimizers determine how model parameters are updated given the computed gradients.
    """

    @abstractmethod
    def update(self, param_id: str, parameters: np.ndarray, gradients: np.ndarray):
        pass


class SGD(BaseOptimizer):
    """Stochastic Gradient Descent optimizer.

    It implements the following update mechanism:
    new_params = old_params - lr * (momentum * velocity + gradients).

    Args:
        lr (float): Learning rate, controls size of parameter update.
        momentum (float): Ratio of previous parameter change that is retained.

    Attributes:
        v (dict): Stores velocity of gradient change as `dict[param_id] = np.ndarray`.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.v: dict = {}

    def update(
        self, param_id: str, parameters: np.ndarray, gradients: np.ndarray
    ) -> None:
        """Updates layer parameters using precomputed gradients.

        Args:
            param_id (str): Layer.param identifier.
            parameters (np.ndarray): Model parameters.
            gradients (np.ndarray): dL / d_params gradients.
        """
        # initialize gradient velocity array
        if param_id not in self.v:
            self.v[param_id] = np.zeros_like(gradients)

        # update gradient velocity array
        self.v[param_id] = self.momentum * self.v[param_id] + gradients

        # parameters are updated in-place
        parameters -= self.lr * self.v[param_id]

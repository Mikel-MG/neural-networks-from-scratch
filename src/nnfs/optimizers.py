from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):
    """Template for optimizers.

    Optimizers determine how model parameters are updated given the computed gradients.
    """

    @abstractmethod
    def update(self, parameters: np.ndarray, gradients: np.ndarray):
        pass


class SGD(BaseOptimizer):
    """Stochastic Gradient Descent optimizer.

    It implements the following update mechanism:
    new_params = old_params - lr * gradients

    Args:
        lr (float): Learning rate.
    """

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, parameters: np.ndarray, gradients: np.ndarray) -> None:
        """Updates layer parameters using precomputed gradients.

        Args:
            parameters (np.ndarray): Model parameters.
            gradients (np.ndarray): dL / d_params gradients.
        """
        # parameters are updated in-place
        parameters -= self.lr * gradients

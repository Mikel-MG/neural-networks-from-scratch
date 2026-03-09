from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):
    """
    Base class of an optimizer
    """

    @abstractmethod
    def update(self, parameters: np.ndarray, gradients: np.ndarray):
        pass


class SGD(BaseOptimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def update(self, parameters: np.ndarray, gradients: np.ndarray) -> None:
        """
        Update layer parameters using precomputed gradients.
        """
        # parameters are updated in-place
        parameters -= self.lr * gradients

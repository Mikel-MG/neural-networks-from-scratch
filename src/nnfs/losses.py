from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """Template for loss functions.

    Loss functions are conceptually equivalent to layers,
    in that they implement forward and backward passes. However,
    only one loss object per model is allowed, which is directly
    provided to the model, instead of to the list of layers.

    Loss objects have no trainable parameters
    """

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        pass

    @abstractmethod
    def backward(self):
        pass


class MSE(Loss):
    """
    #TODO: Description
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Computes the forward pass of the loss function.

        The method stores `y_pred` and `y_true` as instance attributes for
        later use by the backward pass.

        Args:
            y_pred (np.ndarray): Predictions produced by the model.
            y_true (np.ndarray): Ground-truth values for the prediction task.

        Returns:
            The computed loss value.
        """
        self.y_pred = y_pred
        self.y_true = y_true
        diff = y_pred - y_true
        loss_value = np.mean(np.square(diff))
        return float(loss_value)

    def backward(self) -> np.ndarray:
        """
        Compute the dL_d_ypred gradient
        """
        diff = self.y_pred - self.y_true
        dL_d_ypred = 2 / diff.shape[0] * diff
        return dL_d_ypred


class BCE(Loss):
    """
    #TODO: add description
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Run forward pass and store y_pred and y_true
        """
        self.y_pred = y_pred
        self.y_true = y_true
        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss_value = np.mean(bce)
        return float(loss_value)

    def backward(self) -> np.ndarray:
        """
        Compute the dL_d_ypred gradient
        """
        y_pred = self.y_pred
        y_true = self.y_true
        dL_d_ypred = -y_true / y_pred + (1 - y_true) / (1 - y_pred)
        return dL_d_ypred

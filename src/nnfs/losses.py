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
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self) -> np.ndarray:
        pass


class MSE(Loss):
    r"""
    Mean Squared Error loss function

    $\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$
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
        """Computes the backward pass of the loss function.

        Requires the cached values stored by the forward() method.

        Returns:
            The computed dL_d_ypred gradients.
        """
        diff = self.y_pred - self.y_true
        dL_d_ypred = 2 / diff.shape[0] * diff
        return dL_d_ypred


class BCE(Loss):
    r"""
    Binary Cross Entropy loss function

    $\mathcal{L}_{\text{BCE}} = - \frac{1}{n} \sum_{i=1}^{n} \big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \big]$
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
        # clip predictions to avoid log(0)
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # cache predictions and targets for the backward pass
        self.y_pred = y_pred
        self.y_true = y_true

        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        loss_value = np.mean(bce)
        return float(loss_value)

    def backward(self) -> np.ndarray:
        """Computes the backward pass of the loss function.

        Requires the cached values stored by the forward() method.

        Returns:
            The computed dL_d_ypred gradients.
        """
        y_pred = self.y_pred
        y_true = self.y_true
        dL_d_ypred = -y_true / y_pred + (1 - y_true) / (1 - y_pred)
        return dL_d_ypred


class CCE(Loss):
    r"""
    Categorical Cross Entropy loss function

    $\mathcal{L}_{\text{CCE}} = - \frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$

    Note: CCE is typically used to compute the loss between a softmax activation layer and one-hot encoded labels.
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
        # clip predictions to avoid log(0)
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # cache predictions and targets for the backward pass
        self.y_pred = y_pred
        self.y_true = y_true

        # sum over classes
        cce = -np.sum(y_true * np.log(y_pred), axis=1)

        loss_value = np.mean(cce)
        return float(loss_value)

    def backward(self) -> np.ndarray:
        """Computes the backward pass of the loss function.

        Requires the cached values stored by the forward() method.

        Returns:
            The computed dL_d_ypred gradients.
        """
        # use cached predictions and targets
        y_pred = self.y_pred
        y_true = self.y_true

        n_samples = y_true.shape[0]
        dL_d_ypred = -y_true / (y_pred * n_samples)

        return dL_d_ypred

import math

import numpy as np


class BatchGenerator:
    """
    A simple batch generator.

    Args:
        X_data (np.ndarray): Array of shape=(samples, features).
        y_data (np.ndarray): Array of shape=(samples, pred_features).
        batch_size (int): Number of samples per batch of data.

        If batch_size == -1, a single batch is generated, containing all the samples.
    """

    def __init__(self, X_data: np.ndarray, y_data: np.ndarray, batch_size: int = 128):
        # check that each sample has a label or ground truth value
        assert len(X_data) == len(y_data)

        # initialize batch generator
        self.index = 0
        self.X_data = X_data
        self.y_data = y_data

        # account for using the entire dataset in a single batch
        if batch_size == -1:
            batch_size = len(X_data)

        self.batch_size = batch_size

        # compute number of batches
        self.num_batches = math.ceil(len(X_data) / batch_size)

    def next(self):
        """Generate data for the next batch."""

        X_data_batch = self.X_data[self.index : self.index + self.batch_size]
        y_data_batch = self.y_data[self.index : self.index + self.batch_size]

        # advance index
        self.index += self.batch_size

        return X_data_batch, y_data_batch


def shuffle(list_arrays: list[np.ndarray]):
    """Randomly shuffle each row in a list of arrays.

    Args:
        list_arrays (np.ndarray): List of arrays of equal number of rows.

    Returns:
        List of shuffled arrays.
    """
    # check that all arrays have the same number of samples
    N_samples = list_arrays[0].shape[0]
    for array in list_arrays:
        assert array.shape[0] == N_samples

    # generate a random permutation
    permutation = np.random.permutation(len(list_arrays[0]))

    # change every array according to the same permutation
    for i, array in enumerate(list_arrays):
        print(list_arrays[i].shape)
        list_arrays[i] = array[permutation]

    return list_arrays

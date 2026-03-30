import numpy as np
import pytest

from nnfs.utils import class_to_onehot, shuffle


def test_shuffle():
    X = np.arange(1000)
    Y = X[::-1]
    X, Y = shuffle([X, Y])

    Z = X + Y
    assert Z == pytest.approx(Z[0])


def test_onehot():
    y = np.array([0, 3, 2, 1, 3, 2, 1, 0, 2, 2, 0, 3, 1, 0, 3, 2])
    one_hot_y = class_to_onehot(y, 4)

    # check that expected size is correct
    assert one_hot_y.shape == pytest.approx([y.shape[0], 4])

    assert np.sum(one_hot_y, axis=1) == pytest.approx(np.ones(y.shape[0]))

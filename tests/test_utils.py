import numpy as np
import pytest

from nnfs.utils import shuffle


def test_shuffle():
    X = np.arange(1000)
    Y = X[::-1]
    X, Y = shuffle([X, Y])

    Z = X + Y
    assert Z == pytest.approx(Z[0])

import pytest

from nnfs.datasets.data_generators import (generate_concentric_circles,
                                           generate_two_moons)


def test_two_moons_generator():
    # generate data
    X, Y = generate_two_moons(500)

    # check that two moons generate N=500 data points per moon
    # and that X and Y are two-dimensional arrays
    assert X.shape == pytest.approx([1000, 2])
    assert Y.shape == pytest.approx([1000, 1])

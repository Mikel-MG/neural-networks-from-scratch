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


def test_concentric_circles_binary():
    # generate data
    X, Y = generate_concentric_circles(500, 3, binary=True)

    # check that the number of data points is as expected
    assert X.shape == pytest.approx([1500, 2])
    assert Y.shape == pytest.approx([1500, 1])

    # check that the number of labels is as expected
    assert len(set(Y.flatten())) == 2


def test_concentric_circles_nonbinary():
    # generate data
    X, Y = generate_concentric_circles(500, 4, binary=False)

    # check that the number of data points is as expected
    assert X.shape == pytest.approx([2000, 2])
    assert Y.shape == pytest.approx([2000, 1])

    # check that the number of labels is as expected
    assert len(set(Y.flatten())) == 4

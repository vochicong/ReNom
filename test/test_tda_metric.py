import numpy as np
from numpy.testing import assert_array_equal

import pytest

from renom.tda.metric import Distance


def test_cityblock():
    data = np.array([[0., 0.], [1., 1.]])

    m = Distance(metric="cityblock")
    dist_matrix = m.fit_transform(data)
    test_matrix = np.array([[0., 2.], [2., 0.]])

    assert_array_equal(dist_matrix, test_matrix)


def test_euclidean():
    data = np.array([[0., 0.], [1., 1.]])

    m = Distance(metric="euclidean")
    dist_matrix = m.fit_transform(data)
    test_matrix = np.array([[0., np.sqrt(2)], [np.sqrt(2), 0.]])

    assert_array_equal(dist_matrix, test_matrix)


def test_cosine():
    data = np.array([[1., 1.], [1., 2.]])

    m = Distance(metric="cosine")
    dist_matrix = m.fit_transform(data)

    dist = 3 / (np.sqrt(2) * np.sqrt(5))
    test_matrix = np.array([[0., 1 - dist], [1 - dist, 0.]])

    assert_array_equal(dist_matrix, test_matrix)


def test_hamming():
    data = np.array([[0., 0.], [0., 2.]])

    m = Distance(metric="hamming")
    dist_matrix = m.fit_transform(data)
    test_matrix = np.array([[0., 0.5], [0.5, 0.]])

    assert_array_equal(dist_matrix, test_matrix)


def test_none_input():
    data = None
    m = Distance()

    with pytest.raises(Exception):
        m.fit_transform(data)


def test_none_metric():
    data = np.array([[0., 0.], [0., 1.]])
    m = Distance(metric=None)

    with pytest.raises(Exception):
        m.fit_transform(data)


def test_unusable_metric():
    data = np.array([[0., 0.], [0., 1.]])
    m = Distance(metric="somemetric")

    with pytest.raises(Exception):
        m.fit_transform(data)

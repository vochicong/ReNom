import numpy as np
from numpy.testing import assert_array_equal

import pytest

from renom.tda.lens import L1Centrality, LinfCentrality, GaussianDensity, PCA, TSNE


def test_l1():
    dist_matrix = np.array([[0., 1., 1.], [1., 0., 2.], [1., 2., 0.]])

    lens = L1Centrality()
    projected_data = lens.fit_transform(dist_matrix)
    test_data = np.array([2., 3., 3.])
    test_data = test_data.reshape(test_data.shape[0], 1)

    assert_array_equal(projected_data, test_data)


def test_linf():
    dist_matrix = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])

    lens = LinfCentrality()
    projected_data = lens.fit_transform(dist_matrix)
    test_data = np.array([2., 3., 3.])
    test_data = test_data.reshape(test_data.shape[0], 1)

    assert_array_equal(projected_data, test_data)


def test_gd():
    dist_matrix = np.array([[0., 1.], [1., 0.]])

    lens = GaussianDensity(h=0.5)
    projected_data = lens.fit_transform(dist_matrix)
    test_data = np.array([np.exp(0) + np.exp(-1), np.exp(-1) + np.exp(0)])
    test_data = test_data.reshape(test_data.shape[0], 1)

    assert_array_equal(projected_data, test_data)


def test_pca():
    dist_matrix = np.array([[0., 1.], [1., 0.]])

    lens = PCA(components=[0])
    projected_data = lens.fit_transform(dist_matrix)
    projected_data = projected_data / (np.abs(projected_data))
    test_data = np.array([-1., 1.])
    test_data = test_data.reshape(test_data.shape[0], 1)

    assert_array_equal(projected_data, test_data)


def test_tsne():
    dist_matrix = np.array([[0., 1.], [1., 0.]])

    lens = TSNE(components=[0])
    projected_data = lens.fit_transform(dist_matrix)
    print(projected_data)
    projected_data = projected_data / (np.abs(projected_data))
    test_data = np.array([-1., 1.])
    test_data = test_data.reshape(test_data.shape[0], 1)

    assert_array_equal(projected_data, test_data)


def test_l1_none_input():
    dist_matrix = None
    lens = L1Centrality()

    with pytest.raises(Exception):
        lens.fit_transform(dist_matrix)


def test_linf_none_input():
    dist_matrix = None
    lens = LinfCentrality()

    with pytest.raises(Exception):
        lens.fit_transform(dist_matrix)


def test_gd_none_input():
    dist_matrix = None
    lens = GaussianDensity()

    with pytest.raises(Exception):
        lens.fit_transform(dist_matrix)


def test_gd_h0():
    with pytest.raises(Exception):
        GaussianDensity(h=0)


def test_pca_none_input():
    dist_matrix = None
    lens = PCA(components=[0])
    with pytest.raises(Exception):
        lens.fit_transform(dist_matrix)


def test_pca_none_components():
    with pytest.raises(Exception):
        PCA(components=None)


def test_tsne_none_input():
    dist_matrix = None
    lens = TSNE(components=[0])
    with pytest.raises(Exception):
        lens.fit_transform(dist_matrix)


def test_tsne_none_components():
    with pytest.raises(Exception):
        TSNE(components=None)

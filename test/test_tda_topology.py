import numpy as np
from numpy.testing import assert_array_equal

import pytest

from sklearn.cluster import DBSCAN

from renom.tda.lens import PCA, L1Centrality, GaussianDensity
from renom.tda.metric import Distance
from renom.tda.topology import Topology, SearchableTopology


def test_transform_none_none():
    data = np.array([[0., 0.], [1., 1.]])

    t = Topology()

    metric = None
    lens = None
    t.fit_transform(data, metric=metric, lens=lens)

    test_data = np.array([[0., 0.], [1., 1.]])

    assert_array_equal(t.pointcloud, test_data)


def test_transform_none_pca():
    data = np.array([[0., 1.], [1., 0.]])

    t = Topology()

    metric = None
    lens = [PCA(components=[0])]
    t.fit_transform(data, metric=metric, lens=lens)

    test_data = np.array([0., 1.])
    test_data = test_data.reshape(test_data.shape[0], 1)

    assert_array_equal(t.pointcloud, test_data)


def test_transform_multi_lens():
    data = np.array([[0., 0.], [0., 1.], [1., 1.]])

    t = Topology()

    metric = Distance(metric="hamming")
    lens = [L1Centrality(), GaussianDensity(h=0.25)]
    t.fit_transform(data, metric=metric, lens=lens)

    test_data = np.array([[1., 0.], [0., 1.], [1., 0.]])

    assert_array_equal(t.pointcloud, test_data)


def test_map():
    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)

    clusterer = DBSCAN(eps=0.4, min_samples=3)
    t.map(resolution=2, overlap=0.3, clusterer=clusterer)

    test_nodes = np.array([[0.25, 0.25],
                           [0.25, 0.75],
                           [0.75, 0.25],
                           [0.75, 0.75]])

    test_edges = np.array([[0, 1],
                           [0, 2],
                           [0, 3],
                           [1, 2],
                           [1, 3],
                           [2, 3]])

    assert_array_equal(t.nodes, test_nodes)
    assert_array_equal(t.edges, test_edges)


def test_color_categorical_rgb():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)

    clusterer = DBSCAN(eps=0.2, min_samples=3)
    t.map(resolution=2, overlap=0.3, clusterer=clusterer)

    t.color(target, dtype="categorical", ctype="rgb", normalized=False)

    test_color = ['#0000b2', '#00b200', '#b2b200']

    assert t.colorlist == test_color


def test_color_numerical_rgb():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1.1], [0.9],
                       [2], [2], [2]])

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)

    clusterer = DBSCAN(eps=0.2, min_samples=3)
    t.map(resolution=2, overlap=0.3, clusterer=clusterer)

    t.color(target, dtype="numerical", ctype="rgb", normalized=False)

    test_color = ['#0000b2', '#00b200', '#b2b200']

    assert t.colorlist == test_color


def test_color_categorical_gray():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)

    clusterer = DBSCAN(eps=0.2, min_samples=3)
    t.map(resolution=2, overlap=0.3, clusterer=clusterer)

    t.color(target, dtype="categorical", ctype="gray", normalized=False)

    test_color = ['#dcdcdc', '#787878', '#464646']

    assert t.colorlist == test_color


def test_color_numerical_gray():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1.1], [0.9],
                       [2], [2], [2]])

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)

    clusterer = DBSCAN(eps=0.2, min_samples=3)
    t.map(resolution=2, overlap=0.3, clusterer=clusterer)

    t.color(target, dtype="numerical", ctype="gray", normalized=False)

    test_color = ['#dcdcdc', '#787878', '#464646']

    assert t.colorlist == test_color


def test_transform_none_input():
    data = None
    t = Topology()
    with pytest.raises(Exception):
        t.fit_transform(data)


def test_transform_1darray_input():
    data = np.array([])
    t = Topology()
    with pytest.raises(ValueError):
        t.fit_transform(data)


def test_transform_3darray_input():
    data = np.array([[[]]])
    t = Topology()
    with pytest.raises(ValueError):
        t.fit_transform(data)


def test_map_none_input():
    t = Topology()
    t.pointcloud = None
    with pytest.raises(Exception):
        t.map()


def test_map_1darray_input():
    t = Topology()
    t.pointcloud = np.array([])
    with pytest.raises(ValueError):
        t.map()


def test_map_3darray_input():
    t = Topology()
    t.pointcloud = np.array([[[]]])
    with pytest.raises(ValueError):
        t.map()


def test_map_resolution_value_error():
    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)

    with pytest.raises(ValueError):
        t.map(resolution=0, overlap=0.3, clusterer=None)


def test_map_overlap_value_error():
    data = np.array([[0., 0.],
                     [0.25, 0.25],
                     [0.5, 0.5],
                     [0.75, 0.75],
                     [1., 1.],
                     [1., 0.],
                     [0.25, 0.75],
                     [0.75, 0.25],
                     [0., 1.]])

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)

    with pytest.raises(ValueError):
        t.map(resolution=10, overlap=-0.1, clusterer=None)


def test_color_none_input():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = None

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)
    t.map(resolution=2, overlap=0.3, clusterer=None)

    with pytest.raises(Exception):
        t.color(target, dtype="categorical", ctype="rgb", normalized=False)


def test_color_different_size_input():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([0, 1, 2])

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)
    t.map(resolution=2, overlap=0.3, clusterer=None)

    with pytest.raises(ValueError):
        t.color(target, dtype="categorical", ctype="rgb", normalized=False)


def test_color_dtype():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)
    t.map(resolution=2, overlap=0.3, clusterer=None)

    with pytest.raises(Exception):
        t.color(target, dtype="somecategory", ctype="rgb", normalized=False)


def test_color_ctype():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    t = Topology()
    t.fit_transform(data, metric=None, lens=None)
    t.map(resolution=2, overlap=0.3, clusterer=None)

    with pytest.raises(Exception):
        t.color(target, dtype="categorical", ctype="somecolortype", normalized=False)


def test_regist_categorical_data():
    category = np.array([["a"], ["a"], ["a"],
                         ["b"], ["b"], ["b"],
                         ["c"], ["c"], ["c"]])

    t = SearchableTopology()
    t.regist_categorical_data(category)

    assert_array_equal(t.categorical_data, category)


def test_search():
    data = np.array([[0., 0.],
                     [0.1, 0.1],
                     [0.2, 0.2],
                     [0.2, 0.8],
                     [0.1, 0.9],
                     [0., 1.],
                     [0.8, 0.8],
                     [0.9, 0.9],
                     [1., 1.]])

    target = np.array([[0], [0], [0],
                       [1], [1], [1],
                       [2], [2], [2]])

    category = np.array([["a"], ["a"], ["a"],
                         ["b"], ["b"], ["b"],
                         ["c"], ["c"], ["c"]])

    t = SearchableTopology()
    t.regist_categorical_data(category)
    t.fit_transform(data, metric=None, lens=None)
    t.map(resolution=2, overlap=0.3, clusterer=None)
    t.color(target, dtype="categorical", ctype="rgb", normalized=False)
    t.search("a")

    test_color = ['#0000b2', '#cccccc', '#cccccc']
    assert t.colorlist == test_color

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import pytest
from renom.cuda import use_cuda
from renom.core import Variable
import renom as rm
import test_utility


@test_utility.skipgpu
@pytest.mark.skipif(np.lib.NumpyVersion(np.__version__) < '1.13.0', reason='na')
def test_gpu_node_neg():
    with use_cuda():
        g1 = Variable(np.array([1., 2.]))
        g2 = -g1
        assert np.allclose(g2, [-1, -2])
        assert not np.allclose(g2, [-1, -3])

        g3 = -g1 * 2
        assert np.allclose(g3, [-2, -4])
        assert not np.allclose(g3, [-3, -4])

def test_walk():
    g1 = Variable(np.array([1., 2.]))
    g2 = Variable(np.array([1., 2.]))
    g3 = Variable(np.array([1., 2.]))
    g4 = g1 + g2
    g5 = g3 + g4
    g6 = g1+g2+g3+g4+g5
    g7 = g6+g5

    nodes = {id(g) for g in (g1, g2, g3, g4, g5, g6, g7)}
    seen = set()
    for c in g6.walk():
        print(repr(c))
        if id(c) in nodes:
            nodes.remove(id(c))

    assert nodes == {id(g7)}


def test_grad():
    g1 = Variable(np.array([1., 2.]))
    g2 = Variable(np.array([1., 2.]))
    g3 = Variable(np.array([1., 2.]))
    g4 = g1 + g2
    g5 = g3 + g4
    g6 = g1+g2+g3+g4+g5
    g7 = g6+g5

    g = g6.grad(np.array([1., 2.]))
    print(g._refcounts )
    assert g._refcounts[id(g1)] == 2
    assert g._refcounts[id(g2)] == 2
    assert g._refcounts[id(g3)] == 2
    assert g._refcounts[id(g4)] == 2
    assert g._refcounts[id(g5)] == 1
    assert g._refcounts[id(g6)] == 1


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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from renom.core import Variable, DEBUG_NODE_STAT, DEBUG_GRAPH_INIT, DEBUG_NODE_GRAPH
from renom import operation as O
import renom as R


def test_node_dump():
    DEBUG_GRAPH_INIT(True)

    a = Variable(np.array([1, 2, 3, 4, 5]))
    b = Variable(np.array([1, 2, 3, 4, 5]))
    c = a + b  # NOQA

    d = a + b * 2  # NOQA

    DEBUG_NODE_STAT()
    # DEBUG_NODE_GRAPH()


def test_node_clear():
    DEBUG_GRAPH_INIT(True)

    a = Variable(np.random.rand(2, 2).astype(np.float32))
    b = Variable(np.random.rand(2, 2).astype(np.float32))

    layer = R.Lstm(2)

    c = layer(O.dot(a, b))  # NOQA

    DEBUG_NODE_STAT()
#    DEBUG_NODE_GRAPH()


# test_node_clear()

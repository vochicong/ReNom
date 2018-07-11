#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from renom.core import Node, GPUValue, get_gpu
from renom.layers.function.utils import im2col, col2im, transpose_out_size, tuplize
from renom.layers.function.pool2d import max_pool2d, average_pool2d
from renom.cuda import cuda as cu
from .parameterized import Parametrized
from renom.config import precision

# A simple python object designed to hide the previous pool
# from ReNom


class SimpleContainer(object):
    def __init__(self, item):
        self._item = item


class max_unpool2d(Node):

    def __new__(cls, x, prev_pool):
        return cls.calc_value(x, prev_pool._item)

    @classmethod
    def _oper_cpu(cls, x, prev_pool):
        N = len(x)
        col = np.zeros((N, prev_pool.attrs._in_shape[0], prev_pool.attrs._kernel[0],
                        prev_pool.attrs._kernel[1], prev_pool.attrs._out_shape[1], prev_pool.attrs._out_shape[2]))
        col_k = np.rollaxis(col.reshape(
            N, prev_pool.attrs._in_shape[0], -1, prev_pool.attrs._out_shape[1], prev_pool.attrs._out_shape[2]), 2)
        for i in np.ndindex(N, prev_pool.attrs._in_shape[0],
                            prev_pool.attrs._out_shape[1], prev_pool.attrs._out_shape[2]):
            col_k[prev_pool.attrs._index[i]][i] = x[i]
        ret = col2im(col, prev_pool.attrs._in_shape[1:],
                     prev_pool.attrs._stride, prev_pool.attrs._padding)
        ret = cls._create_node(ret)
        ret.attrs._x = x
        return ret

    @classmethod
    def _oper_gpu(cls, x, prev_pool):
        dx = GPUValue(shape=prev_pool.attrs._x.shape)
        with cu.cudnn_handler() as handle:
            cu.cuPoolingBackward(handle, prev_pool.attrs._pool_desc, prev_pool.attrs._x, x, x, dx)
        ret = cls._create_node(dx)
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        self.attrs._x._update_diff(context, dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        self.attrs._x._update_diff(context, dy, **kwargs)


class average_unpool2d(Node):
    def __new__(cls, x, prev_pool):
        return cls.calc_value(x, prev_pool._item)

    @classmethod
    def _oper_cpu(cls, x, prev_pool):
        N = len(x)
        col = np.zeros((N, prev_pool.attrs._in_shape[0], prev_pool.attrs._kernel[0],
                        prev_pool.attrs._kernel[1], prev_pool.attrs._out_shape[1], prev_pool.attrs._out_shape[2]))
        col_k = np.rollaxis(col.reshape(
            N, prev_pool.attrs._in_shape[0], -1, prev_pool.attrs._out_shape[1], prev_pool.attrs._out_shape[2]), 2)
        col_k[:] = x / float(len(col_k))
        dx = col2im(col, prev_pool.attrs._in_shape[1:],
                    prev_pool.attrs._stride, prev_pool.attrs._padding)
        ret = cls._create_node(dx)
        ret.attrs._x = x
        return ret

    @classmethod
    def _oper_gpu(cls, x, prev_pool):
        dx = GPUValue(shape=prev_pool.attrs._x.shape)
        with cu.cudnn_handler() as handle:
            cu.cuPoolingBackward(handle, prev_pool.attrs._pool_desc, prev_pool.attrs._x, x, x, dx)
        ret = cls._create_node(dx)
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        dy = np.ones_like(self.attrs._x)
        self.attrs._x._update_diff(context, dy)

    def _backward_gpu(self, context, dy, **kwargs):
        self.attrs._x._update_diff(context, dy)


class MaxUnPool2d:
    '''Max unpooling function.
    Unpools an input in a network where a previous pooling has occured.
    Note the input shape requirement:
        x.shape == previous_pool.shape

    The output shape will be:
        ret.shape == previous_pool.input.shape

    Args:
        x (Node, np.ndarray):           The input to the unpooling method
        prev_pool (max_pool2d, None):   The previous pool to be unpooled. In the case of none,
                                        the model searches through the history for the previous layer.

    '''

    def __init__(self):
        pass

    def __call__(self, x, prev_pool=None):
        p = x
        if prev_pool:
            assert isinstance(prev_pool, max_pool2d)
        while prev_pool is None:
            if isinstance(p, max_pool2d) and p.shape == x.shape:
                prev_pool = p
            else:
                try:
                    p = p.attrs._x
                except AttributeError:
                    raise Exception("Could not find previous 2D max pool")
        # Currently, ReNom expects all Node arguments as input to other nodes
        # to be called later with _update_diff at least once per use.
        # We hide the 2d pool from ReNom by inserting it into the SimpleContainer
        # class, where ReNom does not look for Node inputs.
        return self.forward(x, SimpleContainer(prev_pool))

    def forward(self, x, prev_pool):
        return max_unpool2d(x, prev_pool)


class AverageUnPool2d:
    '''Average unpooling function.
    Unpools an input in a network where a previous pooling has occured.
    Note the input shape requirement:
        x.shape == previous_pool.shape

    The output shape will be:
        ret.shape == previous_pool.input.shape

    Args:
        x (Node, np.ndarray):           The input to the unpooling method
        prev_pool (average_pool2d, None):   The previous pool to be unpooled. In the case of none,
                                        the model searches through the history for the previous layer.

    '''

    def __init__(self):
        pass

    def __call__(self, x, prev_pool=None):
        p = x
        if prev_pool:
            assert isinstance(prev_pool, average_pool2d)
        while prev_pool is None:
            if isinstance(p, average_pool2d) and p.shape == x.shape:
                prev_pool = p
            else:
                try:
                    p = p.attrs._x
                except AttributeError:
                    raise Exception("Could not find previous 2D average pool")
        return self.forward(x, SimpleContainer(prev_pool))

    def forward(self, x, prev_pool):
        return average_unpool2d(x, prev_pool)

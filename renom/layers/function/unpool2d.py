#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from renom.core import Node, GPUValue, get_gpu
from renom.layers.function.utils import im2col, col2im, transpose_out_size, tuplize
from renom.cuda import cuda as cu
from .parameterized import Parametrized
from renom.config import precision


class max_unpool2d(Node):

    def __new__(cls, x, filter, stride, padding):
        filter, stride, padding = (tuplize(x) for x in (filter, stride, padding))
        in_shape = x.shape[1:]
        out_shape = [x.shape[1], ]
        out_shape.extend(transpose_out_size(in_shape[1:], filter, stride, padding))
        return cls.calc_value(x, in_shape, out_shape, filter, stride, padding)

    @classmethod
    def _oper_cpu(cls, x, in_shape, out_shape, kernel, stride, padding):
        N = len(x)
        col = np.zeros((N, x.attrs._in_shape[0], x.attrs._kernel[0],
                        x.attrs._kernel[1], x.attrs._out_shape[1], x.attrs._out_shape[2]))
        col_k = np.rollaxis(col.reshape(
            N, x.attrs._in_shape[0], -1, x.attrs._out_shape[1], x.attrs._out_shape[2]), 2)
        for i in np.ndindex(N, x.attrs._in_shape[0], x.attrs._out_shape[1], x.attrs._out_shape[2]):
            col_k[x.attrs._index[i]][i] = x[i]
        ret = col2im(col, x.attrs._in_shape[1:], x.attrs._stride, x.attrs._padding)
        ret = cls._create_node(ret)
        ret.attrs._x = x
        ret.attrs._index = x.attrs._index
        return ret



    @classmethod
    def _oper_gpu(cls, x, in_shape, out_shape, karnel, stride, padding):
        N = len(x)
        dx = GPUValue(shape=tuple([N,] + list(out_shape)))
        dy = get_gpu(x).ones_like_me()
        with cu.cudnn_handler() as handle:
            cu.cuPoolingBackward(handle, x.attrs._pool_desc, x.attrs._x, x, x, dx)
        ret = cls._create_node(dx)
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, dt=None):
        self.attrs._x._update_diff(context, dy)

    def _backward_gpu(self, context, dy, dt=None):
        self.attrs._x._update_diff(context, dy)


class avg_unpool2d(Node):
    pass


class MaxUnPool2d:
    def __init__(self, filter=(3, 3),
                 padding=(0, 0), stride=(1, 1)):
        self._padding = padding
        self._stride = stride
        self._kernel = filter

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, index=None):
        #size_i = x.shape[1:]
        #size_o = [size_i[0], ]
        #size_o.extend(transpose_out_size(size_i[1:], self._kernel, self._stride, self._padding))
        return max_unpool2d(x, self._kernel,
                            self._stride, self._padding)

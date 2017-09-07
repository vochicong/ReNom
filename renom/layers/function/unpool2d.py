#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from renom.core import Node, GPUValue, get_gpu
from renom.layers.function.utils import im2col, col2im, transpose_out_size, tuplize
from renom.cuda import cuda as cu
from .parameterized import Parametrized
from renom.config import precision


class max_unpool2d(Node):

    def __new__(cls, x, index, filter, stride, padding):
        filter, stride, padding = (tuplize(x) for x in (filter, stride, padding))
        in_shape = x.shape[1:]
        out_shape = [x.shape[1], ]
        out_shape.extend(transpose_out_size(in_shape[1:], filter, stride, padding))
        return cls.calc_value(x, index, in_shape, out_shape, filter, stride, padding)

    @classmethod
    def _oper_cpu(cls, x, index, in_shape, out_shape, kernel, stride, padding):
        pass

    @classmethod
    def _oper_gpu(cls, x, index, in_shape, out_shape, karnel, stride, padding):
        pass

    def _backward_cpu(self, context, dy, dt=None):
        pass

    def _backward_gpu(self, context, dy, dt=None):
        pass


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
        size_i = x.shape[1:]
        size_o = [size_i[0], ]
        size_o.extend(transpose_out_size(size_i[1:], self._kernel, self._stride, self._padding))
        return max_unpool2d(x, index, size_i, size_o, self._kernel,
                            self._stride, self._padding)

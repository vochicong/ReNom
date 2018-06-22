#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from renom.core import Node, GPUValue, get_gpu
from renom.layers.function.utils import poolnim
from renom.cuda import cuda as cu
from .parameterized import Parametrized
from renom.config import precision

class SimpleContainer(object):
    def __init__(self, item):
        self._item = item

class max_unpoolnd(Node):

    def __new__(cls, x, prev_pool):
        return cls.calc_value(x, prev_pool._item)

    @classmethod
    def _oper_cpu(cls, x, prev_pool):
        result = poolnim(prev_pool.attrs._x, x, prev_pool.attrs._kernel, prev_pool.attrs._stride, mode = "max")
        ret = cls._create_node(result)
        ret.attrs._x = x
        ret.attrs._prev = prev_pool
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


class avg_unpool2d(Node):
    pass


class MaxUnPoolNd:
    def __init__(self):
        pass

    def __call__(self, x, prev_pool):
        return self.forward(x, prev_pool)

    def forward(self, x, prev_pool):
        return max_unpoolnd(x, SimpleContainer(prev_pool))

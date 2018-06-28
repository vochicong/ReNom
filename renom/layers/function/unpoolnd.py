#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from renom.core import Node, GPUValue, get_gpu
from renom.layers.function.utils import poolnim
from renom.cuda import cuda as cu
from .parameterized import Parametrized
from renom.layers.function.poolnd import max_poolnd
from renom.config import precision
from renom.cuda import set_cuda_active

class SimpleContainer(object):
    def __init__(self, item):
        self._item = item

class max_unpoolnd(Node):

    def __new__(cls, x, kernel, stride, padding):
        return cls.calc_value(x, kernel, stride, padding)

    @classmethod
    def _oper_cpu(cls, x, prev_pool):
        result = poolnim(prev_pool.attrs._x, x, prev_pool.attrs._kernel, prev_pool.attrs._stride, mode = "max")
        ret = cls._create_node(result)
        ret.attrs._x = x
        ret.attrs._prev = prev_pool
        return ret

    @classmethod
    def _oper_gpu(cls, x, kernel, stride, padding):
        back_shape = (np.array(x.shape[2:]) - 1) * np.array(stride) + np.array(kernel) - 2 * np.array(padding)
        back_shape = [x.shape[0], x.shape[1],] + (list(back_shape))
        dx = GPUValue(shape=back_shape)
        prev_np = np.empty(back_shape)
        x.to_cpu()
        prev_np[...,::2,::2,::2] = x
        prev_x = get_gpu(prev_np)
        prev_y = get_gpu(x).ones_like_me()
        prev_y = prev_y * get_gpu(x)
        pool_desc = cu.PoolingNDescriptor(kernel, padding, stride, pool_mode=0)

        with cu.cudnn_handler() as handle:
            cu.cuPoolingBackward(handle, pool_desc, prev_x, x, prev_y, dx)
        ret = cls._create_node(dx)
        ret.attrs._x = x
        ret.attrs._pool_desc = pool_desc

        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        self.attrs._x._update_diff(context, dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        pool_desc = self.attrs._pool_desc
        y = GPUValue(shape=self.attrs._x.shape)
        with cu.cudnn_handler() as handle:
            cu.cuPoolingForward(handle, pool_desc, dy, y)
        self.attrs._x._update_diff(context, y, **kwargs)


class avg_unpool2d(Node):
    pass

def check_input(var, length):
    if isinstance(var, tuple):
        assert len(var) is length
        var = list(var)
    elif not isinstance(var, np.ndarray):
        var = np.array(
            tuple([var for _ in range(length)]), dtype=np.int32)
    elif not var.dtype == np.int32:
        var = var.astype(np.int32)
    assert len(var) is length
    return var

class MaxUnPoolNd:
    def __init__(self, kernel=3, padding=0, stride=1):
        self._padding = padding
        self._stride = stride
        self._kernel = kernel

    def __call__(self, x):
        dims = len(x.shape[2:])
        assert dims < 4, "GPU Version can only handle up to 3 dimensions"
        func = lambda var: check_input(var, dims)
        self._padding, self._stride, self._kernel = map(func, [self._padding, self._stride, self._kernel])
        return self.forward(x)

    def forward(self, x):
        return max_unpoolnd(x, self._kernel, self._stride, self._padding)

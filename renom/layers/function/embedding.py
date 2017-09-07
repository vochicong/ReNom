#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.core import Node, get_gpu, precision, GPUValue, Variable
from renom.layers.function.parameterized import Parametrized
from renom.utility.initializer import GlorotNormal
from renom.cuda import cuda as cu


class embedding(Node):

    def __new__(cls, x, w):
        assert x.shape[1] == 1
        return cls.calc_value(x, w)

    @classmethod
    def _oper_cpu(cls, x, w):
        index = x.as_ndarray().astype(np.int)[:, 0]
        value = w[index]
        ret = cls._create_node(value)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._index = index
        return ret

    @classmethod
    def _oper_gpu(cls, x, w):
        z = GPUValue(shape=(len(x), len(w[0])))
        cu.cuembedding_forward(get_gpu(x), get_gpu(w), z)
        ret = cls._create_node(z)
        ret.attrs._x = x
        ret.attrs._w = w
        return ret

    def _backward_cpu(self, context, dy, dt=None):
        if isinstance(self.attrs._w, Node):
            N = len(self.attrs._index)
            dx = np.zeros(self.attrs._w.shape, dtype=self.attrs._w.dtype)
            for i in range(N):
                dx[self.attrs._index[i]] += dy[i]
            self.attrs._w._update_diff(context, dx)

    def _backward_gpu(self, context, dy, dt=None):
        if isinstance(self.attrs._w, Node):
            dx = get_gpu(self.attrs._w).zeros_like_me()
            cu.cuembedding_backward(get_gpu(self.attrs._x), get_gpu(dy), dx)
            self.attrs._w._update_diff(context, dx)


class Embedding(Parametrized):

    def __init__(self, output_size, input_size=None, initializer=GlorotNormal()):
        self._output_size = output_size
        self._initializer = initializer
        super(Embedding, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        size_i = input_size[0] if isinstance(input_size, tuple) else input_size
        size_o = self._output_size
        self.params = {
            "w": Variable(self._initializer((size_i, size_o)), auto_update=True)}

    def forward(self, x):
        return embedding(x, self.params.w)

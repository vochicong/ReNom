#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from renom.core import UnaryOp, Node, get_gpu
from renom.cuda import cuda as cu


class softmax(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        maxes = np.max(arg, axis=1, keepdims=True)
        u = np.exp(arg - maxes)
        summed = np.sum(u, axis=1, keepdims=True)
        z = u / (summed + 1e-8)
        return z

    @classmethod
    def _oper_gpu(cls, arg):
        z = get_gpu(arg).empty_like_me()
        with cu.cudnn_handler() as handle:
            cu.cuSoftmaxForward(handle, arg, z, mode=1)
        return z

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = self * dy
            summed = dx - np.sum(dx, axis=1, keepdims=True)
            self.attrs._arg._update_diff(context, ((1.0 - self) * dy + summed) * self, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            with cu.cudnn_handler() as handle:
                dx = get_gpu(self).empty_like_me()
                cu.cuSoftmaxBackward(handle, get_gpu(self), get_gpu(dy), dx, mode=1)
            self.attrs._arg._update_diff(context, dx, **kwargs)


class Softmax:

    def __call__(self, x):
        return softmax(x)

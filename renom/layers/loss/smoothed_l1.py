#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import renom as rm
from renom.core import BinOp, Node, get_gpu, to_value
from renom.cuda import cuda as cu
from renom.operation import where

try:
    from renom.cuda.cublas import *
    from renom.cuda.cuda_base import *
    if precision == np.float32:
        from renom.cuda.thrust_float import *
    else:
        from renom.cuda.thrust_double import *
except ImportError:
    pass

class smoothed_l1(Node):
    def __new__(cls, lhs, rhs):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        return cls.calc_value(lhs, rhs)

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        d = lhs - rhs
        loss = np.sum(np.where(abs(d) < 1, 0.5*(d**2), abs(d)-0.5))
        ret = cls._create_node(loss)
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        ret.attrs._d = d
        return ret

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        d = lhs - rhs
        flag = abs(d) < 1
        loss = np.sum(flag*0.5*(d**2)+(1-flag)*(abs(d)-0.5))
        ret = cls._create_node(loss)
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        ret.attrs._d = d
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            mask = abs(self.attrs._d < 0.5)
            dx = np.where(mask, self.attrs._d, 0.5*np.sign(self.attrs._d))
            self.attrs._lhs._update_diff(context, dx * dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            mask = abs(self.attrs._d) < 0.5
            dx = rm.where(mask, self.attrs._d, 0.5*np.sign(self.attrs._d))
            self.attrs._lhs._update_diff(context, dx * dy, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            mask = abs(self.attrs._d) < 0.5
            dx = rm.where(mask, self.attrs._d, 0.5*np.sign(self.attrs._d))
            self.attrs._rhs._update_diff(context, dx * dy, **kwargs)


class SmoothedL1(object):
    def __call__(self, x, y):
        return smoothed_l1(x, y)

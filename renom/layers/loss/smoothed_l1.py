#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import renom as rm
from renom.core import BinOp, Node, get_gpu, to_value
from renom.cuda import cuda as cu
from renom.operation import where


class smoothed_l1(Node):
    def __new__(cls, lhs, rhs):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        return cls.calc_value(lhs, rhs)

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        N = float(lhs.shape[0])
        d = lhs - rhs
        abs_d = abs(d)
        loss = np.sum(np.where(abs_d < 1, 0.5*d*d, abs_d-0.5))
        ret = cls._create_node(loss/N)
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        ret.attrs._d = d
        return ret

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        N = float(lhs.shape[0])
        d = lhs - rhs
        abs_d = abs(d.as_ndarray())
        flag = abs_d < 1
        loss = cu.cusum(get_gpu(flag*0.5*(d*d)+(1-flag)*(abs_d-0.5)))
        ret = cls._create_node(loss/N)
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        ret.attrs._d = d
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            N = float(self.attrs._lhs.shape[0])
            mask = abs(self.attrs._d) < 1.0
            dx = np.where(mask, self.attrs._d, np.sign(self.attrs._d))
            self.attrs._lhs._update_diff(context, dx*dy/N, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            N = float(self.attrs._lhs.shape[0])
            mask = abs(self.attrs._d) <= 1.0
            sign = (self.attrs._d > 0)*2 - 1
            dx = mask*self.attrs._d + (1 - mask)*sign
            self.attrs._lhs._update_diff(context, dx*dy/N, **kwargs)


class SmoothedL1(object):
    def __call__(self, x, y):
        return smoothed_l1(x, y)

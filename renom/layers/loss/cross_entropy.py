#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from renom.core import Node, get_gpu, to_value
from renom.cuda import cuda as cu
from renom.operation import log


class cross_entropy(Node):

    def __new__(cls, lhs, rhs):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        return cls.calc_value(lhs, rhs)

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        log_lhs = np.log(lhs + 1e-8)
        ret = cls._create_node(-np.sum(rhs * log_lhs))
        ret.attrs._log_lhs = log_lhs
        ret.attrs._rhs = rhs
        ret.attrs._lhs = lhs
        return ret

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        log_lhs = log(lhs + 1e-8)
        ret = cls._create_node(-cu.cusum(get_gpu(log_lhs * rhs)))
        ret.attrs._log_lhs = log_lhs
        ret.attrs._rhs = rhs
        ret.attrs._lhs = lhs
        return ret

    def _backward_cpu(self, context, dy):
        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, -dy * self.attrs._log_lhs)

        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, -dy * self.attrs._rhs / self.attrs._lhs)

    def _backward_gpu(self, context, dy):
        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, -dy * self.attrs._log_lhs)

        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, -dy * self.attrs._rhs / self.attrs._lhs)


class CrossEntropy:

    def __call__(self, lhs, rhs):
        return cross_entropy(lhs, rhs)

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

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, -dy * self.attrs._log_lhs, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, -dy * self.attrs._rhs / self.attrs._lhs, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, -dy * self.attrs._log_lhs, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, -dy * self.attrs._rhs / self.attrs._lhs, **kwargs)


class CrossEntropy:

    """This function evaluates the cross entropy loss
    between the target ``y`` and the input ``x``.

    .. math::
        E(x) = \sum_{n}^{N}\sum_{k}^{K}(-y*ln(x+\epsilon))

    :math:`N` is batch size.
    :math:`\epsilon` is small number for avoiding division by zero.

    Args:
        x (ndarray,Node): Input array.
        y (ndarray,Node): Target array.

    Raises:
        AssertionError: An assertion error will be raised if the given tensor dimension is less than 2.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>>
        >>> x = np.array([[1.0, 0.5]])
        >>> y = np.array([[0.0, 1.0]])
        >>> print(x.shape, y.shape)
        ((1, 2), (1, 2))
        >>> loss = rm.cross_entropy(x, y)
        >>> print(loss)
        cross_entropy(0.6931471824645996)

    """

    def __call__(self, lhs, rhs):
        return cross_entropy(lhs, rhs)

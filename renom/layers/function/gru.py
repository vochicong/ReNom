#!/usr/bin/env python
# -*- coding: utf-8 -*-


class gru(Node)

    @classmethod
    def _oper_cpu(cls, x, args):
        ret = None
        return ret

    @classmethod
    def _oper_gpu(cls, x, args):
        ret = None
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            dx = 1 * dy
            self.attrs._x._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            dx = 1 * dy
            self.attrs._x._update_diff(context, dx, **kwargs)

class Gru(Parametrized):

    def __init__(self, output_size, input_size=None):
        self._size_o = output_size
        super(Gru, self).__init__(input_size)

    def weight_initiallize(self, size_i):
        size_i = size_i[0]
        size_o = self._size_o
        bias = np.zeros((1, size_o ))

from __future__ import division, print_function
import numpy as np
from renom.core import Node, Variable, to_value, get_gpu, GPUValue, precision
from renom.cuda import cuda as cu
from renom.layers.function.parameterized import Parametrized
import renom as rm
import numpy as np

class l2_norm(Node):
    def __new__(cls, x, w):
        return cls.calc_value(x, w)

    @classmethod
    def _oper_cpu(cls, x, w):
        n = x.shape[0]
        norm = np.sqrt(np.sum(x*x, axis=1, keepdims=True)) + 1e-7
        z = (x/norm) * w
        ret = cls._create_node(z)
        ret.attrs._norm = norm
        ret.attrs._x = x
        ret.attrs._w = w
        return ret

    @classmethod
    def _oper_gpu(cls, x, w):
        n = x.shape[0]
        norm = rm.sqrt(rm.sum(x*x, axis=1, keepdims=True)) + 1e-7
        z = (x/norm) * w
        ret = cls._create_node(get_gpu(z))
        ret.attrs._norm = norm
        ret.attrs._x = x
        ret.attrs._w = w
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        norm  = self.attrs._norm
        if isinstance(self.attrs._x, Node):
            dx = dy * norm - (np.sum(self.attrs._x * dy, axis=1, keepdims=True) * self.attrs._x) / norm
            dx = dx / (norm * norm)
            self.attrs._x._update_diff(context, dx * self.attrs._w, **kwargs)
        if isinstance(self.attrs._w, Node):
            dl = dy * (self.attrs._x / norm)
            self.attrs._w._update_diff(context, np.sum(dl, axis=(0, 2, 3), keepdims=True), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        norm  = self.attrs._norm
        if isinstance(self.attrs._x, Node):
            dx = dy * norm - (rm.sum(self.attrs._x * dy, axis=1, keepdims=True) * self.attrs._x) / norm
            dx = dx / (norm * norm)
            self.attrs._x._update_diff(context, get_gpu(dx*self.attrs._w), **kwargs)
        if isinstance(self.attrs._w, Node):
            dl = dy * (self.attrs._x / norm)
            self.attrs._w._update_diff(context, get_gpu(rm.sum(dl.as_ndarray(), axis=(0, 2, 3), keepdims=True)), **kwargs)

class L2Norm(Parametrized):
    def __init__(self, scale, input_size=None):
        self.scale = scale
        super(L2Norm, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        self.params = { 'w': Variable(np.ones((input_size[0], 1, 1))*self.scale, auto_update=True)}

    def forward(self, x):
        ret = l2_norm(x, self.params['w'])
        return ret



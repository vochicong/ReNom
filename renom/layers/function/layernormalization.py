
import numpy as np
from renom.cuda import cuda as cu
from renom.core import Node, Variable
from .parameterized import Parametrized

class layernorm(Node):
    def __new__(cls, x, gain):
        return cls.calc_value(x, gain)

    @classmethod
    def _oper_cpu(cls, x, gain):
        assert len(x.shape) is 2, "Currently only normalizes for dense networks."
        H = x.shape[1]
        sum = np.sum(x,axis=1,keepdims=True)
        mu = sum / H
        v1 = np.power(x - mu, 2)
        sigma = np.sqrt(np.sum(v1, axis=1,keepdims=True)/H)
        std_dev_x = gain / sigma * (x - mu)
        ret = cls._create_node(std_dev_x)
        ret.attrs._x = x
        ret.attrs._gain = gain
        ret.attrs._sigma = sigma
        ret.attrs._mu = mu
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        mu = self.attrs._mu
        sigma = self.attrs._sigma
        gain = self.attrs._gain
        x = self.attrs._x
        dx = np.zeros_like(x)
        self.attrs._x._update_diff(context, dx, **kwargs)
        self.attrs._gain._update_diff(context, 0, **kwargs)

class LayerNormalization(Parametrized):

    def __init__(self, gain=0.1):
        self._gain = gain

    def weight_initiallize(self, input_size):
        self.params = {
            "gain": Variable(np.array([[self._gain]]), auto_update=True)}

    def forward(self, x):
        return layernorm(x, self.params["gain"])

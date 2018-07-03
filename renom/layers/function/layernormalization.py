
import numpy as np
from renom.cuda import cuda as cu
from renom.core import Node, Variable
from .parameterized import Parametrized
from renom.cuda.gpuvalue import *


class layernorm(Node):
    def __new__(cls, x, gain, bias):
        return cls.calc_value(x, gain, bias)

    @classmethod
    def _oper_cpu(cls, x, gain, bias):
        assert len(x.shape) is 2, "Currently only normalizes for dense networks."
        H = x.shape[1]
        sum = np.sum(x, axis=1, keepdims=True)
        mu = sum / H
        v1 = np.power(x - mu, 2)
        sigma = np.sqrt(np.sum(v1, axis=1, keepdims=True) / H)
        std_dev_x = gain / (sigma + 1e-5) * (x - mu)
        ret = cls._create_node(std_dev_x + bias)
        ret.attrs._std_dev = std_dev_x
        ret.attrs._x = x
        ret.attrs._gain = gain
        ret.attrs._sigma = sigma
        ret.attrs._mu = mu
        ret.attrs._bias = bias
        return ret

    @classmethod
    def _oper_gpu(cls, x, gain, bias):
        assert len(x.shape) is 2, "Currently only normalizes for dense networks."
        H = x.shape[1]
        sum = cu.cusum(get_gpu(x), axis=1, keepdims=True)
        mu = sum / H
        v1 = get_gpu(x).empty_like_me()
        cu.cupow(get_gpu(x) - mu, 2, v1)
        sigma = get_gpu(sum).empty_like_me()
        cu.cusqrt(cu.cusum(v1, axis=1, keepdims=True) / H, sigma)
        std_dev_x = get_gpu(gain) / (sigma + 1e-5) * (get_gpu(x) - mu)
        ret = cls._create_node(std_dev_x + get_gpu(bias))
        ret.attrs._x = x
        ret.attrs._gain = gain
        ret.attrs._bias = bias
        ret.attrs._std_dev = std_dev_x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        x = self.attrs._x
        dx = np.zeros_like(x)
        self.attrs._x._update_diff(context, dx, **kwargs)
        self.attrs._gain._update_diff(context, np.sum(
            self.attrs._std_dev, axis=0, keepdims=True), **kwargs)
        self.attrs._bias._update_diff(context, np.sum(
            np.ones_like(self), axis=0, keepdims=True), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        dx = get_gpu(self.attrs._x).zeros_like_me()
        self.attrs._x._update_diff(context, dx, **kwargs)
        self.attrs._gain._update_diff(context, cu.cusum(
            self.attrs._std_dev, axis=0, keepdims=True), **kwargs)
        self.attrs._bias._update_diff(context, cu.cusum(
            get_gpu(self).ones_like_me(), axis=0, keepdims=True), **kwargs)


class LayerNormalization(Parametrized):
    ''' Layer Normalization Model [1]
    Applies a shift to a standard bell curve formation for each input unit.
    The resultant bell curve can be transformed with the gain/bias parameters, displacing the mean with the bias
    or the variance with gain.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.random.rand(2,3)
        >>> layer = rm.LayerNormalization(bias=0)
        >>> x
        array([[0.5833913 , 0.39715111, 0.19503325],
               [0.74007066, 0.34642472, 0.57293373]])
        >>> layer(x)
        layernorm([[ 0.12076415,  0.00333703, -0.12410118],
                   [ 0.11587134, -0.12813905,  0.01226771]])



    .. [1] https://arxiv.org/pdf/1607.06450.pdf
    '''

    def __init__(self, gain=0.1, bias=1.0):
        self._gain = gain
        self._bias = bias

    def weight_initiallize(self, input_size):
        sz = input_size[0]
        self.params = {
            "gain": Variable(np.array([[self._gain for _ in range(sz)]]), auto_update=True),
            "bias": Variable(np.array([[self._bias for _ in range(sz)]]), auto_update=True)}

    def forward(self, x):
        return layernorm(x, self.params["gain"], self.params["bias"])

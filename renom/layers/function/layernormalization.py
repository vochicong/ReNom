
from __future__ import division
import numpy as np
from renom.cuda import cuda as cu
from renom.core import Node, Variable
import renom.operation as op
from .parameterized import Parametrized
from renom.cuda.gpuvalue import *


def get_std_distribution(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * np.power(x, 2))


def get_gen_distribution(x, mu=None, sigma=None):
    if mu is None:
        mu = get_mu(x)
    if sigma is None:
        sigma = get_sigma(x, mu)
    return 1 / sigma * get_std_distribution((x - mu) / sigma)


def get_mu(x):
    H = x.shape[1]
    sum = np.sum(x, axis=1, keepdims=True)
    return sum / H


def get_sigma(x, mu=None):
    if mu is None:
        mu = get_mu(x)
    H = x.shape[1]
    sum = np.sum(np.power(x - mu, 2), axis=1, keepdims=True)
    return np.sqrt(sum / H)


def get_mu_diff(x):
    H = x.shape[1]
    return 1 / H


def get_sigma_diff(x):
    H = x.shape[1]
    mu = get_mu(x)
    sigma = get_sigma(x, mu)
    inside = (2 * x + H * (2 * mu / H) - 2 * (np.sum(x, axis=1, keepdims=True) / H + mu)) / H
    return 1 / (2 * sigma) * inside


class layernorm(Node):
    def __new__(cls, x, gain, bias):
        return cls.calc_value(x, gain, bias)

    @classmethod
    def _oper_cpu(cls, x, gain, bias):
        assert len(x.shape) is 2, "Currently only normalizes for dense networks."
        mu = get_mu(x)
        sigma = get_sigma(x, mu) + 1e-5
        normalized = (x - mu) / sigma
        ret = cls._create_node(normalized * gain + bias)
        ret.attrs._x = x
        ret.attrs._mu = mu
        ret.attrs._normalized = normalized
        ret.attrs._sigma = sigma
        ret.attrs._gain = gain
        ret.attrs._bias = bias
        return ret

    @classmethod
    def _oper_gpu(cls, x, gain, bias):
        assert len(x.shape) is 2, "Currently only normalizes for dense networks."
        _x, _gain, _bias = map(get_gpu, [x, gain, bias])
        H = x.shape[1]
        sum1 = op.sum(_x, axis=1).reshape(-1, 1)
        mu = get_gpu(sum1 / H)
        sum2 = op.sum((_x - mu) ** 2, axis=1).reshape(-1,1)
        sigma = op.sqrt(sum2 / H) + 1e-5
        normalized = (_x - mu) / get_gpu(sigma)
        ret = cls._create_node(normalized * _gain + _bias)
        ret.attrs._x = x
        ret.attrs._sigma = sigma
        ret.attrs._mu = mu
        ret.attrs._gain = gain
        ret.attrs._bias = bias
        ret.attrs._normalized = normalized
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        x = self.attrs._x
        mu = self.attrs._mu
        sigma = self.attrs._sigma
        gain = self.attrs._gain
        sigma_diff = get_sigma_diff(x)
        mu_diff = get_mu_diff(x)
        dx = dy / sigma - sigma_diff * np.sum(x * dy, axis=1, keepdims=True) / np.power(sigma, 2) - np.sum(
            mu_diff * dy, axis=1, keepdims=True) / sigma + sigma_diff * \
            np.sum(dy, axis=1, keepdims=True) * mu / np.power(sigma, 2)
        dx *= gain
        if isinstance(x, Node):
            self.attrs._x._update_diff(context, dx, **kwargs)
        self.attrs._gain._update_diff(context, np.sum(
            self.attrs._normalized * dy, axis=0, keepdims=True), **kwargs)
        self.attrs._bias._update_diff(context, np.sum(dy, axis=0, keepdims=True), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        dx = get_gpu(self.attrs._x).zeros_like_me()
        H = self.attrs._x.shape[1]
        x = get_gpu(self.attrs._x)
        mu = get_gpu(self.attrs._mu)
        sigma = get_gpu(self.attrs._sigma)
        gain = get_gpu(self.attrs._gain)
        dy = get_gpu(dy)
        mu_diff = get_gpu(get_mu_diff(x))
        sigma_diff = 1 / (2 * sigma) * ((2 * x + H * (2 * mu / H) - get_gpu(2 * (op.sum(x, axis=1).reshape(-1,1) / H + mu)) ) / H)
        dx = get_gpu(dy / sigma) - get_gpu(sigma_diff * get_gpu(op.sum(x * dy, axis=1).reshape(-1,1)) / (sigma ** 2)) - get_gpu(get_gpu(op.sum(
            mu_diff * dy, axis=1).reshape(-1,1)) / sigma) + get_gpu(sigma_diff * \
            get_gpu(op.sum(dy, axis=1).reshape(-1,1)) * mu / (sigma ** 2))
        dx *= gain
        self.attrs._x._update_diff(context, dx, **kwargs)
        self.attrs._gain._update_diff(context, cu.cusum(
            self.attrs._normalized * get_gpu(dy), axis=0, keepdims=True), **kwargs)
        self.attrs._bias._update_diff(context, cu.cusum(
            get_gpu(dy), axis=0, keepdims=True), **kwargs)


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

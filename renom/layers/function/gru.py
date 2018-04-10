#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.layers.activation.sigmoid import sigmoid
from renom.layers.activation.tanh import tanh
from renom.core import Node, Variable, get_gpu, precision, GPUValue
from renom.operation import dot, sum
from renom.utility.initializer import GlorotNormal
from .parameterized import Parametrized
from renom.cuda import cuda as cu


def gate(x):
    return 1. / (1. + np.exp(-x))


def activation(x):
    return np.tanh(x)


def gate_diff(x):
    return x * (- x + 1.)


def activation_diff(x):
    return (1.0 - x**2)


class gru(Node):
    '''
    @ parameters
    cls: Self variable for Python
    x: input value to the node
    pz: The previously calculated value within the same model
    w: the weights to be multiplied with the input
    wr: the weights to be multiplied with the previous input
    b: the biases to be added
    '''
    def __new__(cls, x, pz, w, wr, b):
        return cls.calc_value(x, pz, w, wr, b)

    @classmethod
    def _oper_cpu(cls, x, pz, w, wr, b):
        h_minus = np.zeros((x.shape[0], w.shape[1] // 3), dtype=precision) if pz is None else pz

        m = w.shape[1] // 4
        w_r, w_z, w_h = np.split(w, m, axis=1)
        u_r, u_z, u_h = np.split(wr, m, axis=1)
        b_r, b_z, b_h = np.split(b, m, axis=1)

        print (w_z.shape)
        print (x.shape)
        B = w_z*x + u_z*h_minus + b_z
        C = w_r*x + u_r*h_minus + b_r

        z = sigmoid(B)
        r = sigmoid(C)

        A = w_h*x + u_h*h_minus*r + b_z
        h_tilde = tanh(A)

        h = (1 - z) * h_minus + z * h_tilde

        ret = cls._create_node(h)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._wr = wr
        ret.attrs._b = b
        ret.attrs._pz = pz
        ret.attrs._A = A
        ret.attrs._B = B
        ret.attrs._C = C
        ret.attrs._h_tilde = h_tilde
        ret.attrs._u_h = u_h

        #if isinstance(pz, Node):
        #    pz.attrs._pfgate = gated[:, :m]

        return ret
'''
    @classmethod
    def _oper_gpu(cls, x, pz, ps, w, wr, b):
        if ps is None:
            tmp = GPUValue(shape=(x.shape[0], w.shape[1] // 4))
            s_p = tmp.zeros_like_me()
            z_p = tmp.zeros_like_me()
        else:
            s_p = ps
            z_p = get_gpu(pz)

        u = dot(x, w) + dot(z_p, wr) + b

        z = get_gpu(z_p).empty_like_me()
        state = get_gpu(s_p).empty_like_me()

        cu.cugru_forward_activate(get_gpu(u))
        cu.cugru_forward(get_gpu(u), get_gpu(state), get_gpu(s_p), get_gpu(z))

        ret = cls._create_node(z)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._wr = wr
        ret.attrs._b = b
        ret.attrs._pz = pz
        ret.attrs._u = u
        ret.attrs._pstate = s_p
        ret.attrs._state = state
        ret._state = state

        if isinstance(pz, Node):
            pz.attrs._pfgate = u

        return ret
'''

def _backward_cpu(self, context, dy, **kwargs):
        #n, m = dy.shape

        A, B, C = self.attrs._A, self.attrs._B, self.attrs._C
        h_tilde = self.attrs._h_tilde
        x_t, h_minus = self.attrs._x, self.attrs._pz
        u_h = self.attrs._u_h

        dydwz = sigmoid(B)(1 - sigmoid(B)) * x_t + h_tilde * (sigmoid(B) * (1 - sigmoid(B))) * x_t
        dyduz = sigmoid(B)(1 - sigmoid(B)) * h_minus + h_tilde * (sigmoid(B) * (1 - sigmoid(B))) * h_minus

        dydwr = sigmoid(B) * activation_diff(tanh(A)) * u_h * h_minus * sigmoid(C)(1 - sigmoid(C)) * x_t
        dydur = sigmoid(B) * activation_diff(tanh(A)) * u_h * h_minus * sigmoid(C)(1 - sigmoid(C)) * h_minus

        dydwh = sigmoid(B) * activation_diff(tanh(A)) * x_t
        dyduh = sigmoid(B) * activation_diff(tanh(A)) * h_minus * sigmoid(C)

        dw = np.stack((dydwz, dydwr, dydwh))
        dwr = np.stack((dyduz, dydur, dyduh))

        #if isinstance(self.attrs._x, Node):
        #    self.attrs._x._update_diff(context, dx)

        #if isinstance(w, Node):
        #    w._update_diff(context, np.dot(self.attrs._x.T, dr))

        #if isinstance(wr, Node):
        #    wr._update_diff(context, np.dot(self.T, drt))

        #if isinstance(b, Node):
        #    b._update_diff(context, np.sum(dr, axis=0, keepdims=True))

        #if isinstance(self.attrs._pz, Node):
        #    self.attrs._pz._update_diff(context, np.dot(dr, wr.T))
'''
    def _backward_gpu(self, context, dy, **kwargs):
        w = self.attrs._w
        wr = self.attrs._wr
        b = self.attrs._b

        u = self.attrs._u
        s = tanh(self.attrs._state)
        ps = self.attrs._pstate

        drt = context.restore(wr, get_gpu(u).zeros_like_me())
        dou = context.restore(w, get_gpu(dy).zeros_like_me())
        pfg = getattr(self.attrs, "_pfgate", get_gpu(u).zeros_like_me())

        e = get_gpu(dy)

        dr, dou_n = (get_gpu(a).empty_like_me() for a in (drt, dou))
        cu.cugru_backward(*map(get_gpu, (u, dr, s, ps, e, pfg, dou, dou_n)))

        dx = dot(dr, w.T)

        context.store(wr, dr)
        context.store(w, dou_n)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx)

        if isinstance(w, Node):
            w._update_diff(context, dot(self.attrs._x.T, dr))

        if isinstance(wr, Node):
            wr._update_diff(context, dot(self.T, drt))

        if isinstance(b, Node):
            b._update_diff(context, sum(dr, axis=0))

        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, dot(dr, wr.T))
'''

class Gru(Parametrized):
    '''Long short time memory[4]_ .
    Gru object has 8 weights and 4 biases parameters to learn.

    Weights applied to the input of the input gate, forget gate and output gate.
    :math:`W_{ij}, Wgi_{ij}, Wgf_{ij}, Wgo_{ij}`

    Weights applied to the recuurent input of the input gate, forget gate and output gate.
    :math:`R_{ij}, Rgi_{ij}, Rgf_{ij}, Rgo_{ij}`

    .. math::
        u^t_{i} &= \sum_{j = 0}^{J-1} W_{ij}x^t_{j} +
            \sum_{k = 0}^{K-1} R_{ik}y^{t-1}_{k} + b_i \\\\
        gi^t_{i} &= \sum_{j = 0}^{J-1} Wgi_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgi_{ik}y^{t-1}_{k} + bi_i \\\\
        gf^t_{i} &= \sum_{j = 0}^{J-1} Wgfi_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgf_{ik}y^{t-1}_{k} + bi_f \\\\
        go^t_{i} &= \sum_{j = 0}^{J-1} Wgo_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgo_{ik}y^{t-1}_{k} + bi_o \\\\
        s^t_i &= sigmoid(gi^t_{i})tanh(u^t_{i}) + s^{t-1}_isigmoid(gf^t_{i}) \\\\
        y^t_{i} &= go^t_{i}tanh(s^t_{i})

    If the argument ``input_size`` is passed, this layers' weight is initialized
    in the __init__ function.
    Otherwise, the weight is initialized in its first forward calculation.

    Args:
        output_size (int): Output unit size.
        input_size (int): Input unit size.
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> n, d, t = (2, 3, 4)
        >>> x = rm.Variable(np.random.rand(n, d))
        >>> layer = rm.Gru(2)
        >>> z = 0
        >>> for i in range(t):
        ...     z += rm.sum(layer(x))
        ...
        >>> grad = z.grad()    # Backpropagation.
        >>> grad.get(x)    # Gradient of x.
        Add([[-0.01853334, -0.0585249 ,  0.01290053],
             [-0.0205425 , -0.05837972,  0.00467286]], dtype=float32)
        >>> layer.truncate()

    .. [4] Learning Precise Timing with GRU Recurrent Networks
    '''

    def __init__(self, output_size, input_size=None, initializer=GlorotNormal()):
        self._size_o = output_size
        self._initializer = initializer
        super(Gru, self).__init__(input_size)

    def weight_initiallize(self, size_i):
        size_i = size_i[0]
        size_o = self._size_o
        bias = np.zeros((1, size_o * 3), dtype=precision)
        bias[:, size_o:size_o * 2] = 1
        self.params = {
            "w": Variable(self._initializer((size_i, size_o * 3)), auto_update=True),
            "wr": Variable(self._initializer((size_o, size_o * 3)), auto_update=True),
            "b": Variable(bias, auto_update=True),
        }

    def forward(self, x):
        ret = gru(x, getattr(self, "_z", None),
                   self.params.w,
                   self.params.wr,
                   self.params.b)
        self._z = ret
        self._state = getattr(ret.attrs, '_state', None)
        return ret

    def truncate(self):
        """Truncates temporal connection."""
        self._z = None
        self._state = None

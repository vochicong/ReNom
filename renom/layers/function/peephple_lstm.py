#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.layers.activation.sigmoid import sigmoid
from renom.layers.activation.tanh import tanh
from renom.core import Node, Variable, to_value, get_gpu, precision, GPUValue
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


class peephole_lstm(Node):
    def __new__(cls, x, pz, ps, w, wr, wc, b):
        return cls.calc_value(x, pz, ps, w, wr, wc, b)

    @classmethod
    def _oper_cpu(cls, x, pz, ps, w, wr, wc, b):
        s = np.zeros((x.shape[0], w.shape[1] // 4), dtype=precision) if ps is None else ps
        z = np.zeros((x.shape[0], w.shape[1] // 4), dtype=precision) if pz is None else pz

        u = np.dot(x, w) + np.dot(z, wr) + b
        m = u.shape[1] // 4
        u, gate_u = np.split(u, [m, ], axis=1)
        u = tanh(u)
        fg = sigmoid(s * wc[:, :m] + gate_u[:, :m])
        ig = sigmoid(s * wc[:, m:2 * m] + gate_u[:, m:2 * m])
        state = ig * u + fg * s
        og = sigmoid(state * wc[:, 2 * m:] + gate_u[:, 2 * m:])
        z = tanh(state) * og

        gated = np.hstack((fg, ig, og))

        ret = cls._create_node(z)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._wr = wr
        ret.attrs._wc = wc
        ret.attrs._b = b
        ret.attrs._u = u
        ret.attrs._pz = pz
        ret.attrs._pstate = ps
        ret.attrs._state = state
        ret.attrs._gated = gated
        if isinstance(pz, Node):
            pz.attrs._pfgate = gated[:, :m]

        return ret

    @classmethod
    def _oper_gpu(cls, x, pz, ps, w, wr, wc, b):
        if ps is None:
            s_p = GPUValue(shape=(x.shape[0], w.shape[1] // 4)).zeros_like_me()
            z_p = s_p.zeros_like_me()
        else:
            s_p, z_p = map(get_gpu, (ps, pz))

        s = s_p.empty_like_me()
        u = get_gpu(dot(x, w) + dot(z_p, wr) + b)
        z = z_p.zeros_like_me()
        cu.cupeepholelstm_forward(u, get_gpu(wc), s_p, s, z)

        ret = cls._create_node(z)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._wr = wr
        ret.attrs._wc = wc
        ret.attrs._b = b
        ret.attrs._u = u
        ret.attrs._pz = pz
        ret.attrs._pstate = ps
        ret.attrs._state = s

        if isinstance(pz, Node):
            pz.attrs._pfgate = u
        return ret


    def _backward_cpu(self, context, dy, temporal=False, **kwargs):
        n, m = dy.shape

        w = self.attrs._w
        wr = self.attrs._wr
        wc = self.attrs._wc
        b = self.attrs._b

        u = self.attrs._u
        s = tanh(self.attrs._state)

        gated = self.attrs._gated
        gd = gate_diff(gated)
        ps = self.attrs._pstate

        pfg = getattr(self.attrs, "_pfgate", np.zeros_like(self))

        dot = context.restore(w, np.zeros((n, m), dtype=dy.dtype))
        drt = context.restore(wr, np.zeros((n, m * 4), dtype=dy.dtype))

        do = dy * s * gd[:, 2*m:]

        dou = dy * gated[:, 2*m:] * activation_diff(s) + do * wc[:, 2*m:]

        if temporal:
            dou += pfg * dot + drt[:, m:2*m] * wc[:, :m] + drt[:, 2*m:3*m] * wc[:, m:2 * m]

        df = dou * gd[:, :m] * ps if ps is not None else np.zeros_like(dou)
        di = dou * gd[:, m:2 * m] * u
        du = dou * activation_diff(u) * gated[:, m:2 * m]

        dr = np.hstack((du, df, di, do))

        context.store(wr, dr)
        context.store(w, dou)

        if isinstance(self.attrs._x, Node):
            dx = np.dot(dr, w.T)
            self.attrs._x._update_diff(context, dx)

        if isinstance(w, Node):
            w._update_diff(context, np.dot(self.attrs._x.T, dr))

        if isinstance(wr, Node) and temporal:
            wr._update_diff(context, np.dot(self.T, drt))

        if isinstance(wc, Node):
            dwc = np.zeros(wc.shape, dtype=wc.dtype)
            dwc[:, 2*m:] = np.sum(do*self.attrs._state, axis=0)
            if temporal:
                dwc[:, :m] = np.sum(drt[:, m:2*m]*self.attrs._state, axis=0)
                dwc[:, m:2*m] = np.sum(drt[:, 2*m:3*m]*self.attrs._state, axis=0)
            wc._update_diff(context, dwc)

        if isinstance(b, Node):
            b._update_diff(context, np.sum(dr, axis=0, keepdims=True))

        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, np.dot(dr, wr.T), temporal=True)

    def _backward_gpu(self, context, dy, temporal=False, **kwargs):
        w = self.attrs._w
        wr = self.attrs._wr
        wc = self.attrs._wc
        b = self.attrs._b

        u = self.attrs._u
        s = tanh(self.attrs._state)
        ps = get_gpu(s).zeros_like_me() if self.attrs._pstate is None else self.attrs._pstate

        drt = context.restore(wr, get_gpu(u).zeros_like_me())
        dou = context.restore(w, get_gpu(dy).zeros_like_me())
        pfg = getattr(self.attrs, "_pfgate", get_gpu(u).zeros_like_me())

        dr = get_gpu(drt).empty_like_me()
        dwc = get_gpu(wc).empty_like_me()

        cu.cupeepholelstm_backward(*map(get_gpu, (u, ps, s, pfg, wc, dy, drt, dou, dr, dwc, temporal)))

        context.store(wr, drt)
        context.store(w, dou)

        if isinstance(self.attrs._x, Node):
            dx = dot(dr, w.T)
            self.attrs._x._update_diff(context, dx)

        if isinstance(w, Node):
            w._update_diff(context, dot(self.attrs._x.T, dr))

        if isinstance(wr, Node) and temporal:
            wr._update_diff(context, dot(self.T, drt))

        if isinstance(wc, Node):
            wc._update_diff(context, sum(dwc, axis=0))

        if isinstance(b, Node):
            b._update_diff(context, sum(dr, axis=0))

        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, dot(dr, wr.T), temporal=True)


class PeepholeLstm(Parametrized):
    '''Long short time memory with peephole [4]_ .
    Lstm object has 12 weights and 4 biases parameters to learn.

    Weights applied to the input of the input gate, forget gate and output gate.
    :math:`W_{ij}, Wgi_{ij}, Wgf_{ij}, Wgo_{ij}`

    Weights applied to the recuurent input of the input gate, forget gate and output gate.
    :math:`R_{ij}, Rgi_{ij}, Rgf_{ij}, Rgo_{ij}`

    Weights applied to the state input of the input gate, forget gate and output gate.
    :math:`P_{ij}, Pgi_{ij}, Pgf_{ij}, Pgo_{ij}`

    .. math::
        u^t_{i} &= \sum_{j = 0}^{J-1} W_{ij}x^t_{j} +
            \sum_{k = 0}^{K-1} R_{ik}y^{t-1}_{k} +
            P_{i}s^{t-1}_{i} + b_i \\\\
        gi^t_{i} &= \sum_{j = 0}^{J-1} Wgi_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgi_{ik}y^{t-1}_{k} +
                Pgi_{i}s^{t-1}_{i} + bi_i \\\\
        gf^t_{i} &= \sum_{j = 0}^{J-1} Wgfi_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgf_{ik}y^{t-1}_{k} +
                Pgf_{i}s^{t-1}_{i} + bi_f \\\\
        go^t_{i} &= \sum_{j = 0}^{J-1} Wgo_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgo_{ik}y^{t-1}_{k} +
                Pgo_{i}s^{t}_{i} + bi_o \\\\
        s^t_i &= sigmoid(gi^t_{i})tanh(u^t_{i}) + s^{t-1}_isigmoid(gf^t_{i}) \\\\
        y^t_{i} &= go^t_{i}tanh(s^t_{i})

    Args:
        output_size (int): Output unit size.
        input_size (int): Input unit size.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> n, d, t = (2, 3, 4)
        >>> x = rm.Variable(np.random.rand(n, d))
        >>> layer = rm.Lstm(2)
        >>> z = 0
        >>> for i in range(t):
        ...     z += rm.sum(layer(x))
        ...
        >>> grad = z.grad()    # Backpropagation.
        >>> grad.get(x)    # Gradient of x.
        Add([[-0.01853334, -0.0585249 ,  0.01290053],
             [-0.0205425 , -0.05837972,  0.00467286]], dtype=float32)
        >>> layer.truncate()

    .. [4] Felix A. Gers, Nicol N. Schraudolph, J ̈urgen Schmidhuber.
        Learning Precise Timing with LSTM Recurrent Networks
    '''

    def __init__(self, output_size, input_size=None, initializer=GlorotNormal()):
        self._size_o = output_size
        self._initializer = initializer
        super(PeepholeLstm, self).__init__(input_size)

    def weight_initiallize(self, size_i):
        size_i = size_i[0]
        size_o = self._size_o
        bias = np.zeros((1, size_o * 4), dtype=precision)
        bias[:, size_o:size_o * 2] = 1
        self.params = {
            "w": Variable(self._initializer((size_i, size_o * 4)), auto_update=True),
            "wr": Variable(self._initializer((size_o, size_o * 4)), auto_update=True),
            "wc": Variable(self._initializer((1, size_o * 3)), auto_update=True),
            "b": Variable(bias, auto_update=True),
        }

    def forward(self, x):
        ret = peephole_lstm(x, getattr(self, "_z", None),
                            getattr(self, "_state", None),
                            self.params.w,
                            self.params.wr,
                            self.params.wc,
                            self.params.b)
        self._z = ret
        self._state = getattr(ret.attrs, '_state', None)
        return ret

    def truncate(self):
        """Truncates temporal connection."""
        self._z = None
        self._state = None

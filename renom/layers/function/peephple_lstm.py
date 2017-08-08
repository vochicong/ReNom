#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.layers.activation.sigmoid import sigmoid
from renom.layers.activation.tanh import tanh
from renom.core import Node, Variable, to_value, get_gpu, precision
from renom.operation import dot, sum
from renom.utility.initializer import GlorotNormal
from .parameterized import Parametrized


def gate(x):
    return 1. / (1. + np.exp(-x))


def activation(x):
    return np.tanh(x)


def gate_diff(x):
    return x * (- x + 1.)


def activation_diff(x):
    return (1.0 - x**2)


class peephole_lstm(Node):
    def __new__(cls, x, pz, ps, parameter):
        return cls.calc_value(x, pz, ps, parameter)

    @classmethod
    def _oper_cpu(cls, x, pz, ps, parameter):
        p = parameter
        s = np.zeros((x.shape[0], p["w"].shape[1]), dtype=precision) if ps is None else ps
        z = np.zeros((x.shape[0], p["w"].shape[1]), dtype=precision) if pz is None else pz

        u = dot(x, p["w"]) + dot(z, p["wr"]) + p["b"]

        gate_f = sigmoid(dot(x, p["wf"]) +
                         dot(z, p["wfr"]) + p["wfc"] * s + p["bf"])
        gate_i = sigmoid(dot(x, p["wi"]) +
                         dot(z, p["wir"]) + p["wic"] * s + p["bi"])

        state = gate_i * tanh(u) + gate_f * s

        gate_o = sigmoid(
            dot(x, p["wo"]) + dot(z, p["wor"]) + p["bo"] + p["woc"] * state)

        z = tanh(state) * gate_o

        ret = cls._create_node(z)

        ret.attrs._x = x
        ret.attrs._p = parameter
        ret.attrs._u = u
        ret.attrs._pgated_f = None
        ret.attrs._pstate = ps
        ret.attrs._state = state
        ret.attrs._gated_o = gate_o
        ret.attrs._gated_f = gate_f
        ret.attrs._gated_i = gate_i
        ret.attrs._dt_d = [p[k] for k in ["wr", "wi", "wf", "wo", "w"]]
        ret._state = state

        return ret

    @classmethod
    def _oper_gpu(cls, x, pz, ps, parameter):
        p = parameter
        s = get_gpu(np.zeros((x.shape[0], p["w"].shape[1]), dtype=precision)) if ps is None else ps
        z = get_gpu(s).zeros_like_me() if pz is None else pz

        u = dot(x, p["w"]) + dot(z, p["wr"]) + p["b"]

        gate_f = sigmoid(dot(x, p["wf"]) +
                         dot(z, p["wfr"]) + p["wfc"] * s + p["bf"])
        gate_i = sigmoid(dot(x, p["wi"]) +
                         dot(z, p["wir"]) + p["wic"] * s + p["bi"])

        state = gate_i * tanh(u) + gate_f * s

        gate_o = sigmoid(
            dot(x, p["wo"]) + dot(z, p["wor"]) + p["bo"] + p["woc"] * state)

        z = tanh(state) * gate_o

        ret = cls._create_node(get_gpu(z))
        ret.attrs._x = x
        ret.attrs._p = parameter
        ret.attrs._u = u
        ret.attrs._pgated_f = None
        ret.attrs._pstate = ps
        ret.attrs._state = state
        ret.attrs._gated_o = gate_o
        ret.attrs._gated_f = gate_f
        ret.attrs._gated_i = gate_i
        ret.attrs._dt_d = [p[k] for k in ["wr", "wi", "wf", "wo", "w"]]
        ret._state = state

        return ret

    def _backward_cpu(self, context, dy):
        p = self.attrs._p
        s = self.attrs._state
        ps = self.attrs._pstate
        u = self.attrs._u

        go = self.attrs._gated_o
        gf = self.attrs._gated_f
        gi = self.attrs._gated_i
        pgf = np.zeros_like(gf) if self.attrs._pgated_f is None else self.attrs._pgated_f

        drt, dit, dft, dot, dct = (context.restore(dt, np.zeros_like(dy))
                                   for dt in self.attrs._dt_d)

        activated_s = tanh(s)
        activated_u = tanh(u)

        e = dy + np.dot(drt, p["wr"].T) + np.dot(dit, p["wir"].T) + \
            np.dot(dft, p["wfr"].T) + np.dot(dot, p["wor"].T)

        do = gate_diff(go) * activated_s * e
        ds = go * activation_diff(activated_s) * e
        dc = ds + pgf * dct + p["wfc"] * dft + p["wic"] * dit + p["woc"] * do

        df = gate_diff(gf) * ps * dc if ps is not None else np.zeros_like(gf)
        di = gate_diff(gi) * activated_u * dc

        d = gi * activation_diff(activated_u) * dc

        dx = np.dot(d, p["w"].T) \
            + np.dot(di, p["wi"].T) \
            + np.dot(do, p["wo"].T) \
            + np.dot(df, p["wf"].T)

        for dt_d, dt in zip(self.attrs._dt_d, (d, di, df, do, dc)):
            context.store(dt_d, dt)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx)

        for k, diff in zip(("w", "wo", "wi", "wf"), (d, do, di, df)):
            if isinstance(p[k], Node):
                p[k]._update_diff(context, np.dot(to_value(self.attrs._x).T, diff))

        for k, diff in zip(("wr", "wor", "wir", "wfr"), (drt, dot, dit, dft)):
            if isinstance(p[k], Node):
                p[k]._update_diff(context, np.dot(to_value(self).T, diff))

        for k, diff in zip(("wfc", "wic", "woc"), (dft, dit, do)):
            if isinstance(p[k], Node):
                p[k]._update_diff(context, np.sum(diff * s, axis=0, keepdims=True))

        for k, diff in zip(("b", "bf", "bi", "bo"), (d, df, di, do)):
            if isinstance(p[k], Node):
                p[k]._update_diff(context, np.sum(diff, axis=0, keepdims=True))

    def _backward_gpu(self, context, dy):
        p = self.attrs._p
        s = self.attrs._state
        ps = self.attrs._pstate
        u = self.attrs._u

        go = self.attrs._gated_o
        gf = self.attrs._gated_f
        gi = self.attrs._gated_i
        pgf = get_gpu(gf).zeros_like_me() if self.attrs._pgated_f is None else self.attrs._pgated_f

        drt, dit, dft, doot, dct = (context.restore(dt, get_gpu(dy).zeros_like_me())
                                    for dt in self.attrs._dt_d)

        activated_s = tanh(s)
        activated_u = tanh(u)

        e = dy + get_gpu(dot(drt, p["wr"].T)) \
               + get_gpu(dot(dit, p["wir"].T)) + \
               + get_gpu(dot(dft, p["wfr"].T)) + \
               + get_gpu(dot(doot, p["wor"].T))

        do = gate_diff(go) * activated_s * e
        ds = go * activation_diff(activated_s) * e
        dc = ds + pgf * dct + p["wfc"] * dft + p["wic"] * dit + p["woc"] * do

        df = gate_diff(gf) * ps * dc if ps is not None else get_gpu(gf).zeros_like_me()
        di = gate_diff(gi) * activated_u * dc

        d = gi * activation_diff(activated_u) * dc

        dx = dot(d, p["w"].T) \
            + dot(di, p["wi"].T) \
            + dot(do, p["wo"].T) \
            + dot(df, p["wf"].T)

        for dt_d, dt in zip(self.attrs._dt_d, (d, di, df, do, dc)):
            context.store(dt_d, get_gpu(dt))

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, get_gpu(dx))

        for k, diff in zip(("w", "wo", "wi", "wf"), (d, do, di, df)):
            if isinstance(p[k], Node):
                p[k]._update_diff(context, get_gpu(dot(self.attrs._x.T, diff)))

        for k, diff in zip(("wr", "wor", "wir", "wfr"), (drt, doot, dit, dft)):
            if isinstance(p[k], Node):
                p[k]._update_diff(context, get_gpu(dot(self.T, diff)))

        for k, diff in zip(("wfc", "wic", "woc"), (dft, dit, do)):
            if isinstance(p[k], Node):
                p[k]._update_diff(context, sum(diff * get_gpu(s), axis=0))

        for k, diff in zip(("b", "bf", "bi", "bo"), (d, df, di, do)):
            if isinstance(p[k], Node):
                p[k]._update_diff(context, sum(diff, axis=0))


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
        self.params = {
            "w": Variable(self._initializer((size_i, size_o)), auto_update=True),
            "wr": Variable(self._initializer((size_o, size_o)), auto_update=True),
            "wf": Variable(self._initializer((size_i, size_o)), auto_update=True),
            "wfr": Variable(self._initializer((size_o, size_o)), auto_update=True),
            "wfc": Variable(self._initializer((1, size_o)), auto_update=True),
            "wi": Variable(self._initializer((size_i, size_o)), auto_update=True),
            "wir": Variable(self._initializer((size_o, size_o)), auto_update=True),
            "wic": Variable(self._initializer((1, size_o)), auto_update=True),
            "wo": Variable(self._initializer((size_i, size_o)), auto_update=True),
            "wor": Variable(self._initializer((size_o, size_o)), auto_update=True),
            "woc": Variable(self._initializer((1, size_o)), auto_update=True),
            "b": Variable(np.zeros((1, size_o), dtype=precision), auto_update=True),
            "bf": Variable(np.ones((1, size_o), dtype=precision), auto_update=True),
            "bo": Variable(np.zeros((1, size_o), dtype=precision), auto_update=True),
            "bi": Variable(np.zeros((1, size_o), dtype=precision), auto_update=True),
        }

    def forward(self, x):
        ret = peephole_lstm(x, getattr(self, "_z", None),
                            getattr(self, "_state", None),
                            self.params)
        self._z = to_value(ret)
        self._state = to_value(getattr(ret, '_state', None))
        if hasattr(self, "_last_node") and self._last_node is not None and ret.attrs.get_attrs():
            setattr(self._last_node.attrs, "_pgated_f", to_value(ret.attrs._gated_f))
        self._last_node = ret
        return ret

    def truncate(self):
        """Truncates temporal connection."""
        self._last_node = None
        self._z = None
        self._state = None

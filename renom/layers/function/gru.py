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


def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_diff(x):
    return (1.0 - tanh(x) ** 2)


class gru(Node):
    '''
    @ parameters
    cls: Self variable for Python
    x: input value to the node
    pz: The previously calculated value within the same model
    w: the weights to be multiplied with the input
    u: the weights to be multiplied with the previous input
    b: the biases to be added
    '''
    def __new__(cls, x, pz, w, u, b):
        return cls.calc_value(x, pz, w, u, b)

    @classmethod
    def _oper_cpu(cls, x, pz, w, u, b):

        # Initialize Variables
        m = w.shape[1] // 3
        w_z, w_r, w_h = np.split(w, [m, m * 2, ], axis=1)
        b_z, b_r, b_h = np.split(b, [m, m * 2], axis=1)
        u_z, u_r, u_h = np.split(u, [m, m * 2], axis=1)
        hminus = np.zeros((x.shape[0], w.shape[1] // 3), dtype=precision) if pz is None else pz
        # Perform Forward Calcuations
        # z = sigmoid( dot(x,w_z) + dot(h_minus,u_z) + b_z)
        A = dot(x, w_z) + hminus * u_z + b_z
        B = dot(x, w_r) + u_r * hminus + b_r
        C = dot(x, w_h) + sigmoid(B) * u_h * hminus + b_h
        h = sigmoid(A) + tanh(C)

        # Store Variables for Graph
        ret = cls._create_node(h)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._w_z = w_z
        ret.attrs._w_r = w_r
        ret.attrs._w_h = w_h
        ret.attrs._b = b
        ret.attrs._b_z = b_z
        ret.attrs._b_r = b_r
        ret.attrs._b_h = b_h
        ret.attrs._u = u
        ret.attrs._u_z = u_z
        ret.attrs._u_h = u_h
        ret.attrs._u_r = u_r
        ret.attrs._pz = hminus
        ret.attrs._A = A
        ret.attrs._B = B
        ret.attrs._C = C

        return ret

    def _backward_cpu(self, context, dy, **kwargs):

        #print ('Current depth is: {:d}'.format(self._count))
        x = self.attrs._x
        w_z = self.attrs._w_z
        w_r = self.attrs._w_r
        w_h = self.attrs._w_h
        A = self.attrs._A
        B = self.attrs._B
        C = self.attrs._C
        u_z = self.attrs._u_z
        u_h = self.attrs._u_h
        u_r = self.attrs._u_r
        hminus = self.attrs._pz
        y = dy

        dA = sigmoid_diff(A)
        dB = sigmoid_diff(B)
        dC = tanh_diff(C)

        # Calculate dx
        dx_z = dot(y * dA, w_z.T)
        dx_r = dot(y * dB * dC * u_h * hminus, w_r.T)
        dx_h = dot(y * dC, w_h.T)
        dx = dx_z + dx_r + dx_h

        # Calculate dw
        dw_z = dot(x.T, y * dA)
        dw_r = dot(x.T, y * dB * dC * u_h * hminus)
        dw_h = dot(x.T, y * dC)
        dw = np.concatenate([dw_z, dw_r, dw_h], axis=1)

        # Calculate db
        db_z = np.sum(y * dA, axis=0, keepdims=True)
        db_r = np.sum(y * dB * dC * u_h * hminus, axis=0, keepdims=True)
        db_h = np.sum(y * dC, axis=0, keepdims=True)
        db = np.concatenate([db_z, db_r, db_h], axis=1)

        du_z = np.sum(dA * hminus * y, axis=0, keepdims=True)
        du_r = np.sum(y * dC * dB * u_h * hminus * hminus, axis=0, keepdims=True)
        du_h = np.sum(sigmoid(B) * dC * y * hminus, axis=0, keepdims=True)
        du = np.concatenate([du_z, du_r, du_h], axis=1)

        pz_z = y * dA * u_z
        pz_r = y * dC * dB * u_h * hminus * u_r
        pz_h = y * dC * sigmoid(B) * u_h

        dpz = pz_z + pz_r + pz_h

        self.attrs._x._update_diff(context, dx)
        self.attrs._w._update_diff(context, dw)
        self.attrs._b._update_diff(context, db)
        self.attrs._u._update_diff(context, du)
        if isinstance(hminus, Node):
            self.attrs._pz._update_diff(context, dpz)


class Gru(Parametrized):

    def __init__(self, output_size, input_size=None, initializer=GlorotNormal()):
        self._size_o = output_size
        self._initializer = initializer
        super(Gru, self).__init__(input_size)

    def weight_initiallize(self, size_i):
        size_i = size_i[0]
        size_o = self._size_o
        bias = np.zeros((1, size_o * 3), dtype=precision)
        bias[:, :] = 1
        # Use count to determine the depth of the current node
        self._count = 0
        # At this point, all connected units in the same layer will use the SAME weights
        self.params = {
            "w": Variable(self._initializer((size_i, size_o * 3)), auto_update=True),
            "u": Variable(self._initializer((1, size_o * 3)), auto_update=True),
            "b": Variable(bias, auto_update=True),
        }

    def forward(self, x):
        ret = gru(x, getattr(self, "_z", None),
                  self.params.w,
                  self.params.u,
                  self.params.b)
        ret._count = self._count
        self._count += 1
        self._z = ret
        return ret

    def truncate(self):
        self._count = 0
        """Truncates temporal connection."""
        self._z = None
        self._state = None

class GruSplitMemory(Gru):

    def __init__(self, output_size, num_units, input_size=None, initializer=GlorotNormal()):
        self._num_units = num_units
        print ("Initializing Split Weights GRU model with {:d} output size and {:d} units.".format(output_size,num_units))
        super(GruSplitMemory, self).__init__(output_size,input_size,initializer)

    def weight_initiallize(self, size_i):
        size_i = size_i[0]
        size_o = self._size_o
        num_units = self._num_units
        bias = np.zeros((1, size_o * 3 * num_units), dtype=precision)
        bias[:, :] = 1
        # At this point, all connected units in the same layer will use the SAME weights
        self.params = {
            "w": Variable(self._initializer((size_i, size_o * 3 * num_units)), auto_update=True),
            "u": Variable(self._initializer((1, size_o * 3 * num_units)), auto_update=True),
            "b": Variable(bias, auto_update=True),
        }

    def forward(self, x):
        num_units = self._num_units
        size_o = self._size_o
        assert x.shape[0] == num_units, "There should be at least one input per unit"

        for i in range(num_units-1):
            ret = gru(x[np.newaxis,i], None if i == 0 else ret,
                    self.params.w[:,i * num_units:i * num_units + size_o * 3],
                    self.params.u[:,i * num_units:i * num_units + size_o * 3],
                    self.params.b[:,i * num_units:i * num_units + size_o * 3])


        return ret

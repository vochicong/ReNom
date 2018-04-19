#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.layers.activation.sigmoid import sigmoid
from renom.layers.activation.tanh import tanh
from renom.core import Node, Variable, get_gpu, precision, GPUValue, GetItem
from renom.operation import dot, sum, concat
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
        hminus = Variable(np.zeros((x.shape[0], w.shape[1] // 3), dtype=precision)) if pz is None else pz
        # Perform Forward Calcuations
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

    @classmethod
    def _oper_gpu(cls, x, pz, w, u, b):

        # Initialize Variables
        m = w.shape[1] // 3
        w_z, w_r, w_h = get_gpu(w).split([m, m * 2, ], axis=1)
        b_z, b_r, b_h = get_gpu(b).split([m, m * 2, ], axis=1)
        u_z, u_r, u_h = get_gpu(u).split([m, m * 2, ], axis=1)
        hminus = Variable(np.zeros((x.shape[0], m), dtype=precision)) if pz is None else pz
        hminus.to_gpu()
        x.to_gpu()
        w.to_gpu()
        b.to_gpu()
        u.to_gpu()

        # Perform Forward Calcuations
        input = dot(x,w) + b
        AB = input[:,:m*2] + concat([hminus,hminus],axis=1)*u[:,:m*2]
        A = AB[:,:m]
        B = AB[:,m:m*2]
        sigAB = sigmoid(AB[:,:m*2])
        sigB = sigAB[:,m:m*2]
        C = input[:,m*2:m*3] + hminus*u[:,m*2:m*3]*sigB



        h = sigAB[:,:m] + tanh(C)
        #h = sigAB[:,:m] + tanh(C)
        h.to_cpu()

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
        ret.attrs._sigB = sigB

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
        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, dpz)

    def _backward_gpu(self, context, dy, **kwargs):

        #print ('Current depth is: {:d}'.format(self._count))

        x =         get_gpu(self.attrs._x)
        w_z =       get_gpu(self.attrs._w_z)
        w_r =       get_gpu(self.attrs._w_r)
        w_h =       get_gpu(self.attrs._w_h)
        A =         get_gpu(self.attrs._A)
        B =         get_gpu(self.attrs._B)
        C =         get_gpu(self.attrs._C)
        u_z =       get_gpu(self.attrs._u_z)
        u_h =       get_gpu(self.attrs._u_h)
        u_r =       get_gpu(self.attrs._u_r)
        sigB =      get_gpu(self.attrs._sigB)
        hminus =    get_gpu(self.attrs._pz)

        y = get_gpu(dy)


        print (self._create_node(hminus))
        cu.cugru_backward(A, B, C)
        #dA = get_gpu(sigmoid_diff(A))
        #dB = get_gpu(sigmoid_diff(B))
        #dC = get_gpu(tanh_diff(C))
        dA = A
        dB = B
        dC = C


        ydA = y * dA
        ydC = y * dC
        ydCBuh = ydC * dB * u_h * hminus

        yconc = concat([ydA, ydCBuh, ydC],axis=1)

        # Calculate dx
        dx_z = get_gpu(dot(ydA, w_z.T))
        dx_r = get_gpu(dot(ydCBuh, w_r.T))
        dx_h = get_gpu(dot(ydC, w_h.T))
        dx = dx_z + dx_r + dx_h

        # Calculate dw
        xconc = concat([x.T for _ in range(3)],axis=1)
        dw = dot(xconc,yconc)

        # Calculate db
        db_z = sum(ydA, axis=0)  # , keepdims=True)
        db_r = sum(ydCBuh, axis=0)  # , keepdims=True)
        db_h = sum(ydC, axis=0)  # , keepdims=True)
        db = concat([db_z, db_r, db_h], axis=1)
        #print ('Updating b with: {},{},{}'.format(db_z,db_r,db_h))

        du_z = sum(ydA * hminus, axis=0)  # , keepdims=True)
        du_r = sum(ydCBuh * hminus, axis=0)  # , keepdims=True)
        du_h = sum(ydC * sigB * hminus, axis=0)  # , keepdims=True)
        du = concat([du_z, du_r, du_h], axis=1)

        pz_z = ydA * u_z
        pz_r = ydCBuh * u_r
        pz_h = ydC * sigB * u_h

        dpz = pz_z + pz_r + pz_h


        self.attrs._x._update_diff(context, dx)
        self.attrs._w._update_diff(context, dw)
        self.attrs._b._update_diff(context, db)
        self.attrs._u._update_diff(context, du)
        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, dpz)
'''
Basic Implementations

    @classmethod
    def _oper_gpu(cls, x, pz, w, u, b):

        # Initialize Variables
        m = w.shape[1] // 3
        w_z, w_r, w_h = get_gpu(w).split([m, m * 2, ], axis=1)
        b_z, b_r, b_h = get_gpu(b).split([m, m * 2, ], axis=1)
        u_z, u_r, u_h = get_gpu(u).split([m, m * 2, ], axis=1)
        hminus = Variable(np.zeros((x.shape[0], w.shape[1] // 3), dtype=precision)) if pz is None else pz
        hminus.to_gpu()
        x.to_gpu()
        # Perform Forward Calcuations
        A = dot(x, w_z) + hminus._gpu * u_z + b_z
        B = dot(x, w_r) + u_r * hminus._gpu + b_r
        sigB = sigmoid(B)
        C = dot(x, w_h) + sigB * u_h * hminus._gpu + b_h
        h = sigmoid(A) + tanh(C)
        h.to_cpu()

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
        ret.attrs._sigB = sigB

        return ret

    def _backward_gpu(self, context, dy, **kwargs):

        #print ('Current depth is: {:d}'.format(self._count))

        x =         get_gpu(self.attrs._x)
        w_z =       get_gpu(self.attrs._w_z)
        w_r =       get_gpu(self.attrs._w_r)
        w_h =       get_gpu(self.attrs._w_h)
        A =         get_gpu(self.attrs._A)
        B =         get_gpu(self.attrs._B)
        C =         get_gpu(self.attrs._C)
        u_z =       get_gpu(self.attrs._u_z)
        u_h =       get_gpu(self.attrs._u_h)
        u_r =       get_gpu(self.attrs._u_r)
        hminus =    get_gpu(self.attrs._pz)

        y = get_gpu(dy)

        #A = get_gpu(A)
        #cu.cugru_backward(A)
        #dA = A
        dA = get_gpu(sigmoid_diff(A))
        dB = get_gpu(sigmoid_diff(B))
        dC = get_gpu(tanh_diff(C))

        # Calculate dx
        dx_z = get_gpu(dot(y * dA, w_z.T))
        dx_r = get_gpu(dot(y * dB * dC * u_h * hminus, w_r.T))
        dx_h = get_gpu(dot(y * dC, w_h.T))
        dx = dx_z + dx_r + dx_h
        v = self._create_node(dx)

        # Calculate dw
        dw_z = dot(x.T, y * dA)
        dw_r = dot(x.T, y * dB * dC * u_h * hminus)
        dw_h = dot(x.T, y * dC)
        dw = concat([dw_z, dw_r, dw_h], axis=1)

        # Calculate db
        db_z = sum(y * dA, axis=0)  # , keepdims=True)
        db_r = sum(y * dB * dC * u_h * hminus, axis=0)  # , keepdims=True)
        db_h = sum(y * dC, axis=0)  # , keepdims=True)
        db = concat([db_z, db_r, db_h], axis=1)
        #print ('Updating b with: {},{},{}'.format(db_z,db_r,db_h))

        du_z = sum(dA * hminus * y, axis=0)  # , keepdims=True)
        du_r = sum(y * dC * dB * u_h * hminus * hminus, axis=0)  # , keepdims=True)
        du_h = sum(sigmoid(B) * dC * y * hminus, axis=0)  # , keepdims=True)
        du = concat([du_z, du_r, du_h], axis=1)

        pz_z = y * dA * u_z
        pz_r = y * dC * dB * u_h * hminus * u_r
        pz_h = y * dC * get_gpu(sigmoid(B)) * u_h

        dpz = pz_z + pz_r + pz_h
        v = self._create_node(dpz)


        self.attrs._x._update_diff(context, dx)
        self.attrs._w._update_diff(context, dw)
        self.attrs._b._update_diff(context, db)
        self.attrs._u._update_diff(context, du)
        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, dpz)
'''


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
        #print ('Currently at depth {:d}'.format(self._count))
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

# Encapsulates a single unit in a ReNom model


class GruSimpleUnit(Parametrized):

    def __init__(self, output_size, input_size=None, initializer=GlorotNormal()):
        self._size_o = output_size
        self._initializer = initializer
        super(GruSimpleUnit, self).__init__(input_size)

    def weight_initiallize(self, size_i):
        size_i = size_i[0]
        size_o = self._size_o
        bias = np.zeros((1, size_o * 3), dtype=precision)
        bias[:, :] = 1
        # At this point, all connected units in the same layer will use the SAME weights
        self.params = {
            "w": Variable(self._initializer((size_i, size_o * 3)), auto_update=True),
            "u": Variable(self._initializer((1, size_o * 3)), auto_update=True),
            "b": Variable(bias, auto_update=True),
        }

    def forward(self, x, pz=None):
        ret = gru(x, pz,
                  self.params.w,
                  self.params.u,
                  self.params.b)
        return ret

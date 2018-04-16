#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.layers.activation.sigmoid import sigmoid
from renom.layers.activation.tanh import tanh
from renom.core import Node, Variable, get_gpu, precision, GPUValue, GetItem
from renom.operation import dot, sum
from renom.utility.initializer import GlorotNormal
from .parameterized import Parametrized
from renom.cuda import cuda as cu

class test_unit(Node):
    '''
    @ parameters
    cls: The self variable required by python. Indicates the node itself.
    x: The input variable. This is what the users wishes to train.
    w: The weights with which to perform the calculations on.
    b: The biases with which to perform the calculations on.
    '''
    def __new__(cls, x, w, b):
        return cls.calc_value(x, w, b)

    @classmethod
    def _oper_cpu(cls, x, w, b):

        # Initialize Variables

        # Perform Forward Calcuations
        # Pay attention to the ordering of the dot arguments.
        h = dot(x,w) + b

        # Store Variables for the graph.
        # If this is not done, the variables used by this node will not be visible.
        ret = cls._create_node(h)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b

        return ret

    def _backward_cpu(self, context, dy, **kwargs):

        # Retreive previously stored variables
        x = self.attrs._x
        w = self.attrs._w
        b = self.attrs._b
        y = dy

        # Remember to find the differential for dx as well as dw and db
        dx = dot(y, w.T)
        dw = dot(x.T,y)
        db = np.sum(dy,axis=0)

        self.attrs._x._update_diff(context, dx)
        self.attrs._w._update_diff(context, dw)
        self.attrs._b._update_diff(context, db)



class TestUnit(Parametrized):

    def __init__(self, output_size, input_size=None, initializer=GlorotNormal()):
        self._size_o = output_size
        self._initializer = initializer
        super(TestUnit, self).__init__(input_size)

    def weight_initiallize(self, size_i):
        size_i = size_i[0]
        size_o = self._size_o
        bias = np.zeros((1, size_o), dtype=precision)
        bias[:, :] = 1
        # Use count to determine the depth of the current node
        self._count = 0
        # At this point, all connected units in the same layer will use the SAME weights
        self.params = {
            "w": Variable(self._initializer((size_i, size_o)), auto_update=True),
            "b": Variable(bias, auto_update=True),
        }

    def forward(self, x):
        return test_unit(x,self.params.w,self.params.b)

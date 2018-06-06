#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from renom.layers.function.utils import imncol
from renom.core import Node, Variable, to_value, GPUValue, get_gpu, precision
from .parameterized import Parametrized
from renom.utility.initializer import Gaussian
from renom.cuda import cuda as cu


class convnd(Node):

    def __new__(cls, x, w, b, filter=3, stride=1, padding=0):
        in_shape = x.shape[1:]
        out_shape = [w.shape[0], *[(x.shape[2] + padding * 2 - filter)// stride + 1 for _ in range(len(x.shape[2:]))]]
        return cls.calc_value(x, w, b, in_shape, out_shape, filter, stride, padding)

    @classmethod
    def _oper_cpu(cls, x, w, b, in_shape, out_shape, kernel, stride, padding):
        col = imncol(to_value(x), out_shape, w, b, stride, padding)
        return cls._create_node(col)
        pass

    @classmethod
    def _oper_gpu(cls, x, w, b, in_shape, out_shape, kernel, stride, padding):
        pass

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        pass


class ConvNd(Parametrized):
    """2d convolution layer.

    This class creates a convolution filter to be convolved with
    the input tensor.
    The instance of this class only accepts and outputs 4d tensors.

    At instantiation, in the case of int input, filter, padding, and stride, the shape will be symmetric.

    If the argument `input_size` is passed, this layers' weight is initialized
    in the __init__ function.
    Otherwise, the weight is initialized in its first forward calculation.

    Args:
        channel (int): The dimensionality of the output.
        filter (tuple,int): Filter size of the convolution kernel.
        padding (tuple,int): Size of the zero-padding around the image.
        stride (tuple,int): Stride-size of the convolution.
        input_size (tuple): Input unit size. This must be a tuple like (Channel, Height, Width).
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> n, c, h, w = (10, 3, 32, 32)
        >>> x = np.random.rand(n, c, h, w)
        >>> x.shape
        (10, 3, 32, 32)
        >>> layer = rm.Conv2d(channel=32)
        >>> z = layer(x)
        >>> z.shape
        (10, 32, 30, 30)

    Note:
        Tensor data format is **NCHW**.
    """

    def __init__(self, channel=4, filter=3, padding=0, stride=1, input_size=None, initializer=Gaussian()):
        self._padding = padding
        self._stride = stride
        self._kernel = filter
        self._channel = channel
        self._initializer = initializer
        super(ConvNd, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        # The first dimension is to allow different types of uncorrelated images as inputs, such as RGB information.
        # After this dimension, the image data is assumed to be meaningfully correlated.
        self._dims = len(input_size[1:])
        print(self._dims)
        kern = [self._kernel for _ in range(self._dims)]
        size_f = (self._channel, input_size[0], *kern)
        self.params = {"w": Variable(self._initializer(size_f), auto_update=True),
                       "b": Variable(np.zeros((1, self._channel, 1, 1), dtype=precision), auto_update=True)}

    def forward(self, x):
        return convnd(x, self.params["w"], self.params["b"], self._kernel,
                      self._stride, self._padding)

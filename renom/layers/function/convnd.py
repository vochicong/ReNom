#!/usr / bin / env python
# encoding: utf - 8

import numpy as np
from renom.layers.function.utils import imncol, colnim, pad_dx, pad_image, colnw
from renom.core import Node, Variable, to_value, GPUValue, get_gpu, precision
from .parameterized import Parametrized
from renom.utility.initializer import Gaussian
from renom.cuda import cuda as cu
from renom.cuda import is_cuda_active


class convnd(Node):

    def __new__(cls, x, w, b, filter=3, stride=1, padding=0):
        in_shape = x.shape[1:]
        return cls.calc_value(x, w, b, in_shape, filter, stride, padding)


    @classmethod
    def _oper_cpu(cls, x, w, b, in_shape, kernel, stride, padding):
        col = imncol(to_value(x), w, stride, padding)
        ret = cls._create_node(col + b)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, w, b, in_shape, kernel, stride, padding):
        conv_desc = cu.ConvolutionNDescriptor(padding, stride, precision)
        filter_desc = cu.NdFilterDescriptor(w.shape, precision)

        output_shape = [x.shape[0], w.shape[0]]
        for i in range(len(x.shape[2:])):
            output_shape.append((x.shape[i + 2] + padding[i] * 2 - kernel[i]) // stride[i] + 1)
        y = GPUValue(shape=tuple(output_shape))

        with cu.cudnn_handler() as handle:
            cu.cuConvolutionForward(handle, conv_desc, filter_desc, x, w, y)
            if b is not None:
                cu.cu_add_bias(get_gpu(b), y)

        # assert type(x) is not np.ndarray

        ret = cls._create_node(y)
        ret.attrs._conv_desc = conv_desc
        ret.attrs._filter_desc = filter_desc
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        dx = colnim(dy, self.attrs._w, self.attrs._stride)
        dw = colnw(self.attrs._x, dy, self.attrs._stride)
        db = np.sum(dy, axis=tuple(
            [0, ] + [i for i in range(2, len(self.attrs._b.shape))]), keepdims=True)
        self.attrs._x._update_diff(context, dx)
        self.attrs._w._update_diff(context, dw)
        self.attrs._b._update_diff(context, db)

    def _backward_gpu(self, context, dy, **kwargs):
        dw, db, dx = (get_gpu(g).empty_like_me()
                      for g in (self.attrs._w, self.attrs._b, self.attrs._x))

        with cu.cudnn_handler() as handle:
            cu.cuConvolutionBackward(handle, self.attrs._conv_desc, self.attrs._filter_desc,
                                     self.attrs._x, self.attrs._w, dy, dw, db, dx, **kwargs)
        if isinstance(self.attrs._w, Node):
            self.attrs._w._update_diff(context, dw, **kwargs)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx, **kwargs)

        if isinstance(self.attrs._b, Node):
            self.attrs._b._update_diff(context, db, **kwargs)


class ConvNd(Parametrized):
    """Nd convolution layer.

    This class creates a convolution filter to be convolved with
    the input tensor.
    The instance of this class accepts tensors of any dimensionality and produces an output of equal
    dimensionality as the input

    At instantiation, in the case of int input, filter, padding, and stride, the shape will be symmetric.

    If the argument `input_size` is passed, this layers' weight is initialized
    in the __init__ function.
    Otherwise, the weight is initialized in its first forward calculation.

    Args:
        channel (int): The dimensionality of the output.
        filter int: Filter size of the convolution kernel.
        padding int: Size of the zero - padding around the image.
        stride int: Stride - size of the convolution.
        input_size (tuple): Input unit size. This must be a tuple like (Channel, Height, Width).
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> n, c, h, w = (10, 3, 32, 32)
        >>> x = np.random.rand(n, c, h, w)
        >>> x.shape
        (10, 3, 32, 32)
        >>> layer = rm.ConvNd(channel=32)
        >>> z = layer(x)
        >>> z.shape
        (10, 32, 30, 30)

    Note:
        Tensor data format is **NC(D*)**.
    """

    def __init__(self, channel=2, filter=3, padding=0, stride=1, input_size=None, initializer=Gaussian()):
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
        if is_cuda_active():
            assert self._dims < 4, "GPU Version currently only supports up to 3 dimensions"
        kern = [self._kernel for _ in range(self._dims)]
        self._kernel = np.array(kern)
        self._padding = np.array([self._padding for _ in range(self._dims)], dtype=np.int32)
        self._stride = np.array([self._stride for _ in range(self._dims)], dtype=np.int32)
        f_lst = [self._channel, input_size[0]]
        f_lst.extend(kern)
        size_f = tuple(f_lst)
        size_b = tuple([1, self._channel] + [1 for _ in range(self._dims)])

        self.params = {"w": Variable(self._initializer(size_f), auto_update=True),
                       "b": Variable(np.ones(size_b, dtype=precision), auto_update=True)}

    def forward(self, x):
        return convnd(x, self.params["w"], self.params["b"], self._kernel,
                          self._stride, self._padding)


class Conv3d(Parametrized):

    '''
    Provides an interface for the ConvNd with a more familiar name

    Note:
        Tensor data format is **NCHWD**.
    '''

    def __init__(self, channel=2, filter=3, padding=0, stride=1, input_size=None, initializer=Gaussian()):
        self._padding = padding
        self._stride = stride
        self._kernel = filter
        self._channel = channel
        self._initializer = initializer
        super(Conv3d, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        # The first dimension is to allow different types of uncorrelated images as inputs, such as RGB information.
        # After this dimension, the image data is assumed to be meaningfully correlated.
        self._dims = len(input_size[1:])
        assert self._dims < 4, "Conv3D expects up to 3 dimensions"
        if not isinstance(self._padding, np.ndarray):
            self._padding = np.array(
                tuple([self._padding for _ in range(self._dims)]), dtype=np.int32)
            self._stride = np.array(
                tuple([self._stride for _ in range(self._dims)]), dtype=np.int32)
            self._kernel = np.array(
                tuple([self._kernel for _ in range(self._dims)]), dtype=np.int32)
        f_lst = [self._channel, input_size[0]]
        f_lst.extend(list(self._kernel))
        size_f = tuple(f_lst)
        size_b = tuple([1, self._channel] + [1 for _ in range(self._dims)])

        self.params = {"w": Variable(self._initializer(size_f), auto_update=True),
                       "b": Variable(np.ones(size_b, dtype=precision), auto_update=True)}

    def forward(self, x):
        return convnd(x, self.params["w"], self.params["b"], self._kernel,
                            self._stride, self._padding)

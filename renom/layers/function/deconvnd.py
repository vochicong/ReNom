#!/usr / bin / env python
# encoding: utf - 8

import numpy as np
from renom.layers.function.utils import imncol, colnim, pad_dx, pad_image, colnw
from renom.core import Node, Variable, to_value
from renom import precision
from .parameterized import Parametrized
from renom.utility.initializer import Gaussian
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu
from renom.cuda import is_cuda_active


class deconvnd(Node):

    def __new__(cls, x, w, b, filter=3, stride=1, padding=0):
        in_shape = x.shape[1:]
        return cls.calc_value(x, w, b, in_shape, filter, stride, padding)

    @classmethod
    def _oper_cpu(cls, x, w, b, in_shape, kernel, stride, padding):
        col = colnim(x, w, stride)
        if b is not None:
            col += b
        ret = cls._create_node(col)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, w, b, in_shape, kernel, stride, padding):
        raise NotImplementedError

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            dx = imncol(dy, self.attrs._w, self.attrs._stride, padding=[
                        0 for _ in range(len(self.attrs._stride))])
            self.attrs._x._update_diff(context, dx)

    def _backward_gpu(self, context, dy, **kwargs):
        raise NotImplementedError


def check_input(var, length):
    if isinstance(var, tuple):
        assert len(var) is length
        var = list(var)
    elif not isinstance(var, np.ndarray):
        var = np.array(
            tuple([var for _ in range(length)]), dtype=np.int32)
    elif not var.dtype == np.int32:
        var = var.astype(np.int32)
    assert len(var) is length
    return var


class DeconvNd(Parametrized):
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
        filter (int): Filter size of the convolution kernel.
        padding (int): Size of the zero - padding around the image.
        stride (int): Stride - size of the convolution.
        input_size (tuple): Input unit size. This must be a tuple like (Channel, Height, Width).
        ignore_bias (bool): If `True` is given, bias will not be added.
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

    def __init__(self, channel=2, filter=3, padding=0, stride=1,
                 input_size=None, ignore_bias=False, initializer=Gaussian()):
        self._padding = padding
        self._stride = stride
        self._kernel = filter
        self._channel = channel
        self._initializer = initializer
        self._ignore_bias = ignore_bias
        super(DeconvNd, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        # The first dimension is to allow different types of uncorrelated images as inputs, such as RGB information.
        # After this dimension, the image data is assumed to be meaningfully correlated.
        self._dims = len(input_size[1:])
        if is_cuda_active():
            assert self._dims < 4, "GPU Version currently only supports 2 and 3 dimensions"

        def func(var):
            return check_input(var, self._dims)
        self._kernel, self._padding, self._stride = map(
            func, [self._kernel, self._padding, self._stride])

        assert all([s > 0 for s in input_size[1:]]), \
            "The shape of input array {} is too small. Please give an array which size is lager than 0.".format(
                input_size[1:])

        f_lst = [self._channel, input_size[0]]
        f_lst.extend(self._kernel)
        size_f = tuple(f_lst)
        size_b = tuple([1, self._channel] + [1 for _ in range(self._dims)])

        self.params = {"w": Variable(self._initializer(size_f), auto_update=True)}
        if not self._ignore_bias:
            self.params["b"] = Variable(np.ones(size_b, dtype=precision), auto_update=True)

    def forward(self, x):
        assert len(
            x.shape) > 2, "The dimension of input array must be grater than 3. Actual dim is {}".format(x.ndim)
        assert all([s > 0 for s in x.shape[2:]]), \
            "The shape of input array {} is too small. Please give an array which size is lager than 0.".format(
                x.shape)

        return deconvnd(x, self.params["w"], self.params.get("b", None), self._kernel,
                        self._stride, self._padding)

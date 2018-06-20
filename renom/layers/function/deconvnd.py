
import numpy as np
from renom.layers.function.utils import deimncol, imncol
from renom.core import Node, Variable, to_value, GPUValue, get_gpu, precision
from .parameterized import Parametrized
from renom.utility.initializer import Gaussian, GlorotNormal
from renom.cuda import cuda as cu
from renom.cuda import is_cuda_active

class deconvnd(Node):

    def __new__(cls, x, w, b, filter=3, stride=1, padding=0):
        return cls.calc_value(x, w, b, filter, stride, padding)

    @classmethod
    def _oper_cpu(cls, x, w, b, kernel, stride, padding):
        ret = deimncol(x, w, stride)
        ret = cls._create_node(ret + b)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, w, b, kernel, stride, padding):
        pass

    def _backward_cpu(self, context, dy, **kwargs):
        dx = imncol(dy, self.attrs._w, self.attrs._stride, self.attrs._padding)
        dw = imncol(dy, self.attrs._x, self.attrs._stride, self.attrs._padding)
        db = np.sum(dy, axis=tuple(
            [0, ] + [i for i in range(2, len(self.attrs._b.shape))]), keepdims=True)
        self.attrs._x._update_diff(context, dx, **kwargs)
        self.attrs._w._update_diff(context, dw, **kwargs)
        self.attrs._b._update_diff(context, db, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        pass

class DeconvNd(Parametrized):
    '''2d convolution layer.

    This class creates a convolution filter to be convolved with
    the input tensor.
    The instance of this class only accepts and outputs 4d tensors.

    At instantiation, in the case of int input, filter, padding, and stride, the shape will be symmetric.

    If the argument `input_size` is passed, this layers' weight is initialized
    in the __init__ function.
    Otherwise, the weight is initialized in its first forward calculation.

    Args:
        channel (int): The dimensionality of the output.
        filter (tuple,int): Filter size to witch used as convolution kernel.
        padding (tuple,int): Pad around image by 0 according to this size.
        stride (tuple,int): Specifying the strides of the convolution.
        input_size (tuple): Input unit size. This must be a tuple like (Channel, Height, Width).
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> n, c, h, w = (10, 3, 32, 32)
        >>> x = np.random.rand(n, c, h, w)
        >>> x.shape
        (10, 3, 32, 32)
        >>> layer = rm.Deconv2d(channel=32)
        >>> z = layer(x)
        >>> z.shape
        (10, 32, 34, 34)

    '''

    def __init__(self, channel=1, filter=3, padding=0, stride=1, input_size=None, initializer=GlorotNormal()):
        self._padding = padding
        self._stride = stride
        self._kernel = filter
        self._channel = channel
        self._initializer = initializer
        super(DeconvNd, self).__init__(input_size)

    def weight_initiallize(self, input_size):
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
                       #"w": Variable(np.ones(size_f, dtype=precision), auto_update=True),
                       "b": Variable(np.ones(size_b, dtype=precision), auto_update=True)}

    def forward(self, x):
        return deconvnd(x, self.params["w"], self.params["b"],
                        self._kernel, self._stride, self._padding)

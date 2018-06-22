
import numpy as np
from renom.layers.function.utils import colnim, imncol, colnw, imnw
from renom.core import Node, Variable, to_value, GPUValue, get_gpu, precision
from .parameterized import Parametrized
from renom.utility.initializer import Gaussian, GlorotNormal
from renom.cuda import cuda as cu
from renom.cuda import is_cuda_active

class SimpleContainer(object):
    def __init__(self, item):
        self._item = item

class deconvnd(Node):

    def __new__(cls, x, w, b, prev_conv):
        return cls.calc_value(x, w, b, prev_conv._item)

    @classmethod
    def _oper_cpu(cls, x, w, b, prev_conv):
        dx = colnim(x, w, prev_conv.attrs._stride)
        ret = cls._create_node(dx + b)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._prev = prev_conv
        return ret

    @classmethod
    def _oper_gpu(cls, x, w, b, prev_conv):
        dx = get_gpu(prev_conv.attrs._x).empty_like_me()

        #with cu.cudnn_handler() as handle:
        #    cu.cuConvolutionBackward(handle, self.attrs._conv_desc, self.attrs._filter_desc,
        #                             self.attrs._x, self.attrs._w, dy, dw, db, dx, **kwargs)

        with cu.cudnn_handler() as handle:
            cu.cuConvolutionBackwardData(handle, prev_conv.attrs._conv_desc, prev_conv.attrs._filter_desc,
                get_gpu(w), get_gpu(x), dx)
            cu.cu_add_bias(get_gpu(b), dx)

        ret = cls._create_node(dx)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._prev = prev_conv
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        dx = imncol(dy, self.attrs._w, self.attrs._prev.attrs._stride, self.attrs._prev.attrs._padding)
        dw = imncol(np.swapaxes(dy,0,1), np.swapaxes(self.attrs._x,0,1), self.attrs._prev.attrs._stride, self.attrs._prev.attrs._padding)
        db = np.sum(dy, axis=tuple(
            [0, ] + [i for i in range(2, len(self.attrs._b.shape))]), keepdims=True)
        dw = np.swapaxes(dw,0,1)
        self.attrs._x._update_diff(context, dx, **kwargs)
        self.attrs._w._update_diff(context, dw, **kwargs)
        self.attrs._b._update_diff(context, db, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        dy = get_gpu(dy)
        prev_attrs = self.attrs._prev.attrs
        prev_x = self.attrs._x
        dx = get_gpu(self.attrs._x).empty_like_me()
        dw = get_gpu(self.attrs._w).empty_like_me()
        db = get_gpu(self.attrs._b).empty_like_me()
        swapped_x_shape = tuple([dx.shape[1],dx.shape[0]] + list(dx.shape[2:]))
        swapped_y_shape = tuple([dy.shape[1],dy.shape[0]] + list(dy.shape[2:]))
        swapped_w_shape = tuple([dw.shape[1],dw.shape[0]] + list(dw.shape[2:]))
        swapped_x = GPUValue(shape=swapped_x_shape)
        swapped_y = GPUValue(shape=swapped_y_shape)
        swapped_w = GPUValue(shape=swapped_w_shape)
        for i in range(dy.shape[1]):
            swapped_y[i,...] = dy[:,i,...]
        for i in range(dx.shape[1]):
            swapped_x[i,...] = prev_x[:,i,...]
        with cu.cudnn_handler() as handle:
            cu.cuConvolutionForward(handle, prev_attrs._conv_desc, prev_attrs._filter_desc, get_gpu(dy), self.attrs._w, dx)
            #x_filter_descriptor = cu.NdFilterDescriptor(self.attrs._x.shape, precision)
            #cu.cuConvolutionForward(handle, prev_attrs._conv_desc, x_filter_descriptor, get_gpu(dy), get_gpu(self.attrs._x), dw)
            x_filter_descriptor = cu.NdFilterDescriptor(swapped_x.shape, precision)
            cu.cuConvolutionForward(handle, prev_attrs._conv_desc, x_filter_descriptor, swapped_y, swapped_x, swapped_w)
            cu.cuConvolutionBackwardBias(handle, dy, db)
        for i in range(dw.shape[1]):
            dw[i,...] = swapped_w[:,i,...]
        self.attrs._x._update_diff(context, dx, **kwargs)
        self.attrs._w._update_diff(context, dw, **kwargs)
        self.attrs._b._update_diff(context, db, **kwargs)

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

    def __init__(self, initializer=Gaussian()):
        self._initializer = initializer
        super(DeconvNd, self).__init__()

    def weight_initiallize(self, input_size, prev_conv = None):
        if prev_conv is None:
            return
        size_f = prev_conv.attrs._w.shape
        size_b = np.array([1 for _ in range(len(size_f))])
        size_b[1] = prev_conv.attrs._x.shape[1]
        self.params = {"w": Variable(self._initializer(size_f), auto_update=True),
                       "b": Variable(np.ones(size_b, dtype=precision), auto_update=True)}

    def forward(self, x, prev_conv = None):
        if not "w" in self.params:
            self.weight_initiallize(0, prev_conv)
        return deconvnd(x, self.params["w"], self.params["b"],SimpleContainer(prev_conv))

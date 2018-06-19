
class deconvnd(Node):

    def __new__(cls, x, w, b, filter=3, stride=1, padding=0):
        pass

    @classmethod
    def _oper_cpu(cls, x, w, b, in_shape, out_shape, kernel, stride, padding):
        pass

    @classmethod
    def _oper_gpu(cls, x, w, b, in_shape, out_shape, kernel, stride, padding):
        pass

    def _backward_cpu(self, context, dy, **kwargs):
        pass

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
        self._padding, self._stride, self._kernel = (padding, stride, filter)
        self._channel = channel
        self._initializer = initializer
        super(DeconvNd, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        size_f = (input_size[0], self._channel,
                  self._kernel[0], self._kernel[1])
        self.params = {"w": Variable(self._initializer(size_f), auto_update=True),
                       "b": Variable(np.zeros((1, self._channel, 1, 1), dtype=precision), auto_update=True)}

    def forward(self, x):
        return deconvnd(x, self.params["w"], self.params["b"],
                        self._kernel, self._stride, self._padding)

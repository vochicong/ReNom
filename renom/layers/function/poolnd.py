

import numpy as np
from renom.core import Node, GPUValue, get_gpu
from renom.layers.function.utils import imnpool, poolnim
from renom.cuda import cuda as cu


class npool_base(Node):

    def __new__(cls, x, kernel, stride, padding):
        return cls.calc_value(x, kernel, stride, padding)

    def _backward_gpu(self, context, dy, **kwargs):
        dx = get_gpu(self.attrs._x).empty_like_me()
        with cu.cudnn_handler() as handle:
            cu.cuPoolingBackward(handle, self.attrs._pool_desc, self.attrs._x, self, dy, dx)
        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx, **kwargs)


class max_poolnd(npool_base):

    @classmethod
    def _oper_cpu(cls, x,kernel, stride, padding):
        result = imnpool(x, kernel, stride, padding, mode = "max")
        ret = cls._create_node(result)
        ret.attrs._x = x
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, karnel, stride, padding):
        N = x.shape[0]
        pool_desc = cu.PoolingNDescriptor(karnel, padding, stride, pool_mode=0)
        out_shape = (np.array(x.shape[2:]) - np.array(karnel)) // np.array(stride) + 1
        y = GPUValue(shape=tuple([N,x.shape[1] ] + list(out_shape)))
        with cu.cudnn_handler() as handle:
            cu.cuPoolingForward(handle, pool_desc, x, y)
        ret = cls._create_node(y)
        ret.attrs._pool_desc = pool_desc
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        result = poolnim(self.attrs._x, dy, self.attrs._kernel, self.attrs._stride, mode = "max")
        self.attrs._x._update_diff(context, result, **kwargs)

class average_poolnd(npool_base):

    @classmethod
    def _oper_cpu(cls, x, kernel, stride, padding):
        result = imnpool(x, kernel, stride, padding, mode = "average")
        ret = cls._create_node(result)
        ret.attrs._x = x
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, karnel, stride, padding):
        N = x.shape[0]
        pool_desc = cu.PoolingNDescriptor(karnel, padding, stride, pool_mode=1)
        out_shape = (np.array(x.shape[2:]) - np.array(karnel)) // np.array(stride) + 1
        y = GPUValue(shape=tuple([N,x.shape[1] ] + list(out_shape)))
        with cu.cudnn_handler() as handle:
            cu.cuPoolingForward(handle, pool_desc, x, y)
        ret = cls._create_node(y)
        ret.attrs._pool_desc = pool_desc
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        dx = poolnim(self.attrs._x, dy, self.attrs._kernel, self.attrs._stride, mode = "average")
        self.attrs._x._update_diff(context, dx, **kwargs)

class NPoolBase:

    def __init__(self, kernel=3, padding=0, stride=1):
        self._padding = padding
        self._stride = stride
        self._kernel = kernel

    def __call__(self, x):
        if not isinstance(self._padding, np.ndarray):
            self._padding = np.array(
                tuple([self._padding for _ in range(len(x.shape[2:]))]), dtype=np.int32)
            self._stride = np.array(
                tuple([self._stride for _ in range(len(x.shape[2:]))]), dtype=np.int32)
            self._kernel = np.array(
                tuple([self._kernel for _ in range(len(x.shape[2:]))]), dtype=np.int32)
        return self.forward(x)


class MaxPoolNd(NPoolBase):

    def forward(self, x):
        return max_poolnd(x, self._kernel, self._stride, self._padding)

class AveragePoolNd(NPoolBase):

    def forward(self, x):
        return average_poolnd(x, self._kernel, self._stride, self._padding)



import numpy as np
from renom.core import Node, GPUValue, get_gpu
from renom.layers.function.utils import imnpool, poolnim
from renom.cuda import cuda as cu


class npool_base(Node):

    def __new__(cls, x, kernel=3, stride=1, padding=0):
        in_shape = x.shape[1:]
        out_shape = [x.shape[1], ]
        for i in range(len(x.shape[2:])):
            out_shape.append((x.shape[i + 2] + padding[i] * 2 - kernel[i]) // stride[i] + 1)
        return cls.calc_value(x, in_shape, out_shape, kernel, stride, padding)

    def _backward_gpu(self, context, dy, **kwargs):
        dx = get_gpu(self.attrs._x).empty_like_me()
        with cu.cudnn_handler() as handle:
            cu.cuPoolingBackward(handle, self.attrs._pool_desc, self.attrs._x, self, dy, dx)
        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx, **kwargs)


class max_poolnd(npool_base):

    @classmethod
    def _oper_cpu(cls, x, in_shape, out_shape, kernel, stride, padding):
        result = imnpool(x, kernel[0], stride[0], padding[0])
        ret = cls._create_node(result[0])
        ret.attrs._x = x
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        ret.attrs._indices = result[1]
        return ret

    @classmethod
    def _oper_gpu(cls, x, in_shape, out_shape, karnel, stride, padding):
        N = x.shape[0]
        pool_desc = cu.PoolingNDescriptor(karnel, padding, stride, pool_mode=0)
        y = GPUValue(shape=tuple([N, ] + list(out_shape)))
        with cu.cudnn_handler() as handle:
            cu.cuPoolingForward(handle, pool_desc, x, y)
        ret = cls._create_node(y)
        ret.attrs._pool_desc = pool_desc
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        print(self.attrs._x)
        result = poolnim(self.attrs._x, dy, self.attrs._indices)
        print(dy)
        print(self.attrs._indices)
        self.attrs._x._update_diff(context, result, **kwargs)


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

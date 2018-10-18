import numpy as np
from renom.core import Node
from renom.layers.function.utils import imnpool, poolnim
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu
from renom.cuda import is_cuda_active


class SimpleContainer(object):
    def __init__(self, item):
        self._item = item


class max_unpoolnd(Node):

    def __new__(cls, x, prev_pool):
        return cls.calc_value(x, prev_pool._item)

    @classmethod
    def _oper_cpu(cls, x, prev_pool):
        result = poolnim(prev_pool.attrs._x, x, prev_pool.attrs._kernel, prev_pool.attrs._stride, prev_pool.attrs._padding, mode="max")
        ret = cls._create_node(result)
        ret.attrs._x = x
        ret.attrs._original_x = prev_pool.attrs._x
        ret.attrs._kernel = prev_pool.attrs._kernel
        ret.attrs._stride = prev_pool.attrs._stride
        ret.attrs._padding = prev_pool.attrs._padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, prev_pool):
        print(prev_pool.attrs._x.shape)
        #real_shape = (1, 1, 3, 3)
        dx = GPUValue(shape=prev_pool.attrs._x.shape)
        print(prev_pool.shape, x.shape)
        with cu.cudnn_handler() as handle:
            try:                                                         # y                 # dy        # x                          # dx
                cu.cuPoolingBackward(handle, prev_pool.attrs._pool_desc, get_gpu(prev_pool), get_gpu(x), get_gpu(prev_pool.attrs._x), dx)
            except:
                assert False
        ret = cls._create_node(dx)
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        dx = imnpool(self.attrs._original_x, self.attrs._kernel, self.attrs._stride, self.attrs._padding, mode="max", alternate_input=dy)
        self.attrs._x._update_diff(context, dx)

class average_unpoolnd(Node):

    def __new__(cls, x, prev_pool):
        return cls.calc_value(x, prev_pool._item)

    @classmethod
    def _oper_cpu(cls, x, prev_pool):
        result = poolnim(prev_pool.attrs._x, x, prev_pool.attrs._kernel, prev_pool.attrs._stride, mode="average")
        ret = cls._create_node(result)
        ret.attrs._x = x
        ret.attrs._original_x = prev_pool.attrs._x
        ret.attrs._kernel = prev_pool.attrs._kernel
        ret.attrs._stride = prev_pool.attrs._stride
        ret.attrs._padding = prev_pool.attrs._padding
        return ret


class MaxUnPoolNd:

    def __init__(self): pass

    def __call__(self, x, prev_pool):
        return self.forward(x, SimpleContainer(prev_pool))

    def forward(self, x, prev_pool):
        return max_unpoolnd(x, prev_pool)


class AverageUnPoolNd:

    def __init__(self): pass

    def __call__(self, x, prev_pool):
        return self.forward(x, SimpleContainer(prev_pool))

    def forward(self, x, prev_pool):
        return average_unpoolnd(x, prev_pool)

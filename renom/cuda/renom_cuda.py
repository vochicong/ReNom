'''
A module that handles methods that might want to use different modules for their implementation.

Could possibly be used to handle populating the renom.cuda namespace
'''
import numpy as np
import renom
from renom.cuda import thrust, cuda_base, cudnn, cublas

def do_cublas(gpu_value,transpose = False):
    ones_vector = None
    if not transpose:
        if ones_vector is None or len(ones_vector) < gpu_value.shape[0]:
            ones_vector = renom.core.GPUValue(shape=((gpu_value.shape[0],))).fill(1)
        ret = renom.core.GPUValue(shape=((1,gpu_value.shape[1],)))
    else:
        if ones_vector is None or len(ones_vector) < gpu_value.shape[1]:
            ones_vector = renom.core.GPUValue(shape=((gpu_value.shape[1],))).fill(1)
        ret = renom.core.GPUValue(shape=((gpu_value.shape[0],1,)))
    cublas.cublas_sum(gpu_value, ones_vector, ret, transpose)
    return ret

def cusum(gpu_value, axis=None, keepdims=False):
    can_use_cublas = False

    if len(gpu_value.shape) == 2 and (axis == (0,) or axis is None):
        pass#can_use_cublas = True

    if can_use_cublas:
        ret = do_cublas(gpu_value)
        if axis is None:
            ret = do_cublas(ret, True)
        return ret
    else:
        return thrust.cu_sum(gpu_value, axis, keepdims)

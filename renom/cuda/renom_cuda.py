'''
A module that handles methods that might want to use different modules for their implementation.

Could possibly be used to handle populating the cuda namespace.
'''
import renom
from renom.cuda import thrust, cuda_base, cudnn, cublas#, gpuvalue

def cusum(gpu_value, axis=None, keepdims=False):
    can_use_cublas = False

    if axis is None and len(gpu_value.shape) <= 2:
        can_use_cublas = True

    #if len(gpu_value.shape) > 1:
    #    ones_array = renom.core.GPUValue(shape=(gpu_value.shape[0],)).fill(1)
    #    ret = renom.core.GPUValue(shape=(gpu_value.shape[1],))
    #    cublas.cublas_sum(gpu_value, ones_array, ret)
    #    tmp_val = ret
    #    ones_array = renom.core.GPUValue(shape=(tmp_val.shape[0],)).fill(1)
    #    ret = renom.core.GPUValue(shape=(1,))
    #    cublas.cublas_sum(tmp_val, ones_array, ret)
    #else:
    #    ones_array = renom.core.GPUValue(shape=(gpu_value.shape[0],)).fill(1)
    #    ret = renom.core.GPUValue(shape=(1,))
    #    cublas.cublas_sum(gpu_value, ones_array, ret)

    #print("Summing")
    #print(gpu_value.new_array(), gpu_value.shape)
    #print(ret.new_array(), ret.shape)
    #ret2 = thrust.cu_sum(gpu_value, axis, keepdims)
    #print(ret2.new_array(), ret2.shape)
    #assert False

    if can_use_cublas:
        if len(gpu_value.shape) > 1:
            ones_array = renom.core.GPUValue(shape=(gpu_value.shape[0],)).fill(1)
            ret = renom.core.GPUValue(shape=(gpu_value.shape[1],))
            cublas.cublas_sum(gpu_value, ones_array, ret)
            tmp_val = ret
            ones_array = renom.core.GPUValue(shape=(tmp_val.shape[0],)).fill(1)
            ret = renom.core.GPUValue(shape=(1,))
            cublas.cublas_sum(tmp_val, ones_array, ret)
        else:
            ones_array = renom.core.GPUValue(shape=(gpu_value.shape[0],)).fill(1)
            ret = renom.core.GPUValue(shape=(1,))
            cublas.cublas_sum(gpu_value, ones_array, ret)
        return ret
    else:
        return thrust.cu_sum(gpu_value, axis, keepdims)

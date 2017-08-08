import bisect
cimport numpy as np
import numpy as pnp
from cuda_base import *
cimport cython
from numbers import Number
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t
from cuda_utils cimport _VoidPtr
from renom.config import precision

def cuMalloc(uintptr_t nbytes):
    cdef void * p
    runtime_check(cudaMalloc( & p, nbytes))
    return < uintptr_t > p


def cuMemset(uintptr_t ptr, int value, size_t size):
    p = <void * >ptr
    runtime_check(cudaMemset(p, value, size))
    return


def cuCreateStream():
    cdef cudaStream_t stream
    runtime_check(cudaStreamCreate( & stream))
    return < uintptr_t > stream


def cuSetDevice(int dev):
    runtime_check(cudaSetDevice(dev))
    return


def cuCreateCtx(device=0):
    cdef CUcontext ctx
    driver_check(cuCtxCreate( & ctx, 0, device))
    return int(ctx)


def cuGetDeviceCxt():
    cdef CUdevice device
    driver_check(cuCtxGetDevice( & device))
    return int(device)


def cuGetDeviceCount():
    cdef int count
    runtime_check(cudaGetDeviceCount( & count))
    return int(count)


def cuGetDeviceProperty(device):
    cdef cudaDeviceProp property
    runtime_check(cudaGetDeviceProperties( & property, device))
    property_dict = {
        "name": property.name,
        "totalGlobalMem": property.totalGlobalMem,
        "sharedMemPerBlock": property.sharedMemPerBlock,
        "regsPerBlock": property.regsPerBlock,
        "warpSize": property.warpSize,
        "memPitch": property.memPitch,
        "maxThreadsPerBlock": property.maxThreadsPerBlock,
        "maxThreadsDim": property.maxThreadsDim,
        "maxGridSize": property.maxGridSize,
        "totalConstMem": property.totalConstMem,
        "major": property.major,
        "minor": property.minor,
        "clockRate": property.clockRate,
        "textureAlignment": property.textureAlignment,
        "deviceOverlap": property.deviceOverlap,
        "multiProcessorCount": property.multiProcessorCount,
        "kernelExecTimeoutEnabled": property.kernelExecTimeoutEnabled,
        "computeMode": property.computeMode,
        "concurrentKernels": property.concurrentKernels,
        "ECCEnabled": property.ECCEnabled,
        "pciBusID": property.pciBusID,
        "pciDeviceID": property.pciDeviceID,
        "tccDriver": property.tccDriver,
    }

    return property_dict


def cuFree(uintptr_t ptr):
    p = <void * >ptr
    runtime_check(cudaFree(p))
    return

# cuda runtime check
def runtime_check(error):
    if error != 0:
        error_msg = cudaGetErrorString(error)
        raise Exception(error_msg)
    return

# cuda runtime check
def driver_check(error):
    cdef char * string
    if error != 0:
        cuGetErrorString(error, < const char**> & string)
        error_msg = str(string)
        raise Exception(error_msg)
    return

# Memcpy
cdef void cuMemcpyH2D(void* cpu_ptr, uintptr_t gpu_ptr, int size):
    # cpu to gpu
    runtime_check(cudaMemcpy(<void *>gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice))
    return


cdef cuMemcpyD2H(uintptr_t gpu_ptr, void *cpu_ptr, int size):
    # gpu to cpu
    runtime_check(cudaMemcpy(cpu_ptr, <void *>gpu_ptr, size, cudaMemcpyDeviceToHost))
    return


def cuMemcpyD2D(uintptr_t gpu_ptr1, uintptr_t gpu_ptr2, int size):
    # gpu to gpu
    runtime_check(cudaMemcpy(< void*>gpu_ptr2, < const void*>gpu_ptr1, size, cudaMemcpyDeviceToDevice))
    return


def cuMemcpyH2DAsync(np.ndarray[float, ndim=1, mode="c"] cpu_ptr, uintptr_t gpu_ptr, int size, int stream=0):
    # cpu to gpu
    runtime_check(cudaMemcpyAsync( < void*>gpu_ptr, & cpu_ptr[0], size, cudaMemcpyHostToDevice, < cudaStream_t > stream))
    return


def cuMemcpyD2HAsync(uintptr_t gpu_ptr, np.ndarray[float, ndim=1, mode="c"] cpu_ptr, int size, int stream=0):
    # gpu to cpu
    runtime_check(cudaMemcpyAsync( & cpu_ptr[0], < const void*>gpu_ptr, size, cudaMemcpyDeviceToHost, < cudaStream_t > stream))
    return


def cuMemcpyD2DAsync(uintptr_t gpu_ptr1, uintptr_t gpu_ptr2, int size, int stream=0):
    # gpu to gpu
    runtime_check(cudaMemcpyAsync( < void*>gpu_ptr2, < const void*>gpu_ptr1, size, cudaMemcpyDeviceToDevice, < cudaStream_t > stream))
    return


class Mempool(object):
    
    def __init__(self, nbytes, ptr):
        self.ptr = ptr
        self.nbytes = nbytes
        self.available = False

    def free(self):
        cuFree(self.ptr)


class allocator(object):

    def __init__(self):
        self.pool_list = []
        self.nbyte_list = []


    def malloc(self, nbytes):
        pool = self.getAvailablePool(nbytes)
        if pool is None:
            ptr = cuMalloc(nbytes)
            pool = Mempool(nbytes=nbytes, ptr=ptr)
            index = bisect.bisect(self.nbyte_list, pool.nbytes)
            self.pool_list.insert(index, pool)
            self.nbyte_list.insert(index, pool.nbytes)
        return pool.ptr

    def memset(self, ptr, value, nbytes):
        ptr = cuMemset(ptr, value, nbytes)

    def free(self, ptr):
        for p in self.pool_list:
            if ptr == p.ptr:
                p.available = True
                break

    def memcpyH2D(self, cpu_ptr, gpu_ptr, nbytes):
        # todo: this copy is not necessary
        buf = cpu_ptr.flatten()

        cdef _VoidPtr ptr = _VoidPtr(buf)
        cuMemcpyH2D(ptr.ptr, gpu_ptr, nbytes)


    def memcpyD2D(self, gpu_ptr1, gpu_ptr2, nbytes):
        cuMemcpyD2D(gpu_ptr1, gpu_ptr2, nbytes)

    def cuMemcpyD2DAsync(self, gpu_ptr1, gpu_ptr2, nbytes):
        cuMemcpyD2DAsync(gpu_ptr1, gpu_ptr2, nbytes, self.device_stream)

    def memcpyD2H(self, gpu_ptr, cpu_ptr, nbytes):
        shape = cpu_ptr.shape
        cpu_ptr = cpu_ptr.reshape(-1)

        cdef _VoidPtr ptr = _VoidPtr(cpu_ptr)

        cuMemcpyD2H(gpu_ptr, ptr.ptr, nbytes)
        cpu_ptr.reshape(shape)

    def getAvailablePool(self, size):
        pool = None
        min = size
        max = size * 1.5
        for p in self.pool_list:
            if min <= p.nbytes and p.nbytes < max and p.available:
                pool = p
                pool.available = False
                break
        return pool


gpu_allocator = allocator()


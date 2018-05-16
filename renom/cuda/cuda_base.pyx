from __future__ import print_function
import sys
import traceback
import contextlib
import bisect
cimport numpy as np
import numpy as pnp
cimport cython
from numbers import Number
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t
from cuda_utils cimport _VoidPtr
from renom.config import precision
import collections
import renom.core
import renom.cuda


@contextlib.contextmanager
def use_device(device_id):
    active = renom.cuda.is_cuda_active()
    cdef int cur

    if active:
        cur = cuGetDevice()
        cuSetDevice(device_id)  # switch dedice

    try:
        yield
    finally:
        if active:
            cuSetDevice(cur)   # restore device

def cuMalloc(uintptr_t nbytes):
    cdef void * p
    runtime_check(cudaMalloc( & p, nbytes))
    return < uintptr_t > p


cpdef cuMemset(uintptr_t ptr, int value, size_t size):
    p = <void * >ptr
    runtime_check(cudaMemset(p, value, size))
    return


cpdef uintptr_t cuCreateStream():
    cdef cudaStream_t stream
    runtime_check(cudaStreamCreate( & stream))
    return < uintptr_t > stream


cpdef cuSetDevice(int dev):
    runtime_check(cudaSetDevice(dev))


cpdef int cuGetDevice():
    cdef int dev
    runtime_check(cudaGetDevice(&dev))
    return dev

cpdef cuDeviceSynchronize():
    runtime_check(cudaDeviceSynchronize())


cpdef cuCreateCtx(device=0):
    cdef CUcontext ctx
    driver_check(cuCtxCreate( & ctx, 0, device))
    return int(ctx)


cpdef cuGetDeviceCxt():
    cdef CUdevice device
    driver_check(cuCtxGetDevice( & device))
    return int(device)


cpdef cuGetDeviceCount():
    cdef int count
    runtime_check(cudaGetDeviceCount( & count))
    return int(count)


cpdef cuGetDeviceProperty(device):
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


cpdef cuFree(uintptr_t ptr):
    p = <void * >ptr
    runtime_check(cudaFree(p))
    return

# cuda runtime check
cpdef runtime_check(error):
    if error != cudaSuccess:
        error_msg = cudaGetErrorString(error)
        raise Exception("CUDA Error: {}".format(error_msg))
    return

# cuda runtime check
cpdef driver_check(error):
    cdef char * string
    if error != 0:
        cuGetErrorString(error, < const char**> & string)
        error_msg = str(string)
        raise Exception(error_msg)
    return

# Memcpy
# TODO: in memcpy function, dest arguments MUST come first!

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


cpdef cuMemcpyD2DAsync(uintptr_t gpu_ptr1, uintptr_t gpu_ptr2, int size, int stream=0):
    # gpu to gpu
    runtime_check(cudaMemcpyAsync( < void*>gpu_ptr2, < const void*>gpu_ptr1, size, cudaMemcpyDeviceToDevice, < cudaStream_t > stream))
    return

def check_heap_device(*heaps):
    devices = {h._ptr.device_id for h in heaps if isinstance(h, renom.core.GPUValue)}
    
    current = {cuGetDevice()}
    if devices != current:
        raise RuntimeError('Invalid device_id: %s currennt: %s' % (devices, current))


cdef class GPUHeap(object):
    def __init__(self, nbytes, ptr, device_id):
        self.ptr = ptr
        self.nbytes = nbytes
        self.device_id = device_id

    def __int__(self):
        return self.ptr

    def __dealloc__(self):
        cdef cudaError_t err = cudaSetDevice(self.device_id)
        if err != cudaSuccess:
            err = cudaFree(<void * >self.ptrp)

        if err != cudaSuccess:
            print("Error in GPUHeap.__dealloc__():", err, file=sys.stderr)

    cpdef memcpyH2D(self, cpu_ptr, size_t nbytes):
        # todo: this copy is not necessary
        buf = cpu_ptr.ravel()
        cdef _VoidPtr ptr = _VoidPtr(buf)

        with renom.cuda.use_device(self.device_id):
            cuMemcpyH2D(ptr.ptr, self.ptr, nbytes)

    cpdef memcpyD2H(self, cpu_ptr, size_t nbytes):
        shape = cpu_ptr.shape
        cpu_ptr = pnp.reshape(cpu_ptr, -1)

        cdef _VoidPtr ptr = _VoidPtr(cpu_ptr)

        with renom.cuda.use_device(self.device_id):
            cuMemcpyD2H(self.ptr, ptr.ptr, nbytes)

        pnp.reshape(cpu_ptr, shape)

    cpdef memcpyD2D(self, gpu_ptr, size_t nbytes):
        assert self.device_id == gpu_ptr.device_id
        with renom.cuda.use_device(self.device_id):
            cuMemcpyD2D(self.ptr, gpu_ptr.ptr, nbytes)

    cpdef copy_from(self, other, size_t nbytes):
        cdef void *buf
        cdef int ret
        cdef uintptr_t src, dest

        assert nbytes <= self.nbytes
        assert nbytes <= other.nbytes

        n = min(self.nbytes, other.nbytes)
        if self.device_id == other.device_id:
            # self.memcpyD2D(other, n)
            other.memcpyD2D(self, n)
        else:
            runtime_check(cudaDeviceCanAccessPeer(&ret, self.device_id, other.device_id))
            if ret:
                src = other.ptr
                dest = self.ptr
                runtime_check(cudaMemcpyPeer(<void *>dest, self.device_id, <void*>src, other.device_id, nbytes))
            else:
                buf = malloc(n)
                if not buf:
                    raise MemoryError()
                try:
                    with renom.cuda.use_device(other.device_id):
                        cuMemcpyD2H(other.ptr, buf, n)

                    with renom.cuda.use_device(self.device_id):
                        cuMemcpyH2D(buf, self.ptr, n)

                finally:
                    free(buf)


cdef class GpuAllocator(object):

    def __init__(self):
        self._pool_lists = collections.defaultdict(list)

    @property
    def pool_list(self):
        device = cuGetDevice()
        return self._pool_lists[device]

    cpdef GPUHeap malloc(self, size_t nbytes):
        cdef GPUHeap pool = self.getAvailablePool(nbytes)
        if pool is None:
            ptr = cuMalloc(nbytes)
            pool = GPUHeap(nbytes=nbytes, ptr=ptr, device_id=cuGetDevice())

        return pool

    cpdef free(self, GPUHeap pool):
        if pool.nbytes:
            device_id = pool.device_id
            if not getattr(bisect, 'bisect', None):
                # Python is shutting down
                return
            self._pool_lists[device_id]
            index = bisect.bisect(self._pool_lists[device_id], (pool.nbytes,))
            self._pool_lists[device_id].insert(index, (pool.nbytes, pool))

    cpdef GPUHeap getAvailablePool(self, size_t size):
        pool = None
        cdef size_t min = size
        cdef size_t max = size * 2 + 4096

        device = cuGetDevice()
        pools = self._pool_lists[device]

        cdef size_t idx = bisect.bisect_left(pools, (size,))

        cdef GPUHeap p
        cdef size_t i

        for i in range(idx, len(pools)):
            _, p = pools[i]
            if p.nbytes >= max:
                break

            if min <= p.nbytes:
                pool = p
                del pools[i]
                break

        return pool

    cpdef release_pool(self, deviceID=None):
        if deviceID is None:
            self._pool_lists = collections.defaultdict(list)
        else:
            del self._pool_lists[deviceID]


gpu_allocator = GpuAllocator()

cdef GpuAllocator c_gpu_allocator
c_gpu_allocator = gpu_allocator 

cpdef GpuAllocator get_gpu_allocator():
    return c_gpu_allocator


cpdef _cuSetLimit(limit, value):
    cdef size_t c_value=999;

    cuInit(0)

    ret = cuCtxGetLimit(&c_value, limit)

    cuCtxSetLimit(limit, value)


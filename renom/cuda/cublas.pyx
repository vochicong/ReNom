import contextlib
import numpy as np
from cublas import *
from libc.stdint cimport uintptr_t
import cuda_base

cublas_handle = []


def create_cublasHander():
    cdef cublasHandle_t handle
    cublasCreate_v2( & handle)
    return < uintptr_t > handle


@contextlib.contextmanager
def cublas_handler(device=None):
    global cublas_handle
    handler = None
    if cublas_handle:
        handler = cublas_handle[0]
    else:
        handler = create_cublasHander()
        cublas_handle.append(handler)
    yield < uintptr_t > handler


def check(cublasStatus_t status):
    if status != CUBLAS_STATUS_SUCCESS:
        raise Exception("An error has occurred in cuBlas function. Error code %d."%status)

# Scal
def cublas_scal(alpha, gpu_value):
    cdef int size = gpu_value.size
    cdef uintptr_t ptr = <uintptr_t>gpu_value._ptr
    
    cuda_base.check_heap_device(gpu_value)
    if dtype == np.float32:
        cublasSscal(size, <float>alpha, <float*>ptr, 1)
    elif gpu_value1.dtype == np.float64:
        cublasDscal(size, <double>alpha, <double*>ptr, 1)
    return

# AXPY
def cublas_axpy(gpu_value1, gpu_value2):
    cdef int n = gpu_value1.size
    cdef uintptr_t ptr1 = <uintptr_t>gpu_value1._ptr
    cdef uintptr_t ptr2 = <uintptr_t>gpu_value2._ptr
    
    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    if gpu_value1.dtype == np.float32:
        cublasSaxpy(n, <float>1.0, <const float*>ptr1, 1, <float*>ptr2, 1)
    elif gpu_value1.dtype == np.float64:
        cublasDaxpy(n, <double>1.0, <const double*>ptr1, 1, <double*>ptr2, 1)
    return

# GEMM
def cublas_gemm(gpu_value1, t1, gpu_value2, t2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    shape1 = gpu_value1.shape or (1, 1)
    shape2 = gpu_value2.shape or (1, 1)
    
    cdef char c1 = 'T' if t1 == 1 else 'N'
    cdef char c2 = 'T' if t2 == 1 else 'N'
    cdef int n = shape2[0] if t2 == 1 else shape2[1]
    cdef int m = shape1[1] if t1 == 1 else shape1[0]
    cdef int k = shape2[1] if t2 == 1 and t1 == 0 else shape2[0]
    cdef uintptr_t ptr1 = <uintptr_t>gpu_value1._ptr
    cdef uintptr_t ptr2 = <uintptr_t>gpu_value2._ptr
    cdef uintptr_t ptr3 = <uintptr_t>gpu_value3._ptr
    
    if len(shape1) > 2:
        raise Exception("Operation cuBlas gemm is only accept 2 dimentional matrix.")
    
    if gpu_value1.dtype == np.float32:
        cublasSgemm(c2, c1, n, m, k, 1.0, <float*>ptr2, shape2[1], <float*>ptr1, shape1[1], 0.0, <float*>ptr3, n)
    else:
        cublasDgemm(c2, c1, n, m, k, 1.0, <double*>ptr2, shape2[1], <double*>ptr1, shape1[1], 0.0, <double*>ptr3, n)
    return

# GEAM
def cublas_geam(handle, a, gpu_value1, t1, b, gpu_value2, t2, gpu_value3):
    cdef int n, m
    cdef float f_alf = <float>a
    cdef float f_bet = <float>b
    cdef double d_alf = <double>a
    cdef double d_bet = <double>b
    cdef uintptr_t ptr1 = <uintptr_t>gpu_value1._ptr
    cdef uintptr_t ptr2 = <uintptr_t>gpu_value2._ptr
    cdef uintptr_t ptr3 = <uintptr_t>gpu_value3._ptr
    cdef cublasHandle_t handler = <cublasHandle_t><uintptr_t>handle
    cdef cublasOperation_t c1 = CUBLAS_OP_T if t1 == 1 else CUBLAS_OP_N
    cdef cublasOperation_t c2 = CUBLAS_OP_T if t2 == 1 else CUBLAS_OP_N

    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    shape1 = gpu_value1.shape or (1, 1)
    shape2 = gpu_value2.shape or (1, 1)
    
    if t1 == 0:
        n = shape1[1]
        m = shape1[0]
    elif t2 == 0:
        n = shape2[1]
        m = shape2[0]
    else:
        n = shape2[0]
        m = shape2[1]
    
    if gpu_value1.dtype == np.float32:
        check(cublasSgeam(handler, c1, c2, n, m, &f_alf, <const float*>ptr1, shape1[1], &f_bet,
                          <const float*>ptr2, shape2[1], <float*>ptr3, n))
    elif gpu_value1.dtype == np.float64:
        check(cublasDgeam(handler, c1, c2, n, m, &d_alf, <const double*>ptr1, shape1[1], &d_bet,
                          <const double*>ptr2, shape2[1], <double*>ptr3, n))
    return

def cublas_transpose(handle, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    cublas_geam(handle, 1.0, gpu_value1, 1, 0.0, gpu_value1, 1, gpu_value2)



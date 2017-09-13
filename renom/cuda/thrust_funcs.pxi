#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
関数命名規則
関数名: cu〜    (python側から呼ばれる関数)
        
引数名: gpu_value
"""
import numpy as np
from libc.stdint cimport uintptr_t
from libcpp cimport bool
import cuda_base
import operator
import functools

def cunegate(input, result):
    cuda_base.check_heap_device(input, result)

    cdef VALUE_TYPE * first = <VALUE_TYPE * > < uintptr_t > input._ptr
    cdef VALUE_TYPE * last = first + <size_t > input.size
    cdef VALUE_TYPE * output = <VALUE_TYPE * > < uintptr_t > result._ptr
    thrust_negate(first, last, output)


def curelu_foward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_relu_forward(ptr1, ptr2, size)


def curelu_backard(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_relu_backward(ptr1, ptr2, size)


def culeaky_leru_forward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_leaky_relu_forward( < VALUE_TYPE > s, ptr1, ptr2, size);


def culeaky_leru_backward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_leaky_relu_backward( < VALUE_TYPE > s, ptr1, ptr2, size);


def cueru_forward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_elu_forward( < VALUE_TYPE > s, ptr1, ptr2, size);


def cueru_backward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_elu_backward( < VALUE_TYPE > s, ptr1, ptr2, size);


def cusigmoid(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_sigmoid(ptr1, ptr2, size)


def cutanh(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_tanh(ptr1, ptr2, size)


cdef basic_operation(Operation op, gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2
    cdef VALUE_TYPE * ptr3 = <VALUE_TYPE * > < uintptr_t > gpu_value3._ptr
    cdef int elem_size_a = gpu_value1.size
    cdef int elem_size_b
    if hasattr(gpu_value2, "_ptr"):
        ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
        value = 0
        elem_size_b = gpu_value2.size
    else:
        ptr2 = <VALUE_TYPE * >0
        value = gpu_value2
        elem_size_b = 1
    thrust_operation(op, < VALUE_TYPE > value, elem_size_a, ptr1, elem_size_b, ptr2, ptr3)


def cumul(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    basic_operation(Operation.MUL, gpu_value1, gpu_value2, gpu_value3)


def cuadd(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    basic_operation(Operation.ADD, gpu_value1, gpu_value2, gpu_value3)


def cusub(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    basic_operation(Operation.SUB, gpu_value1, gpu_value2, gpu_value3)


def cudiv(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    basic_operation(Operation.DIV, gpu_value1, gpu_value2, gpu_value3)


def curdiv(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    basic_operation(Operation.RDIV, gpu_value1, gpu_value2, gpu_value3)


def cupow(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    basic_operation(Operation.POW, gpu_value1, gpu_value2, gpu_value3)


def curpow(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    basic_operation(Operation.RPOW, gpu_value1, gpu_value2, gpu_value3)


def cufill(value, gpu_value):
    cdef int size = <int > gpu_value.size
    cdef VALUE_TYPE v = <VALUE_TYPE > value
    cdef VALUE_TYPE * ptr = <VALUE_TYPE * > < uintptr_t > gpu_value._ptr

    cuda_base.check_heap_device(gpu_value)
    thrust_fill(v, ptr, size)


def culoge(gpu_value1, gpu_value2):
    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_loge(ptr1, ptr2, size)


def cuexp(gpu_value1, gpu_value2):
    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_exp(ptr1, ptr2, size)


def cusqrt(gpu_value1, gpu_value2):
    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_sqrt(ptr1, ptr2, size)


def cucross_entropy(gpu_value1, gpu_value2, gpu_value3):
    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE * ptr3 = <VALUE_TYPE * > < uintptr_t > gpu_value3._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    thrust_cross_entropy(ptr1, ptr2, ptr3, size)


def cubroadcast(gpu_value1, gpu_value2):
    cdef int size_1 = <int > gpu_value1.size
    cdef int size_2 = <int > gpu_value2.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_broad_cast(size_1, ptr1, size_2, ptr2)


def cuabs_forward(gpu_value1, gpu_value2):
    cdef int size = gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_abs_forward(ptr1, ptr2, size)


def cuabs_backward(gpu_value1, gpu_value2):
    cdef int size = gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_abs_backward(ptr1, ptr2, size)


def cusum(gpu_value1, gpu_value2=None, axis=None):
    cdef int size = gpu_value1.size
    cdef VALUE_TYPE * ptr1
    cdef VALUE_TYPE * ptr2

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    if axis is None:
        ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
        return thrust_all_reduce(ptr1, size)
    else:
        axis_size = gpu_value1.shape[axis]

        if axis == 1:
            strides = 1
        elif axis == 0:
            if len(gpu_value1.shape) == 1:
                strides = 1
            else:
                strides = gpu_value1.shape[1]
        else:
            strides = np.prod(gpu_value1.shape[:axis])

        step = 1 if axis != 1 else gpu_value1.shape[1]
        thrust_strided_reduce(< VALUE_TYPE*> < uintptr_t > gpu_value1._ptr,
                               < VALUE_TYPE * > < uintptr_t > gpu_value2._ptr,
                               strides, axis_size, step, size)


def cumin(value, gpu_value1, gpu_value2=None):
    cdef int size = gpu_value1.size
    cdef VALUE_TYPE * ptr1 = < VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = < VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE v = <VALUE_TYPE > value

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_min(v, ptr1, ptr2, size)


def cumax(value, gpu_value1, gpu_value2=None):
    cdef int size = gpu_value1.size
    cdef VALUE_TYPE * ptr1 = < VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = < VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE v = <VALUE_TYPE > value

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_max(v, ptr1, ptr2, size)


def culstm_forward_activate(u):
    cdef int N = u.shape[0]
    cdef int M = u.shape[1]

    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    thrust_forward_lstm_activate(N, M, ptr_u)


def culstm_forward(u, s, ps, z):
    cdef int N = u.shape[0]
    cdef int M = u.shape[1]

    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_s = < VALUE_TYPE * > < uintptr_t > s._ptr
    cdef VALUE_TYPE * ptr_ps = < VALUE_TYPE * > < uintptr_t > ps._ptr
    cdef VALUE_TYPE * ptr_z = < VALUE_TYPE * > < uintptr_t > z._ptr
    thrust_forward_lstm(N, M, ptr_u, ptr_s, ptr_ps, ptr_z)


def culstm_backward(u, du, s, ps, e, pgf, dou, dou_n, temporal):
    cdef int N = u.shape[0]
    cdef int M = u.shape[1]
    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_du = < VALUE_TYPE * > < uintptr_t > du._ptr
    cdef VALUE_TYPE * ptr_s = < VALUE_TYPE * > < uintptr_t > s._ptr
    cdef VALUE_TYPE * ptr_ps = < VALUE_TYPE * > < uintptr_t > ps._ptr
    cdef VALUE_TYPE * ptr_e = < VALUE_TYPE * > < uintptr_t > e._ptr
    cdef VALUE_TYPE * ptr_pgf = < VALUE_TYPE * > < uintptr_t > pgf._ptr
    cdef VALUE_TYPE * ptr_dou = < VALUE_TYPE * > < uintptr_t > dou._ptr
    cdef VALUE_TYPE * ptr_dou_n = < VALUE_TYPE * > < uintptr_t > dou_n._ptr
    cdef bool temp = temporal
    thrust_backward_lstm(N, M, ptr_u, ptr_du, ptr_s, ptr_ps,
                         ptr_e, ptr_pgf, ptr_dou, ptr_dou_n, temp)


def cubinarize(gpu_value1, th, gpu_value2):
    cdef int N = gpu_value1.size
    cdef VALUE_TYPE * gpu_ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * gpu_ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE threathold = th
    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_binarize(gpu_ptr1, threathold, N, gpu_ptr2)


def cuembedding_forward(gpu_value1, weight, gpu_value2):
    cdef int N = gpu_value1.shape[0]
    cdef int K = weight.shape[0]
    cdef int M = weight.shape[1]
    cdef VALUE_TYPE * gpu_ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * gpu_ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE * weight_ptr = <VALUE_TYPE * > < uintptr_t > weight._ptr
    cuda_base.check_heap_device(gpu_value1, gpu_value2, weight)
    thrust_embedding_forward(N, K, M, gpu_ptr1, weight_ptr, gpu_ptr2)


def cuembedding_backward(gpu_index, gpu_dy, gpu_dx):
    cdef int N = gpu_index.shape[0]
    cdef int K = gpu_dx.shape[0]
    cdef int M = gpu_dx.shape[1]
    cdef VALUE_TYPE * index_ptr = <VALUE_TYPE * > < uintptr_t > gpu_index._ptr
    cdef VALUE_TYPE * dy_ptr = <VALUE_TYPE * > < uintptr_t > gpu_dy._ptr
    cdef VALUE_TYPE * dx_ptr = <VALUE_TYPE * > < uintptr_t > gpu_dx._ptr
    cuda_base.check_heap_device(gpu_dy, gpu_index, gpu_dx)
    thrust_embedding_backward(N, K, M, index_ptr, dy_ptr, dx_ptr)


def cuconcat(gpu_value1, gpu_value2, gpu_value3, axis):

    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    cdef size_t size = gpu_value1.nbytes + gpu_value2.nbytes
    if gpu_value3.nbytes < size:
        raise ValueError("Insufficient destination buffer size")

    if (not gpu_value1.shape) or (not gpu_value2.shape):
        raise ValueError("zero-dimensional arrays cannot be concatenated")

    s1 = gpu_value1.shape[:axis] + gpu_value1.shape[axis+1:]
    s2 = gpu_value1.shape[:axis] + gpu_value1.shape[axis+1:]

    if s1 != s2:
        raise ValueError("all the input array dimensions except"
                         " for the concatenation axis must match exactly")

    cdef size_t size1 = functools.reduce(operator.__mul__, gpu_value1.shape[axis:], 1)
    cdef size_t size2 = functools.reduce(operator.__mul__, gpu_value2.shape[axis:], 1)
    cdef size_t rec_size = size1 + size2

    cdef VALUE_TYPE *ptr1 = <VALUE_TYPE*><uintptr_t> gpu_value1._ptr
    cdef VALUE_TYPE *ptr2 = <VALUE_TYPE*><uintptr_t> gpu_value2._ptr
    cdef VALUE_TYPE *ptr3 = <VALUE_TYPE*><uintptr_t> gpu_value3._ptr

    thrust_copy_memory_stride(ptr3, ptr1, gpu_value1.size, rec_size, size1)
    thrust_copy_memory_stride(ptr3 + size1, ptr2, gpu_value2.size, rec_size, size2)

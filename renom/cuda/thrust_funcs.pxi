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
import renom.core
import renom.cuda


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


def culstm_backward(u, du, s, ps, e, pgf, dou, dou_n):
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
    thrust_backward_lstm(N, M, ptr_u, ptr_du, ptr_s, ptr_ps,
                         ptr_e, ptr_pgf, ptr_dou, ptr_dou_n)


def cupeepholelstm_forward(u, wc, prestate, state, z):
    cuda_base.check_heap_device(u, prestate, state, wc, z)

    cdef int N = u.shape[0]
    cdef int M = u.shape[1]
    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_z = < VALUE_TYPE * > < uintptr_t > z._ptr
    cdef VALUE_TYPE * ptr_ps = < VALUE_TYPE * > < uintptr_t > prestate._ptr
    cdef VALUE_TYPE * ptr_s = < VALUE_TYPE * > < uintptr_t > state._ptr
    cdef VALUE_TYPE * ptr_wc = < VALUE_TYPE * > < uintptr_t > wc._ptr
    thrust_forward_peephole_lstm(N, M, ptr_u, ptr_wc, ptr_ps, ptr_s, ptr_z)


def cupeepholelstm_backward(u, prestate, state, prefg, wc, dy, drt, dot, dr, dou, dwc):
    cuda_base.check_heap_device(u, prestate, state, prestate, wc,
                                dy, drt, dot, dou, dr, dwc)
    cdef int N = u.shape[0]
    cdef int M = u.shape[1]

    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_ps = < VALUE_TYPE * > < uintptr_t > prestate._ptr
    cdef VALUE_TYPE * ptr_s = < VALUE_TYPE * > < uintptr_t > state._ptr
    cdef VALUE_TYPE * ptr_pfg = < VALUE_TYPE * > < uintptr_t > prefg._ptr
    cdef VALUE_TYPE * ptr_wc = < VALUE_TYPE * > < uintptr_t > wc._ptr
    cdef VALUE_TYPE * ptr_dy = < VALUE_TYPE * > < uintptr_t > dy._ptr
    cdef VALUE_TYPE * ptr_drt = < VALUE_TYPE * > < uintptr_t > drt._ptr
    cdef VALUE_TYPE * ptr_dot = < VALUE_TYPE * > < uintptr_t > dot._ptr
    cdef VALUE_TYPE * ptr_dr = < VALUE_TYPE * > < uintptr_t > dr._ptr
    cdef VALUE_TYPE * ptr_dou = < VALUE_TYPE * > < uintptr_t > dou._ptr
    cdef VALUE_TYPE * ptr_dwc = < VALUE_TYPE * > < uintptr_t > dwc._ptr
    thrust_backward_peephole_lstm(N, M, ptr_u, ptr_ps, ptr_s, ptr_pfg, ptr_wc,
                                  ptr_dy, ptr_drt, ptr_dot, ptr_dr, ptr_dou, ptr_dwc)


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

    s1 = gpu_value1.shape[:axis] + gpu_value1.shape[axis + 1:]
    s2 = gpu_value1.shape[:axis] + gpu_value1.shape[axis + 1:]

    if s1 != s2:
        raise ValueError("all the input array dimensions except"
                         " for the concatenation axis must match exactly")

    cdef size_t size1 = functools.reduce(operator.__mul__, gpu_value1.shape[axis:], 1)
    cdef size_t size2 = functools.reduce(operator.__mul__, gpu_value2.shape[axis:], 1)
    cdef size_t rec_size = size1 + size2

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE * ptr3 = <VALUE_TYPE * > < uintptr_t > gpu_value3._ptr

    thrust_copy_memory_stride(ptr3, ptr1, gpu_value1.size, rec_size, size1)
    thrust_copy_memory_stride(ptr3 + size1, ptr2, gpu_value2.size, rec_size, size2)


ctypedef object(*REDUCE_FUNC)(
    size_t max_grids, size_t num_threads,
    VALUE_TYPE * src, size_t src_size,
    object result_shape, size_t result_size,
    size_t src_per_result,
    size_t sequence_stride,
    size_t num_axis,
    reduce_shape_infos * reductions_infos,
    reduce_shape_infos * seqs_infos,
    object args)


import collections


def _del_items(src, indexes):
    ret = list(src)
    for i in reversed(indexes):
        del ret[i]
    return ret


def _calc_index(reductions, kept_shapes_size, n):
    ret = 0
    if kept_shapes_size:
        ret = n % kept_shapes_size

    for info in reductions:
        v = n
        if info.group_size:
            v = v % info.group_size
        v = v // info.out_size
        ret += v * info.in_size

    return ret


cdef _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, REDUCE_FUNC func, args):
    assert num_threads < 600

    if not gpu_value1.shape:
        return gpu_value1

    if isinstance(axis, int):
        axis = [axis]
    elif not axis:
        axis = list(range(len(gpu_value1.shape)))

    axis = list(sorted(set(axis)))

    if (max(axis) >= len(gpu_value1.shape)) or (min(axis) < 0):
        raise ValueError('Invalid axis: %s' % (axis,))

    if len(axis) == len(gpu_value1.shape):
        reduce_axis = [0]
        src_shape = (gpu_value1.size,)
        src_size = gpu_value1.size

        result_shape = ()
        result_size = 1
    else:
        reduce_axis = axis
        src_shape = gpu_value1.shape
        src_size = gpu_value1.size

        result_shape = _del_items(src_shape, reduce_axis)
        result_size = functools.reduce(operator.__mul__, result_shape, 1)

    if len(reduce_axis) >= 16:
        raise ValueError('Invalid axis: %s' % (axis,))

    kept_shapes = src_shape[reduce_axis[-1] + 1:]
    kept_shapes_size = functools.reduce(operator.__mul__, kept_shapes, 1)

    src_per_result = src_size // result_size
    sequence_per_result = src_shape[reduce_axis[0]]
    sequence_stride = kept_shapes_size
    src_per_sequence = src_per_result // sequence_per_result

    max_threads_per_result = min(src_per_result, num_threads)
    preferred_result_per_block = num_threads // max_threads_per_result

    num_blocks = min((result_size - 1) // preferred_result_per_block + 1, max_grids)

    cdef reduce_shape_infos reduction_infos
    group_size = 0
    f = 0

    for n, i in enumerate(reduce_axis):
        in_shape = src_shape[i:]
        in_size = functools.reduce(operator.__mul__, in_shape, 1)
        out_shape = _del_items(src_shape[i + 1:], [p - i - 1 for p in reduce_axis[n + 1:]])
        out_size = functools.reduce(operator.__mul__, out_shape, 1)

        reduction_infos.in_size[n] = in_size
        reduction_infos.out_size[n] = out_size
        reduction_infos.group_size[n] = group_size

        group_size = out_size

    cdef reduce_shape_infos seq_infos

    group_size = 0
    f = 0
    for n, i in enumerate(reduce_axis):
        in_shape = src_shape[i + 1:]
        in_size = functools.reduce(operator.__mul__, in_shape, 1)
        out_shape = [src_shape[p] for p in reduce_axis[n + 1:]]
        out_size = functools.reduce(operator.__mul__, out_shape, 1)

        seq_infos.in_size[n] = in_size
        seq_infos.out_size[n] = out_size
        seq_infos.group_size[n] = group_size

        group_size = out_size

    if not keepdims:
        ret_shape = result_shape
    else:
        ret_shape = list(gpu_value1.shape)
        for s in axis:
            ret_shape[s] = 1

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr

    return func(num_blocks, num_threads, ptr1, src_size, ret_shape, result_size, src_per_result, sequence_stride,
                len(reduce_axis), & reduction_infos, & seq_infos, args)


cdef _cusum(size_t max_grids, size_t num_threads,
            VALUE_TYPE * src, size_t src_size,
            object result_shape, size_t result_size,
            size_t src_per_result,
            size_t sequence_stride,
            size_t num_axis,
            reduce_shape_infos * reductions_infos,
            reduce_shape_infos * seqs_infos,
            object args):

    result = renom.core.GPUValue(shape=result_shape)
    cdef VALUE_TYPE * ptr = <VALUE_TYPE * > < uintptr_t > result._ptr

    thrust_reduce_sum(max_grids, num_threads,
                      src, src_size,
                      ptr, result_size,
                      src_per_result,
                      sequence_stride,
                      num_axis,
                      reductions_infos,
                      seqs_infos)

    return result


def cusum(gpu_value1, axis=None, keepdims=False, max_grids=65536, num_threads=512):
    return _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, _cusum, None)


cdef _cu_reduce_min(size_t max_grids, size_t num_threads,
                    VALUE_TYPE * src, size_t src_size,
                    object result_shape, size_t result_size,
                    size_t src_per_result,
                    size_t sequence_stride,
                    size_t num_axis,
                    reduce_shape_infos * reductions_infos,
                    reduce_shape_infos * seqs_infos,
                    object args):

    result = renom.core.GPUValue(shape=result_shape)
    cdef VALUE_TYPE * ptr = <VALUE_TYPE * > < uintptr_t > result._ptr

    thrust_reduce_min(max_grids, num_threads,
                      src, src_size,
                      ptr, result_size,
                      src_per_result,
                      sequence_stride,
                      num_axis,
                      reductions_infos,
                      seqs_infos)

    return result


def cu_reduce_min(gpu_value1, axis=None, keepdims=False, max_grids=65536, num_threads=512):
    return _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, _cu_reduce_min, None)


cdef _cu_reduce_max(size_t max_grids, size_t num_threads,
                    VALUE_TYPE * src, size_t src_size,
                    object result_shape, size_t result_size,
                    size_t src_per_result,
                    size_t sequence_stride,
                    size_t num_axis,
                    reduce_shape_infos * reductions_infos,
                    reduce_shape_infos * seqs_infos,
                    object args):

    result = renom.core.GPUValue(shape=result_shape)
    cdef VALUE_TYPE * ptr = <VALUE_TYPE * > < uintptr_t > result._ptr

    thrust_reduce_max(max_grids, num_threads,
                      src, src_size,
                      ptr, result_size,
                      src_per_result,
                      sequence_stride,
                      num_axis,
                      reductions_infos,
                      seqs_infos)

    return result


def cu_reduce_max(gpu_value1, axis=None, keepdims=False, max_grids=65536, num_threads=512):
    return _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, _cu_reduce_max, None)


cdef _cu_reduce_argmin(size_t max_grids, size_t num_threads,
                       VALUE_TYPE * src, size_t src_size,
                       object result_shape, size_t result_size,
                       size_t src_per_result,
                       size_t sequence_stride,
                       size_t num_axis,
                       reduce_shape_infos * reductions_infos,
                       reduce_shape_infos * seqs_infos,
                       object args):

    result = renom.core.GPUValue(shape=result_shape, dtype='int64')
    cdef size_t * ptr = <size_t * > < uintptr_t > result._ptr

    cdef size_t mod, div
    mod, div = args

    thrust_reduce_argmin(max_grids, num_threads,
                         src, src_size,
                         ptr, result_size,
                         src_per_result,
                         sequence_stride,
                         num_axis,
                         reductions_infos,
                         seqs_infos,
                         mod, div)

    return result


def cu_reduce_argmin(gpu_value1, axis=None, max_grids=65536, num_threads=512):
    if axis is not None:
        if not isinstance(axis, int) or axis >= len(gpu_value1.shape):
            raise ValueError("Invalid axis")

        mod = functools.reduce(operator.__mul__, gpu_value1.shape[axis:], 1)
        div = functools.reduce(operator.__mul__, gpu_value1.shape[axis + 1:], 1)

    else:
        mod = functools.reduce(operator.__mul__, gpu_value1.shape, 1)
        div = 1

    keepdims = False
    return _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, _cu_reduce_argmin, (mod, div))


cdef _cu_reduce_argmax(size_t max_grids, size_t num_threads,
                       VALUE_TYPE * src, size_t src_size,
                       object result_shape, size_t result_size,
                       size_t src_per_result,
                       size_t sequence_stride,
                       size_t num_axis,
                       reduce_shape_infos * reductions_infos,
                       reduce_shape_infos * seqs_infos,
                       object args):

    result = renom.core.GPUValue(shape=result_shape, dtype='int64')
    cdef size_t * ptr = <size_t * > < uintptr_t > result._ptr

    cdef size_t mod, div
    mod, div = args

    thrust_reduce_argmax(max_grids, num_threads,
                         src, src_size,
                         ptr, result_size,
                         src_per_result,
                         sequence_stride,
                         num_axis,
                         reductions_infos,
                         seqs_infos,
                         mod, div)

    return result


def cu_reduce_argmax(gpu_value1, axis=None, max_grids=65536, num_threads=512):
    if axis is not None:
        if not isinstance(axis, int) or axis >= len(gpu_value1.shape):
            raise ValueError("Invalid axis")

        mod = functools.reduce(operator.__mul__, gpu_value1.shape[axis:], 1)
        div = functools.reduce(operator.__mul__, gpu_value1.shape[axis + 1:], 1)

    else:
        mod = functools.reduce(operator.__mul__, gpu_value1.shape, 1)
        div = 1

    keepdims = False

    return _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, _cu_reduce_argmax, (mod, div))


def cu_add_bias(bias, gpu_value):
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > bias._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value._ptr
    cdef int size = <int > gpu_value.size
    cdef int wh = <int > (gpu_value.shape[2] * gpu_value.shape[3])
    cdef int n = <int > gpu_value.shape[0]
    thrust_add_bias(size, n, wh, ptr1, ptr2)


def cu_transpose(gpu_value1, axis):
    strides = [np.prod(gpu_value1.shape[i + 1:], dtype='int') for i in range(len(gpu_value1.shape))]

    if not axis:
        axis = tuple(reversed(range(len(gpu_value1.shape))))

    if len(axis) >= 16:
        raise ValueError('Invalid axis: %s' % (axis,))

    new_shape = [gpu_value1.shape[i] for i in axis]

    cdef size_t src_strides[16]
    for i, s in enumerate(axis):
        src_strides[i] = strides[s]

    cdef size_t new_strides[16]
    for i in range(len(new_shape)):
        new_strides[i] = np.prod(new_shape[i + 1:], dtype='int')

    cdef VALUE_TYPE * ptr = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    size = np.prod(gpu_value1.shape)

    result = renom.core.GPUValue(shape=new_shape)
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > result._ptr

    thrust_transpose(size,
                     len(new_shape),
                     ptr, src_strides,
                     ptr2, new_strides)

    return result

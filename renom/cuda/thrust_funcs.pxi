#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
関数命名規則
関数名: cu〜    (python側から呼ばれる関数)

引数名: gpu_value
"""
import numpy as np
from libc.stdint cimport uintptr_t
from libc.stdio cimport printf
from libcpp cimport bool
import cuda_base
import operator
import functools
import renom.core
import renom.cuda

# For debug
import time


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
    thrust_leaky_relu_forward(< VALUE_TYPE > s, ptr1, ptr2, size);


def culeaky_leru_backward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_leaky_relu_backward(< VALUE_TYPE > s, ptr1, ptr2, size);


def cueru_forward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_elu_forward(< VALUE_TYPE > s, ptr1, ptr2, size);


def cueru_backward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_elu_backward(< VALUE_TYPE > s, ptr1, ptr2, size);


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


ctypedef void(*BINOP_FUNC)(
    VALUE_TYPE * a, VALUE_TYPE * b, VALUE_TYPE * c,
    size_t size, binop_strides * strides)


cdef bin_operation(BINOP_FUNC func, lhs, rhs, ret):
    cuda_base.check_heap_device(lhs, rhs, ret)

    if not isinstance(rhs, renom.core.GPUValue):
        rhs = renom.core.GPUValue(np.array(rhs))

    cdef binop_strides strides

    start_t = time.time()
    if lhs.shape == rhs.shape == ret.shape:
        strides.size = 1
        strides.result_strides[0] = 1
        strides.lhs_strides[0] = 1
        strides.rhs_strides[0] = 1
    else:
        ret_strides = [np.prod(ret.shape[i + 1:], dtype='int') for i in range(len(ret.shape))]

        lhs_strides = [np.prod(lhs.shape[i + 1:], dtype='int') for i in range(len(lhs.shape))]
        lhs_strides = [0] * (len(ret.shape) - len(lhs.shape)) + lhs_strides

        for i, (arg, dest) in enumerate(zip(reversed(lhs.shape), reversed(ret.shape)), 1):
            if arg != dest:
                lhs_strides[i * -1] = 0

        rhs_strides = [np.prod(rhs.shape[i + 1:], dtype='int') for i in range(len(rhs.shape))]
        rhs_strides = [0] * (len(ret.shape) - len(rhs.shape)) + rhs_strides

        for i, (arg, dest) in enumerate(zip(reversed(rhs.shape), reversed(ret.shape)), 1):
            if arg != dest:
                rhs_strides[i * -1] = 0

        strides.size = len(ret_strides)
        for i in range(strides.size):
            strides.result_strides[i] = ret_strides[i]
            strides.lhs_strides[i] = lhs_strides[i]
            strides.rhs_strides[i] = rhs_strides[i]

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > lhs._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > rhs._ptr
    cdef VALUE_TYPE * ptr3 = <VALUE_TYPE * > < uintptr_t > ret._ptr
    size = np.prod(ret.shape, dtype='int')

    assert strides.size < 6, "Binary operation error. Only tensors that has less than 6dims are accepted. Actual is {} dim tensor.".format(
        strides.size)
    func(ptr1, ptr2, ptr3, size, & strides)


def cumul(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    bin_operation(thrust_mul, gpu_value1, gpu_value2, gpu_value3)


def cuadd(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    bin_operation(thrust_add, gpu_value1, gpu_value2, gpu_value3)


def cusub(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    bin_operation(thrust_sub, gpu_value1, gpu_value2, gpu_value3)


def cudiv(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    bin_operation(thrust_div, gpu_value1, gpu_value2, gpu_value3)


def curdiv(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    bin_operation(thrust_rdiv, gpu_value1, gpu_value2, gpu_value3)


def cupow(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    bin_operation(thrust_pow, gpu_value1, gpu_value2, gpu_value3)


def curpow(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    bin_operation(thrust_rpow, gpu_value1, gpu_value2, gpu_value3)


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


def cugru_forward(input,hminus,u,ABC,h):
    cdef int X = input.shape[0]
    cdef int Y = input.shape[1]
    cdef int M = input.shape[1] // 3
    cdef VALUE_TYPE * ptr_input = < VALUE_TYPE * > < uintptr_t > input._ptr
    cdef VALUE_TYPE * ptr_hminus = < VALUE_TYPE * > < uintptr_t > hminus._ptr
    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_ABC = < VALUE_TYPE * > < uintptr_t > ABC._ptr
    cdef VALUE_TYPE * ptr_h = < VALUE_TYPE * > < uintptr_t > h._ptr
    thrust_forward_gru(X,Y,M,ptr_input,ptr_hminus,ptr_u,ptr_ABC,ptr_h)


def cugru_backward(a, b, c, d, e, f, g, h, i):
    cdef int H = a.shape[0]
    cdef int W = a.shape[1]
    cdef int M = a.shape[1] // 3
    cdef int V = i.shape[1]


    cdef VALUE_TYPE * ptr_a = < VALUE_TYPE * > < uintptr_t > a._ptr
    cdef VALUE_TYPE * ptr_b = < VALUE_TYPE * > < uintptr_t > b._ptr
    cdef VALUE_TYPE * ptr_c = < VALUE_TYPE * > < uintptr_t > c._ptr
    cdef VALUE_TYPE * ptr_d = < VALUE_TYPE * > < uintptr_t > d._ptr
    cdef VALUE_TYPE * ptr_e = < VALUE_TYPE * > < uintptr_t > e._ptr
    cdef VALUE_TYPE * ptr_f = < VALUE_TYPE * > < uintptr_t > f._ptr
    cdef VALUE_TYPE * ptr_g = < VALUE_TYPE * > < uintptr_t > g._ptr
    cdef VALUE_TYPE * ptr_h = < VALUE_TYPE * > < uintptr_t > h._ptr
    cdef VALUE_TYPE * ptr_i = < VALUE_TYPE * > < uintptr_t > i._ptr
    thrust_backward_gru(H, W, M, V, ptr_a, ptr_b, ptr_c, ptr_d, ptr_e, ptr_f, ptr_g, ptr_h, ptr_i)


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


def cuconcat(gpu_values, gpu_value2, axis):
    for i in range(len(gpu_values[:-1])):
        cuda_base.check_heap_device(gpu_values[i], gpu_values[i + 1], gpu_value2)

    buffer_size = np.sum([val.nbytes for val in gpu_values])
    if gpu_value2.nbytes < buffer_size:
        raise ValueError("Insufficient destination buffer size")

    cdef size_t rec_size = 0
    for gpu_value in gpu_values:
        if (not gpu_value.shape):
            raise ValueError("zero-dimensional arrays cannot be concatenated")
        rec_size += functools.reduce(operator.__mul__, gpu_value.shape[axis:], 1)

    cdef size_t size = 0
    cdef concated_size
    cdef VALUE_TYPE * ptr1
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    for gpu_value in gpu_values:
        s1 = gpu_value.shape[:axis] + gpu_value.shape[axis + 1:]
        concated_size = <int > functools.reduce(operator.__mul__, gpu_value.shape[axis:], 1)
        ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value._ptr
        thrust_copy_memory_stride(ptr2 + size, ptr1, gpu_value.size, rec_size, concated_size)
        size += <int > concated_size


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

    if len(reduce_axis) >= RENOM_CUDA_MAX_AXIS:
        raise ValueError("Number of axis should be less than %d" % RENOM_CUDA_MAX_AXIS)

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


cdef _build_slice_infos(getitem_slice_infos * infos, slices):
    if len(slices) >= RENOM_CUDA_MAX_AXIS:
        raise ValueError("Number of axis should be less than %d" % RENOM_CUDA_MAX_AXIS)

    infos.shape_len = len(slices)
    for i, (start, stop, step, adv_indexes, stride, dest_stride) in enumerate(slices):
        infos.slice_info[i].start = start
        infos.slice_info[i].stop = stop
        infos.slice_info[i].step = step
        if adv_indexes:
            infos.slice_info[i].adv_indexes_len = adv_indexes.size
            infos.slice_info[i].adv_indexes = <long long * > < uintptr_t > adv_indexes._ptr
        else:
            infos.slice_info[i].adv_indexes_len = 0
            infos.slice_info[i].adv_indexes = NULL

        infos.slice_info[i].stride = stride
        infos.slice_info[i].dest_stride = dest_stride


def cu_get_item(gpu_value1, size, dest_size, slices):

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr

    result = renom.core.GPUValue(shape=(dest_size,))
    cdef VALUE_TYPE * ptr_result = <VALUE_TYPE * > < uintptr_t > result._ptr

    cdef getitem_slice_infos infos
    _build_slice_infos( & infos, slices)

    cdef getitem_slice_info * info

    thrust_getitem(ptr1, ptr_result, dest_size, & infos)

    return result


def cu_set_item(value, valuesize, gpu_value1, slices, strides, broadcasted_strides):
    if not isinstance(value, renom.core.GPUValue):
        if isinstance(value, renom.core.Node):
            value = value.get_gpu()
        elif isinstance(value, np.ndarray):
            value = renom.core.GPUValue(array=value)
        else:
            value = renom.core.GPUValue(array=np.array(value))

    if value.dtype.name != gpu_value1.dtype.name:
        raise ValueError()

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > value._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr

    cdef getitem_slice_infos infos
    _build_slice_infos( & infos, slices)

    infos.stride_size = len(strides)
    for i, (s, b) in enumerate(zip(strides, broadcasted_strides)):
        infos.strides[i] = s
        infos.broadcasted_strides[i] = b

    thrust_setitem(ptr1, valuesize, ptr2, & infos)

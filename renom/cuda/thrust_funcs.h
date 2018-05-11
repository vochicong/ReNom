#ifndef THRUST_FUNCS_H__
#define THRUST_FUNCS_H__
#include "cuda_runtime.h"
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


__device__ VALUE_TYPE atomicAdd(VALUE_TYPE *address, const VALUE_TYPE vlaue);

namespace renom{

	// Operation
	enum Operation {MUL, ADD, DIV, RDIV, SUB, POW, RPOW};

	// Negate function
	void thrust_negate(VALUE_TYPE *first, VALUE_TYPE *last, VALUE_TYPE *out);

	// Relu Forward function
	struct relu_forward_function;
	void thrust_relu_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Relu Backward function
	struct relu_backward_function;
	void thrust_relu_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Sigmoid function
	struct sigmoid_function;
	void thrust_sigmoid(VALUE_TYPE *a, VALUE_TYPE *b, int size);

	// Tanh function
	struct tanh_function;
	void thrust_tanh(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    const unsigned int RENOM_CUDA_MAX_STRIDES= 5;

    struct binop_strides {
        size_t size;
        size_t result_strides[RENOM_CUDA_MAX_STRIDES]; // Max 5
        size_t lhs_strides[RENOM_CUDA_MAX_STRIDES];
        size_t rhs_strides[RENOM_CUDA_MAX_STRIDES];
    };

    void thrust_add(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_mul(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_sub(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_div(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_rdiv(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_pow(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_rpow(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);

    void thrust_add_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_mul_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_sub_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_div_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_rdiv_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_pow_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);
    void thrust_rpow_num(VALUE_TYPE *a, VALUE_TYPE b, VALUE_TYPE *c, size_t size);

    __global__ void cuda_copy_memory_stride(VALUE_TYPE *dest, VALUE_TYPE *src, const size_t src_elems,
                             const size_t size_stride, const size_t size_srcblock);

    void thrust_copy_memory_stride(VALUE_TYPE *dest, VALUE_TYPE *src, const size_t src_elems,
                             const size_t size_stride, const size_t size_srcblock);

    // Add bias
    void thrust_add_bias(int size, int n, int wh, VALUE_TYPE *bias, VALUE_TYPE *a);
    __global__ void cuda_add_bias(int size, int n, int wh, VALUE_TYPE *bias, VALUE_TYPE *a);


    // Fill
    void thrust_fill(VALUE_TYPE value, VALUE_TYPE *a, int size);

    // Log e function
    struct loge_function;
    void thrust_loge(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Log e function
    struct exp_function;
    void thrust_exp(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // sqrt function
    struct sqrt_function;
    void thrust_sqrt(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Cross entropy
    struct cross_entropy_function;
    void thrust_cross_entropy(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, int size);

    // abs
    struct abs_forward_function;
    void thrust_abs_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size);
    struct abs_backward_function;
    void thrust_abs_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // sum
    VALUE_TYPE thrust_all_reduce(VALUE_TYPE* a, int size); // sum up all elements.
    __global__ void cuda_strided_sum(VALUE_TYPE *a, VALUE_TYPE *b, int stride, int axis_size, int step, int size);
    void thrust_strided_reduce(VALUE_TYPE* a, VALUE_TYPE* b, int stride, int axis_size, int step, int size);

    // min
    struct min_function;
    void thrust_min(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // max
    struct max_function;
    void thrust_max(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    const unsigned int RENOM_CUDA_MAX_AXIS= 5;

    struct reduce_shape_infos {
        size_t out_size[RENOM_CUDA_MAX_AXIS];
        size_t in_size[RENOM_CUDA_MAX_AXIS];
        size_t group_size[RENOM_CUDA_MAX_AXIS];
    };


    void thrust_reduce_sum(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reductions_infos,
        reduce_shape_infos *seqs_infos);

    void thrust_reduce_max(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reductions_infos,
        reduce_shape_infos *seqs_infos);

    void thrust_reduce_argmax(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        size_t *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reductions_infos,
        reduce_shape_infos *seqs_infos,
        size_t mod, size_t div);

    void thrust_reduce_min(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reductions_infos,
        reduce_shape_infos *seqs_infos);

    void thrust_reduce_argmin(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        size_t *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reductions_infos,
        reduce_shape_infos *seqs_infos,
        size_t mod, size_t div);

    __global__ void cuda_transpose(size_t size, size_t shapesize,
        VALUE_TYPE *src, const size_t src_strides[16],
        VALUE_TYPE *result, const size_t result_strides[16]);

    void thrust_transpose(
        size_t size, size_t shapesize,
        VALUE_TYPE *src, const size_t src_strides[16],
        VALUE_TYPE *result, const size_t result_strides[16]);



    struct getitem_slice_info {
        long long start, stop;
        long long step;

        long long adv_indexes_len;
        long long *adv_indexes;

        size_t stride, dest_stride;
    };

    struct getitem_slice_infos {
        size_t shape_len;
        getitem_slice_info slice_info[16];
        size_t stride_size;
        size_t strides[16];
        size_t broadcasted_strides[16];
    };

    void thrust_getitem(
        VALUE_TYPE *src,
        VALUE_TYPE *result, size_t result_size,
        getitem_slice_infos *info);

    void thrust_setitem(
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *dest,
        getitem_slice_infos *info);

    __global__ void cuda_concat_blocks(VALUE_TYPE *a, const size_t nsize, VALUE_TYPE *b, const size_t block_len, const size_t copy_len);
    void thrust_concat_blocks(VALUE_TYPE *a, const size_t nsize, VALUE_TYPE *b, const size_t block_len, const size_t copy_len);

    struct leaky_relu_forward_function;
    void thrust_leaky_relu_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Leaky Relu backward
    struct leaky_relu_backward_function;
    void thrust_leaky_relu_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    struct elu_forward_function;
    void thrust_elu_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Elu backward
    struct elu_backward_function;
    void thrust_elu_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    // Lstm forward activation without peep hole
    __global__ void cuda_forward_lstm_activate(int N, int M, VALUE_TYPE *u);
    void thrust_forward_lstm_activate(int N, int M, VALUE_TYPE *u);

    // Lstm forward without peep hole
    __global__ void cuda_forward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *s, VALUE_TYPE *ps, VALUE_TYPE *z);
    void thrust_forward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *s, VALUE_TYPE *ps, VALUE_TYPE *z);

    // Lstm backward activation without peep hole
    __global__ void cuda_backward_lstm_activate(int N, int M, VALUE_TYPE *u);
    void thrust_backward_lstm_activate(int N, int M, VALUE_TYPE *u);

    // Lstm backward without peep hole
    __global__ void cuda_backward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *du, VALUE_TYPE *s, VALUE_TYPE *ps, \
                    VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou);
    void thrust_backward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *du, VALUE_TYPE *s, VALUE_TYPE *ps, \
                    VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou);

    // Peephole Lstm forward
    __global__ void cuda_forward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *wc, VALUE_TYPE *pstate, VALUE_TYPE *state, VALUE_TYPE *z);
    void thrust_forward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *wc, VALUE_TYPE *pstate, VALUE_TYPE *state, VALUE_TYPE *z);

    // Peephole Lstm backward
    __global__ void cuda_backward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *prestate, VALUE_TYPE *state, \
            VALUE_TYPE *prefg, VALUE_TYPE *wc, VALUE_TYPE *dy, VALUE_TYPE *drt, \
            VALUE_TYPE *dot, VALUE_TYPE *dr, VALUE_TYPE *dou, VALUE_TYPE *dwc);

    void thrust_backward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *prestate, VALUE_TYPE *state, \
            VALUE_TYPE *prefg, VALUE_TYPE *wc, VALUE_TYPE *dy, VALUE_TYPE *drt, \
            VALUE_TYPE *dot, VALUE_TYPE *dr, VALUE_TYPE *dou, VALUE_TYPE *dwc);

    // Binarize
    void thrust_binarize(VALUE_TYPE *a, VALUE_TYPE prob, int size, VALUE_TYPE *b);
    __global__ void cuda_binarize(VALUE_TYPE *a, VALUE_TYPE prob, int size, VALUE_TYPE *b);

    // Embedding
    void thrust_embedding_forward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *w, VALUE_TYPE *y);
    __global__ void cuda_embedding_forward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *w, VALUE_TYPE *y);

    void thrust_embedding_backward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *dy, VALUE_TYPE *dx);
    __global__ void cuda_embedding_backward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *dy, VALUE_TYPE *dx);

		void thrust_optimizer_sgd(int H, int W, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE momentum, VALUE_TYPE *pdy, VALUE_TYPE *ndy);
		__global__ void cuda_optimizer_sgd(int H, int W, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE momentum, VALUE_TYPE *pdy, VALUE_TYPE *ndy);

		void thrust_optimizer_adagrad(int H, int W, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE *pdy, VALUE_TYPE *ndy, VALUE_TYPE *r);
    __global__ void cuda_optimizer_adagrad(int H, int W, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE *pdy, VALUE_TYPE *ndy, VALUE_TYPE *r);

		void thrust_optimizer_rmsprop(int H, int W, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE gamma, VALUE_TYPE *pdy, VALUE_TYPE *ndy, VALUE_TYPE *r);
    __global__ void cuda_optimizer_rmsprop(int H, int W, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE gamma, VALUE_TYPE *pdy, VALUE_TYPE *ndy, VALUE_TYPE *r);

		void thrust_optimizer_adam(int H, int W, VALUE_TYPE learning_rate, VALUE_TYPE *dy, VALUE_TYPE eps, VALUE_TYPE gamma, VALUE_TYPE gamma_orig, VALUE_TYPE beta, VALUE_TYPE beta_orig, VALUE_TYPE min, bool flug, VALUE_TYPE *u, VALUE_TYPE *r, VALUE_TYPE *ndy);

}
#endif

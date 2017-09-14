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

	// GPU code
	__global__ void cuda_mul(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size);
	__global__ void cuda_add(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size);
	__global__ void cuda_sub(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size);
	__global__ void cuda_div(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size);
	__global__ void cuda_rdiv(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size);
	__global__ void cuda_pow(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size);
	__global__ void cuda_rpow(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size);
	void thrust_operation(Operation op, VALUE_TYPE value, int elem_size_a, VALUE_TYPE *a, int elem_size_b, VALUE_TYPE *b, VALUE_TYPE *c);

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

	///////////////Broad cast
	__global__ void cuda_broad_cast(int elem_size_a, VALUE_TYPE *a, int elem_size_b, VALUE_TYPE *b);
	void thrust_broad_cast(int elem_size_a, VALUE_TYPE *a, int elem_size_b, VALUE_TYPE *b);

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
			VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou, bool temporal);
	void thrust_backward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *du, VALUE_TYPE *s, VALUE_TYPE *ps, \
			VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou, bool temporal);

    // Peephole Lstm forward
    __global__ void cuda_forward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *wc, VALUE_TYPE *pstate, VALUE_TYPE *state, VALUE_TYPE *z);
    void thrust_forward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *wc, VALUE_TYPE *pstate, VALUE_TYPE *state, VALUE_TYPE *z);

    // Peephole Lstm backward
    __global__ void cuda_backward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *prestate, VALUE_TYPE *state, \
            VALUE_TYPE *prefg, VALUE_TYPE *wc, VALUE_TYPE *dy, VALUE_TYPE *drt, \
            VALUE_TYPE *dot, VALUE_TYPE *dr, VALUE_TYPE *dou, VALUE_TYPE *dwc, bool temporal);

    void thrust_backward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *prestate, VALUE_TYPE *state, \
            VALUE_TYPE *prefg, VALUE_TYPE *wc, VALUE_TYPE *dy, VALUE_TYPE *drt, \
            VALUE_TYPE *dot, VALUE_TYPE *dr, VALUE_TYPE *dou, VALUE_TYPE *dwc, bool temporal);

    // Binarize
    void thrust_binarize(VALUE_TYPE *a, VALUE_TYPE prob, int size, VALUE_TYPE *b); 
    __global__ void cuda_binarize(VALUE_TYPE *a, VALUE_TYPE prob, int size, VALUE_TYPE *b);

    // Embedding
    void thrust_embedding_forward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *w, VALUE_TYPE *y);
    __global__ void cuda_embedding_forward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *w, VALUE_TYPE *y);

    void thrust_embedding_backward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *dy, VALUE_TYPE *dx);
    __global__ void cuda_embedding_backward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *dy, VALUE_TYPE *dx);
    
}
#endif

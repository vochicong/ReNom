#include <stdio.h>
#include "thrust_funcs.h"

namespace renom{

	/////////////// Basic operaion
	__global__ void cuda_mul(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= size)
		{
			return;
		}
		if(b != NULL)
			c[idx] = a[(int)(idx%a_step)] * b[(int)(idx%b_step)];
		else
			c[idx] = a[(int)(idx%a_step)] * value;
	}
	
	__global__ void cuda_add(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= size)
		{
			return;
		}
		if(b != NULL)
			c[idx] = a[(int)(idx%a_step)] + b[(int)(idx%b_step)];
		else
			c[idx] = a[(int)(idx%a_step)] + value;
			
	}
	
	__global__ void cuda_sub(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= size)
		{
			return;
		}
		if(b != NULL)
			c[idx] = a[(int)(idx%a_step)] - b[(int)(idx%b_step)];
		else
			c[idx] = a[(int)(idx%a_step)] - value;
	}
	
	__global__ void cuda_div(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= size)
		{
			return;
		}
		if(b != NULL)
			c[idx] = a[(int)(idx%a_step)] / b[(int)(idx%b_step)];
		else
			c[idx] = a[(int)(idx%a_step)] / value;
	}
	
	__global__ void cuda_rdiv(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= size)
		{
			return;
		}
		if(b != NULL)
			c[idx] = b[(int)(idx%b_step)] / a[(int)(idx%a_step)];
		else
			c[idx] = value / a[(int)(idx%a_step)];
	}
	
	__global__ void cuda_pow(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= size)
		{
			return;
		}
		if(b != NULL)
			c[idx] = powf(a[(int)(idx%a_step)], b[(int)(idx%b_step)]);
		else
			c[idx] = powf(a[(int)(idx%a_step)], value);
	}
	
	__global__ void cuda_rpow(VALUE_TYPE value, int a_step, VALUE_TYPE *a, int b_step, VALUE_TYPE *b, VALUE_TYPE *c, size_t size)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= size)
		{
			return;
		}
		if(b != NULL)
			c[idx] = powf(b[(int)(idx%b_step)], a[(int)(idx%a_step)]);
		else
			c[idx] = powf(value, a[(int)(idx%a_step)]);
	}
	
	void thrust_operation(Operation op, VALUE_TYPE value, int elem_size_a, VALUE_TYPE *a, int elem_size_b, VALUE_TYPE *b, VALUE_TYPE *c){
		size_t size = max(elem_size_a, elem_size_b);
		switch(op)
		{
			case MUL:
				cuda_mul <<<ceil((size)/256.0), 256>>> (value, elem_size_a, a, elem_size_b, b, c, size);
				break;
			case ADD:
				cuda_add <<<ceil((size)/256.0), 256>>> (value, elem_size_a, a, elem_size_b, b, c, size);
				break;
			case SUB:
				cuda_sub <<<ceil((size)/256.0), 256>>> (value, elem_size_a, a, elem_size_b, b, c, size);
				break;
			case DIV:
				cuda_div <<<ceil((size)/256.0), 256>>> (value, elem_size_a, a, elem_size_b, b, c, size);
				break;
			case RDIV:
				cuda_rdiv <<<ceil((size)/256.0), 256>>> (value, elem_size_a, a, elem_size_b, b, c, size);
				break;
			case POW:
				cuda_pow <<<ceil((size)/256.0), 256>>> (value, elem_size_a, a, elem_size_b, b, c, size);
				break;
			case RPOW:
				cuda_rpow <<<ceil((size)/256.0), 256>>> (value, elem_size_a, a, elem_size_b, b, c, size);
				break;
		}
	}

	// Negate
	void thrust_negate(VALUE_TYPE *first, VALUE_TYPE *last, VALUE_TYPE *output) {
	    thrust::device_ptr<VALUE_TYPE> dev_first(first);
	    thrust::device_ptr<VALUE_TYPE> dev_last(last);
	    thrust::device_ptr<VALUE_TYPE> dev_output(output);
	
	    thrust::negate<VALUE_TYPE> op;
	    thrust::transform(dev_first, dev_last, dev_output, op);
	}
	
	// Relu forward
	struct relu_forward_function
	{
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return (x > 0)? x:0;
	        }
	};
	
	void thrust_relu_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a((VALUE_TYPE*)a);
		thrust::device_ptr<VALUE_TYPE> dev_b((VALUE_TYPE*)b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, relu_forward_function());
	}
	
	// Relu backward
	struct relu_backward_function
	{
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return (x > 0)? 1:0;
	        }
	};
	
	void thrust_relu_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a((VALUE_TYPE*)a);
		thrust::device_ptr<VALUE_TYPE> dev_b((VALUE_TYPE*)b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, relu_backward_function());
	}
	
	// Leaky Relu forward
	struct leaky_relu_forward_function
	{

		const VALUE_TYPE s;
		leaky_relu_forward_function(VALUE_TYPE s_) : s(s_){}
	
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return (x > 0)? x:x*s;
	        }
	};
	
	void thrust_leaky_relu_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a((VALUE_TYPE*)a);
		thrust::device_ptr<VALUE_TYPE> dev_b((VALUE_TYPE*)b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, leaky_relu_forward_function(s));
	}
	
	// Leaky Relu backward
	struct leaky_relu_backward_function
	{
		const VALUE_TYPE s;
		leaky_relu_backward_function(VALUE_TYPE s_) : s(s_){}
		
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return (x > 0)? 1:0;
	        }
	};
	
	void thrust_leaky_relu_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a((VALUE_TYPE*)a);
		thrust::device_ptr<VALUE_TYPE> dev_b((VALUE_TYPE*)b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, leaky_relu_backward_function(s));
	}
	
	
	// Elu forward
	struct elu_forward_function
	{

		const VALUE_TYPE s;
		elu_forward_function(VALUE_TYPE s_) : s(s_){}
	
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return (x > 0)? x:s*(exp(x) - 1);
	        }
	};
	
	void thrust_elu_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a((VALUE_TYPE*)a);
		thrust::device_ptr<VALUE_TYPE> dev_b((VALUE_TYPE*)b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, elu_forward_function(s));
	}
	
	// Elu backward
	struct elu_backward_function
	{
		const VALUE_TYPE s;
		elu_backward_function(VALUE_TYPE s_) : s(s_){}
		
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return (x > 0)? 1:(x + s);
	        }
	};
	
	void thrust_elu_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a((VALUE_TYPE*)a);
		thrust::device_ptr<VALUE_TYPE> dev_b((VALUE_TYPE*)b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, elu_backward_function(s));
	}
	
	
	
	// Sigmoid
	struct sigmoid_function
	{
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return 1.0/(1.0 + exp(-x));
	        }
	};
	void thrust_sigmoid(VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a((VALUE_TYPE*)a);
		thrust::device_ptr<VALUE_TYPE> dev_b((VALUE_TYPE*)b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, sigmoid_function());
	}
	
	// Tanh
	struct tanh_function
	{
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return tanh(x);
	        }
	};
	void thrust_tanh(VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a((VALUE_TYPE*)a);
		thrust::device_ptr<VALUE_TYPE> dev_b((VALUE_TYPE*)b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, tanh_function());
	}
	
	//fill
	void thrust_fill(VALUE_TYPE value, VALUE_TYPE *a, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_ptr(a);
		thrust::fill(dev_ptr, dev_ptr + size, value);
	}
	
	// loge function
	struct loge_function
	{
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return log(x);
	        }
	};
	void thrust_loge(VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a(a);
		thrust::device_ptr<VALUE_TYPE> dev_b(b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, loge_function());
	}
	
	// loge function
	struct exp_function
	{
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return exp(x);
	        }
	};
	void thrust_exp(VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a(a);
		thrust::device_ptr<VALUE_TYPE> dev_b(b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, exp_function());
	}
	
	// sqrt function
	struct sqrt_function
	{
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return sqrt(x);
	        }
	};
	void thrust_sqrt(VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a(a);
		thrust::device_ptr<VALUE_TYPE> dev_b(b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, sqrt_function());
	}
	
	// Cross entropy
	struct cross_entropy_function
	{
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return y*log(x + 10e-8);
	        }
	};
	
	void thrust_cross_entropy(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, int size){
		thrust::device_ptr<VALUE_TYPE> dev_a(a);
		thrust::device_ptr<VALUE_TYPE> dev_b(b);
		thrust::device_ptr<VALUE_TYPE> dev_c(c);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_c, cross_entropy_function());
	}
	
	///////////////Broad cast
	__global__ void cuda_broad_cast(int elem_size_a, VALUE_TYPE *a, int elem_size_b, VALUE_TYPE *b)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx < elem_size_b)
		{
			for(int i = 0; i < (int)(elem_size_a/elem_size_b);i++)
				b[idx] += a[(int)(idx + elem_size_b*i)];
		}
	}
	
	void thrust_broad_cast(int elem_size_a, VALUE_TYPE *a, int elem_size_b, VALUE_TYPE *b)
	{
		cuda_broad_cast <<<ceil((elem_size_b)/256.0), 256>>> (elem_size_a, a, elem_size_b, b);
	}
	
	// abs
	struct abs_forward_function
	{
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return abs(x);
	        }
	};
	
	void thrust_abs_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a(a);
		thrust::device_ptr<VALUE_TYPE> dev_b(b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, abs_forward_function());
	}
	
	struct abs_backward_function
	{
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return (x > 0)? 1.0:-1.0;
	        }
	};
	
	void thrust_abs_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a(a);
		thrust::device_ptr<VALUE_TYPE> dev_b(b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, abs_backward_function());
	}
	
	// sum
	VALUE_TYPE thrust_all_reduce(VALUE_TYPE* a, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_ptr(a);
		return thrust::reduce(dev_ptr, dev_ptr + size);
	}
	
	__global__ void cuda_strided_sum(VALUE_TYPE *a, VALUE_TYPE *b, int stride, int axis_size, int step, int size)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= size)
			return;
	
		for(int i = 0; i < axis_size; i++)
		{
			b[idx] += a[idx*step + i*stride];
		}
	}
	
	void thrust_strided_reduce(VALUE_TYPE* a, VALUE_TYPE* b, int stride, int axis_size, int step, int size)
	{
		cuda_strided_sum <<<ceil((size/axis_size)/256.0), 256>>> (a, b, stride, axis_size, step, int(size/axis_size));
	}

	// min
	struct min_function
	{
		const VALUE_TYPE m;
		min_function(VALUE_TYPE m_) : m(m_){}
	
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return min(m, x);
	        }
	};
	
	void thrust_min(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a(a);
		thrust::device_ptr<VALUE_TYPE> dev_b(b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, min_function(v));
	}

	// max
	struct max_function
	{
		const VALUE_TYPE m;
		max_function(VALUE_TYPE m_) : m(m_){}
	
	    __host__ __device__
	        VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& y) const { 
	            return max(m, x);
	        }
	};
	
	void thrust_max(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size)
	{
		thrust::device_ptr<VALUE_TYPE> dev_a(a);
		thrust::device_ptr<VALUE_TYPE> dev_b(b);
		thrust::transform(dev_a, dev_a+size, dev_b, dev_b, max_function(v));
	}

	// Lstm forward
	__global__ void cuda_forward_lstm_activate(int N, int M, VALUE_TYPE *u)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		
		if(idx>= N*M) return;
		
		if((idx%M)<M/4)
			u[idx] = tanh(u[idx]);
		else
			u[idx] = 1.0/(1.0 + exp(-u[idx]));
	}
	
	void thrust_forward_lstm_activate(int N, int M, VALUE_TYPE *u)
	{
		cuda_forward_lstm_activate <<<ceil((N*M)/256.0), 256>>> (N, M, u);
	}
	
	__global__ void cuda_forward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *s, VALUE_TYPE *ps, VALUE_TYPE *z)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int size = N*M/4;
		int index = (idx/(M/4))*M + idx%(M/4);
		if(idx < size)
		{
			s[idx] = u[index+M/4*2]*u[index] + u[index+M/4]*ps[idx];
			z[idx] = tanh(s[idx]) * u[index+M/4*3];
		}
		else
		{
			return;
		}
	}
	
	void thrust_forward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *s, VALUE_TYPE *ps, VALUE_TYPE *z)
	{
		cuda_forward_lstm <<<ceil((N*M/4)/256.0), 256>>> (N, M, u, s, ps, z);
	}
	
	// Lstm backward
	__device__ VALUE_TYPE sigmoid_diff(VALUE_TYPE z)
	{
		return z*(1-z);
	}
	
	__device__ VALUE_TYPE tanh_diff(VALUE_TYPE z)
	{
		return 1 - pow(z, 2);
	}
	
	__global__ void cuda_backward_lstm_activate(int N, int M, VALUE_TYPE *u)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		
		if(idx>= N*M) return;
		
		if((idx%M)<M/4)
			u[idx] = tanh(u[idx]);
		else
			u[idx] = 1.0/(1.0 + exp(-u[idx]));
	}
	
	void thrust_backward_lstm_activate(int N, int M, VALUE_TYPE *u)
	{
		cuda_forward_lstm_activate <<<ceil((N*M)/256.0), 256>>> (N, M, u);
	}
	
	__global__ void cuda_backward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *du, VALUE_TYPE *s, VALUE_TYPE *ps, \
			VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int size = N*M/4;
		int index = (idx/(M/4))*M + idx%(M/4);
		
		if(idx < size)
		{
			next_dou[idx] = e[idx]*u[index+M/4*3] * tanh_diff(s[idx]) + pfg[index+M/4]*dou[idx];
			du[index+M/4] = next_dou[idx]*sigmoid_diff(u[index+M/4])*ps[idx];		// f
			du[index+M/4*2] = next_dou[idx]*sigmoid_diff(u[index+M/4*2])*u[index];	// i
			du[index+M/4*3] = e[idx]*s[idx]*sigmoid_diff(u[index+M/4*3]);			// o
			du[index] = next_dou[idx]*tanh_diff(u[index])*u[index+M/4*2];			// c
		}
		else
		{
			return;
		}
	}
	
	void thrust_backward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *du, VALUE_TYPE *s, VALUE_TYPE *ps,\
			VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou)
	{
		cuda_backward_lstm <<<ceil((N*M/4)/256.0), 256>>> (N, M, u, du, s, ps, e, pfg, dou, next_dou);
	}
	
	

}

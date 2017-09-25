#include <stdio.h>
#include <algorithm>
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


        __global__ void cuda_copy_memory_stride(VALUE_TYPE *dest, VALUE_TYPE *src, const size_t src_elems,
                             const size_t size_stride, const size_t size_srcblock) {
            size_t pos = threadIdx.x + blockIdx.x * blockDim.x;
            if (pos < src_elems) {
                size_t n = pos / size_srcblock;
                size_t m = pos % size_srcblock;
                size_t d= n * size_stride + m;
                dest[d] = src[pos];
            }
        }

        // Copy memory block
        void thrust_copy_memory_stride(VALUE_TYPE *dest, VALUE_TYPE *src, const size_t src_elems,
                             const size_t size_stride, const size_t size_srcblock) {
            cuda_copy_memory_stride <<<ceil(src_elems/256.0), 256>>> (dest, src, src_elems, size_stride, size_srcblock);
        }



        __global__ void cuda_reduce_sum_axis(VALUE_TYPE *src, const size_t nsize, const size_t elemsize,
                             const size_t childsize, const size_t axis_shape, VALUE_TYPE *output,
                             const size_t result_len) {


            size_t pos = threadIdx.x + blockIdx.x * blockDim.x;
            output[pos] = 0;
            if (pos < result_len) {
                size_t n = (pos / childsize) * elemsize + (pos % childsize);
                for (size_t i=0; i < axis_shape; i++) {
                    output[pos] += src[n + i*childsize];
                }
            }
        }


        // Reduce vector elems
        void thrust_reduce_sum_axis(VALUE_TYPE *src, const size_t nsize, const size_t elemsize,
                             const size_t childsize, const size_t axis_shape, VALUE_TYPE *output,
                             const size_t result_len)
        {
            cuda_reduce_sum_axis <<<ceil(result_len/256.0), 256>>> (src, nsize, elemsize, childsize, axis_shape, output, result_len);
        }


        __global__ void cuda_sum_blocks(VALUE_TYPE *a, const size_t nsize, const size_t block_len, const size_t part_len) {
            size_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= nsize) {
                return;
            }

            size_t n_block = i / block_len;
            size_t block_top = n_block * block_len;

            size_t offset = i - block_top;
            size_t n_seq = offset / part_len;

            if (n_seq % 2) {
                return;
            }

            if (i + part_len >= block_top + block_len) {
                return;
            }

             
            if (i+part_len >= nsize) {
                return;
            }
            a[i] += a[i+part_len];
        }

        void thrust_sum_blocks(const size_t g, VALUE_TYPE *a, const size_t nsize, const size_t block_len, const size_t part_len) {
            dim3 block(g);
            dim3 grid((nsize + block.x - 1) / block.x, 1);
            cuda_sum_blocks<<<grid, block>>> (a, nsize, block_len, part_len);
        }


        __global__ void cuda_sum_blocks2(VALUE_TYPE *a, const size_t nsize, const size_t elemsize, const size_t childsize) {
//            printf("============= %d\n", childsize);

            size_t num = (nsize-1) / blockDim.x + 1;
            size_t begin = num * threadIdx.x;
            size_t end = begin + num;
            if (end >= nsize) {
                end = nsize;
            }

//            printf("%\d\n", blockDim.x);

            printf("@@@@@@@@ x:%lu begin:%lu end:%lu elem:%lu s:%lu\n", threadIdx.x, begin, end, elemsize, childsize);
            size_t s = childsize;

//            printf("begin: %lu end: %lu\n", begin, end);
            while (s < elemsize) {
                for (size_t i = begin; i < end; i++) {
                    size_t n_block = i / elemsize;

                    size_t block_top = n_block * elemsize;
                    size_t offset = i - block_top;
                    size_t n_seq = offset / s;

//                    printf("==== begin: %lu s: %lu i:%lu block_top: %lu offset: %lu n_seq: %lu\n", begin, s, i, block_top, offset, n_seq);

                    if (n_seq % 2) {
                        continue;
                    }

//                    printf("11111 begin: %lu \n", begin);
                    
                    if (i + s >= block_top + elemsize) {
                        continue;
                    }
             
//                    printf("22222222222 begin: %lu \n", begin);
                    if (i + s >= nsize) {
                        continue;
                    }
//                    printf(">>> x:%lu i:%lu s:%lu\n", threadIdx.x, i, s);
//                    if (i == 0) {
//                        printf("begin:%lu end: %lu s:%lu\n", begin, end, s);
//                    }
                    a[i] += a[i + s];
                }
                __syncthreads();
                s *= 2;
            }
        }


        void thrust_sum_blocks2(VALUE_TYPE *a, const size_t nsize, const size_t elemsize, const size_t childsize) {
            size_t x = (nsize > 1024)?1024:nsize;
            cuda_sum_blocks2<<<1, x>>> (a, nsize, elemsize, childsize);
        }

        __global__ void cuda_concat_blocks(VALUE_TYPE *a, const size_t nsize, VALUE_TYPE *b, const size_t block_len, const size_t copy_len) {
            size_t i = threadIdx.x + blockIdx.x * blockDim.x;
            size_t n_block = i / block_len;
            size_t block_top = n_block * block_len;

            size_t offset = i - block_top;
            if (offset < copy_len) {
                b[n_block*copy_len+offset] = a[i];
            }
        }


        void thrust_concat_blocks(VALUE_TYPE *a, const size_t nsize, VALUE_TYPE *b, const size_t block_len, const size_t copy_len) {
            cuda_concat_blocks<<<ceil(nsize/256.0), 256>>> (a, nsize, b, block_len, copy_len);
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
		cuda_backward_lstm_activate <<<ceil((N*M)/256.0), 256>>> (N, M, u);
	}
	
	__global__ void cuda_backward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *du, VALUE_TYPE *s, VALUE_TYPE *ps, \
			VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou, bool temporal)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int size = N*M/4;
		int index = (idx/(M/4))*M + idx%(M/4);
		
		if(idx < size)
		{
			next_dou[idx] = e[idx]*u[index+M/4*3] * tanh_diff(s[idx]) + ((temporal)?pfg[index+M/4]*dou[idx]:0);
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
			VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou, bool temporal)
	{
		cuda_backward_lstm <<<ceil((N*M/4)/256.0), 256>>> (N, M, u, du, s, ps, e, pfg, dou, next_dou, temporal);
	}

    // Peephole Lstm forward
    __global__ void cuda_forward_peephole_lstm(\
            int N,\
            int M,\
            VALUE_TYPE *u,\
            VALUE_TYPE *wc,\
            VALUE_TYPE *prestate,\
            VALUE_TYPE *state,\
            VALUE_TYPE *z)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = M*N;

        int c3 = idx%M;
        int c4 = (int)(idx/M)*M*4 + idx%M;

        int u4 = c4;
        int f4 = c4+M;
        int i4 = c4+2*M;
        int o4 = c4+3*M;

        int f3 = c3;
        int i3 = c3+M;
        int o3 = c3+2*M;
        
        if(idx>= size) return;

        u[u4] = tanh(u[u4]); // u
        u[f4] = 1.0/(1.0 + exp(-(u[f4]+wc[f3]*prestate[idx]))); // f
        u[i4] = 1.0/(1.0 + exp(-(u[i4]+wc[i3]*prestate[idx]))); // i
        state[idx] = u[u4]*u[i4] + prestate[idx]*u[f4];
        u[o4] = 1.0/(1.0 + exp(-(u[o4]+wc[o3]*state[idx]))); // o
        z[idx] = tanh(state[idx])*u[o4]; // output
    }

    void thrust_forward_peephole_lstm(\
            int N,\
            int M,\
            VALUE_TYPE *wc,\
            VALUE_TYPE *pstate,\
            VALUE_TYPE *state,\
            VALUE_TYPE *u,\
            VALUE_TYPE *z)
    {
        cuda_forward_peephole_lstm <<<ceil((N*M/4)/256.0), 256>>> (N, M/4, wc, pstate, state, u, z);
    }

    // Peephole Lstm backward
    __global__ void cuda_backward_peephole_lstm( \
            int N, \
            int M, \
            VALUE_TYPE *u,  \
            VALUE_TYPE *prestate, \
            VALUE_TYPE *state, \
            VALUE_TYPE *prefg, \
            VALUE_TYPE *wc, \
            VALUE_TYPE *dy, \
            VALUE_TYPE *drt, \
            VALUE_TYPE *dot, \
            VALUE_TYPE *dr, \  // in place 
            VALUE_TYPE *dou, \ // in place
            VALUE_TYPE *dwc, \ // in place
            bool temporal)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = N*M;

        if(idx >= size)return;

        int row3 = (int)(idx/M)*M*3;
        int row4 = (int)(idx/M)*M*4;

        int c3 = idx%M; // Row is not considered.
        int c4 = row4 + idx%M;

        int u4 = c4;
        int f4 = c4+M;
        int i4 = c4+2*M;
        int o4 = c4+3*M;

        int f3 = c3;
        int i3 = c3+M;
        int o3 = c3+2*M;

        VALUE_TYPE tanh_s = tanh(state[idx]);

        dr[o4] = dy[idx] * tanh_s * sigmoid_diff(u[o4]);
        dou[idx] = dy[idx]*u[o4]*tanh_diff(tanh_s) + dr[o4]*wc[o3];
        if(temporal){
            dou[idx] += prefg[f4]*dot[idx];
            dou[idx] += drt[f4]*wc[f3];
            dou[idx] += drt[i4]*wc[i3];

            dwc[f3+row3] = drt[f4]*state[idx];
            dwc[i3+row3] = drt[i4]*state[idx];
        }else{
            dwc[f3+row3] = 0;
            dwc[i3+row3] = 0;
        }
        dwc[o3+row3] = dr[o4]*state[idx];
        dr[f4] = dou[idx] * sigmoid_diff(u[f4]) * prestate[idx];
        dr[i4] = dou[idx] * sigmoid_diff(u[i4]) * u[u4];
        dr[u4] = dou[idx] * tanh_diff(u[u4]) * u[i4];
    }

    void thrust_backward_peephole_lstm( \
            int N, \
            int M, \
            VALUE_TYPE *u,  \
            VALUE_TYPE *prestate, \
            VALUE_TYPE *state, \
            VALUE_TYPE *prefg, \
            VALUE_TYPE *wc, \
            VALUE_TYPE *dy, \
            VALUE_TYPE *drt, \
            VALUE_TYPE *dot, \
            VALUE_TYPE *dr, \
            VALUE_TYPE *dou, \
            VALUE_TYPE *dwc, \
            bool temporal)
    {
        cuda_backward_peephole_lstm <<<ceil((N*M/4)/256.0), 256>>> \
                (N, M/4, u, prestate, state, prefg, wc, dy, drt, dot, dr, dou, dwc, temporal);
    }

    // Binalize
    __global__ void cuda_binalize(VALUE_TYPE *a, VALUE_TYPE prob, int size, VALUE_TYPE *b){
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= size)return;

        if(a[idx] < prob){
            b[idx] = 0.0;
        }else{
            b[idx] = 1.0;
        }
    }

    void thrust_binarize(VALUE_TYPE *a, VALUE_TYPE prob, int size, VALUE_TYPE *b){
        cuda_binalize <<<ceil(size/256.0), 256>>>(a, prob, size, b);
    }

    // Embedding
    __global__ void cuda_embedding_forward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *w, VALUE_TYPE *y)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= N)return;
        for(int i=0; i<M; i++)
        {
            y[idx*M + i] = w[(int)(a[idx])*M + i];
        }
    }

    void thrust_embedding_forward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *w, VALUE_TYPE *y)
    {
        cuda_embedding_forward <<<ceil(N/256.0), 256>>> (N, K, M, a, w, y);
    }


    __global__ void cuda_embedding_backward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *dy, VALUE_TYPE *dx)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= N)return;
        for(int i=0; i<M; i++)
        {
#ifdef USE_RENOM_ATOMICADD
            renom_atomicAdd(&dx[(int)(a[idx])*M + i], dy[idx*M+i]);
#else
            atomicAdd(&dx[(int)(a[idx])*M + i], dy[idx*M+i]);
#endif
        }
    }
    
    void thrust_embedding_backward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *dy, VALUE_TYPE *dx)
    {
        cuda_embedding_backward <<<ceil(N/256.0), 256>>> (N, K, M, a, dy, dx);
    }
}

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

    void thrust_add_bias(int size, int n, int wh, VALUE_TYPE *bias, VALUE_TYPE *a)
    {
        cuda_add_bias <<<ceil((size)/256.0), 256>>> (size, n, wh, bias, a); 
    }

    __global__ void cuda_add_bias(int size, int n, int wh, VALUE_TYPE *bias, VALUE_TYPE *a)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= size)
            return;
        a[idx] += bias[(int)(idx%(size/n)/wh)];
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



        class Reduce_Add {
        public:
            typedef VALUE_TYPE REDUCE_VALUE;
            typedef VALUE_TYPE SRC_VALUE;
            typedef VALUE_TYPE RESULT_VALUE;

            __device__ inline static void set(const size_t pos, const VALUE_TYPE &val, REDUCE_VALUE &ret) {
                ret = val;
            }

            __device__ inline static void reduce_src(const size_t pos, const VALUE_TYPE &val, REDUCE_VALUE &ret) {
                ret += val;
            }

            __device__ inline static void reduce_share(const REDUCE_VALUE &v, REDUCE_VALUE &ret) {
                ret += v;
            }

            __device__ inline static void set_result(const REDUCE_VALUE &v, RESULT_VALUE &ret, const Reduce_Add *) {
                ret = v;
            }

        };

        struct ValueWithPos {
            size_t pos;
            VALUE_TYPE val;
        };

        #define MMIN(l, r) ((l < r) ? (l) : (r))
        #define MMAX(l, r) ((l > r) ? (l) : (r))

        class Reduce_Min {
        public:
            typedef ValueWithPos REDUCE_VALUE;
            typedef VALUE_TYPE SRC_VALUE;
            typedef VALUE_TYPE RESULT_VALUE;

            __device__ inline static void set(const size_t pos, const VALUE_TYPE &val, REDUCE_VALUE &ret) {
                ret.pos = pos;
                ret.val = val;
            }

            __device__ inline static void reduce_src(const size_t pos, const VALUE_TYPE &val, REDUCE_VALUE &ret) {
                if (val < ret.val) {
                    ret.val = val;
                    ret.pos = pos;
                }
                else if (val == ret.val) {
                    ret.pos = MMIN(pos, ret.pos);
                }
            }

            __device__ inline static void reduce_share(const REDUCE_VALUE &v, REDUCE_VALUE &ret) {
                if (v.val < ret.val) {
                    ret.val = v.val;
                    ret.pos = v.pos;
                }
                else if (v.val == ret.val) {
                    ret.pos = MMIN(v.pos, ret.pos);
                }
            }

            __device__ inline static void set_result(const REDUCE_VALUE &v, RESULT_VALUE &ret, Reduce_Min *) {
                ret = v.val;
            }
        };


        class Reduce_ArgMin: public Reduce_Min {
        public:
            typedef size_t RESULT_VALUE;

            size_t mod, div;
            Reduce_ArgMin(size_t n_mod, size_t n_div):mod(n_mod), div(n_div) {}

            __device__ inline static void set_result(const REDUCE_VALUE &v, RESULT_VALUE &ret, Reduce_ArgMin*self) {
                ret = (v.pos % self->mod) / self->div;
            }
        };

        class Reduce_Max{
        public:
            typedef ValueWithPos REDUCE_VALUE;
            typedef VALUE_TYPE SRC_VALUE;
            typedef VALUE_TYPE RESULT_VALUE;

            __device__ inline static void set(const size_t pos, const VALUE_TYPE &val, REDUCE_VALUE &ret) {
                ret.pos = pos;
                ret.val = val;
            }

            __device__ inline static void reduce_src(const size_t pos, const VALUE_TYPE &val, REDUCE_VALUE &ret) {
                if (val > ret.val) {
                    ret.val = val;
                    ret.pos = pos;
                }
                else if (val == ret.val) {
                    ret.pos = MMIN(pos, ret.pos);
                }
            }

            __device__ inline static void reduce_share(const REDUCE_VALUE &v, REDUCE_VALUE &ret) {
                if (v.val > ret.val) {
                    ret.val = v.val;
                    ret.pos = v.pos;
                }
                else if (v.val == ret.val) {
                    ret.pos = MMIN(v.pos, ret.pos);
                }
            }

            __device__ inline static void set_result(const REDUCE_VALUE &v, RESULT_VALUE &ret, Reduce_Max *) {
                ret = v.val;
            }
        };

        class Reduce_ArgMax: public Reduce_Max {
        public:
            typedef size_t RESULT_VALUE;

            size_t mod, div;
            Reduce_ArgMax(size_t n_mod, size_t n_div):mod(n_mod), div(n_div) {}

            __device__ inline static void set_result(const REDUCE_VALUE &v, RESULT_VALUE &ret, Reduce_ArgMax *self) {
                ret = (v.pos % self->mod) / self->div;
            }
        };


        #define CALC_INDEX_STEP(i) {\
            size_t v = n; \
            if (group_size[i]) { \
                v = v % group_size[i]; \
            } \
            v = v / out_size[i]; \
            ret += v * in_size[i]; \
        }

        template <int LEN>
        __device__ inline size_t calc_index_loop(const size_t *out_size, const size_t *in_size, const size_t *group_size, size_t n) {

            
            size_t ret = 0;
            for (int i=0; i < LEN; i++) {
                CALC_INDEX_STEP(i);
            }
            return ret;
        }

        __device__ inline size_t calc_index(int len, const size_t *out_size, const size_t *in_size, const size_t *group_size, size_t sequence_stride, size_t n) {
            size_t ret = 0;

            if (sequence_stride) {
                ret = n % sequence_stride;
            }

            if (len == 1) return ret + calc_index_loop<1>(out_size, in_size, group_size, n);
            if (len == 2) return ret + calc_index_loop<2>(out_size, in_size, group_size, n);
            if (len == 3) return ret + calc_index_loop<3>(out_size, in_size, group_size, n);
            if (len == 4) return ret + calc_index_loop<4>(out_size, in_size, group_size, n);
            if (len == 5) return ret + calc_index_loop<5>(out_size, in_size, group_size, n);
            if (len == 6) return ret + calc_index_loop<6>(out_size, in_size, group_size, n);
            if (len == 7) return ret + calc_index_loop<7>(out_size, in_size, group_size, n);
            if (len == 8) return ret + calc_index_loop<8>(out_size, in_size, group_size, n);
            if (len == 9) return ret + calc_index_loop<9>(out_size, in_size, group_size, n);
            if (len == 10) return ret + calc_index_loop<10>(out_size, in_size, group_size, n);
            if (len == 11) return ret + calc_index_loop<11>(out_size, in_size, group_size, n);
            if (len == 12) return ret + calc_index_loop<12>(out_size, in_size, group_size, n);
            if (len == 13) return ret + calc_index_loop<13>(out_size, in_size, group_size, n);
            if (len == 14) return ret + calc_index_loop<14>(out_size, in_size, group_size, n);
            if (len == 15) return ret + calc_index_loop<15>(out_size, in_size, group_size, n);
            if (len == 16) return ret + calc_index_loop<16>(out_size, in_size, group_size, n);

            assert(0);  // never reach here
            return ret;
        }

        template <typename T>
        __global__ static void cuda_reduce_array(
            size_t num_blocks, size_t num_threads,
            typename T::SRC_VALUE *src, size_t src_size,
            typename T::RESULT_VALUE *result, size_t result_size,
            size_t src_per_result,
            size_t sequence_stride,
            int num_axis,
            reduce_shape_infos reduction_infos,
            reduce_shape_infos seq_infos,
            T adapter) {

            __shared__ typename T::REDUCE_VALUE sharemem[1024];

            size_t blockidx = blockIdx.x;
            size_t threadid = threadIdx.x;

            size_t max_threads_per_result = MMIN(src_per_result, num_threads);
            size_t result_per_block = (result_size - 1) / num_blocks + 1;
            size_t block_result_from =  result_per_block * blockidx;
            size_t block_result_to = MMIN(result_per_block * (blockidx + 1), result_size);
            size_t block_result_step = MMAX(num_threads / max_threads_per_result, 1);

            size_t threads_per_result = MMIN((num_threads-1) / block_result_step + 1, max_threads_per_result);
            size_t src_per_thread = (src_per_result - 1) / threads_per_result + 1;

            size_t *reduction_infos_out_size = &(reduction_infos.out_size[0]);
            size_t *reduction_infos_in_size = &(reduction_infos.in_size[0]);
            size_t *reduction_infos_group_size = &(reduction_infos.group_size[0]);

            size_t *seq_infos_out_size = &(seq_infos.out_size[0]);
            size_t *seq_infos_in_size = &(seq_infos.in_size[0]);
            size_t *seq_infos_group_size = &(seq_infos.group_size[0]);

            for (size_t idx_result_start=block_result_from; 
                 idx_result_start < block_result_to; 
                 idx_result_start += block_result_step ) {

                size_t idx_result = idx_result_start + threadid / threads_per_result;
                if (idx_result >= block_result_to) {
                    continue;
                }

                size_t nth_thread = threadid % threads_per_result;
                size_t nth_in_seq = nth_thread * src_per_thread;

                if (nth_in_seq >= src_per_result) {
                    continue;
                }

                size_t src_top_idx = calc_index(num_axis, reduction_infos_out_size, reduction_infos_in_size, reduction_infos_group_size, sequence_stride, idx_result);
                size_t cur_idx = src_top_idx + calc_index(num_axis, seq_infos_out_size, seq_infos_in_size, seq_infos_group_size, 0, nth_in_seq);

                typename T::REDUCE_VALUE s;
                T::set(cur_idx, src[cur_idx], s);

                size_t sum_to = MMIN(nth_in_seq + src_per_thread, src_per_result);

                for (size_t n=nth_in_seq+1; n < sum_to; n++) {



                    size_t pos = calc_index(num_axis, seq_infos_out_size, seq_infos_in_size, seq_infos_group_size, 0, n);

                    size_t p = src_top_idx + pos;
                    T::reduce_src(p, src[p], s);
//                    s = T::oper(s, src[src_top_idx + pos]);
                }
                

                sharemem[threadid] = s;

                __syncthreads();
                if (nth_thread == 0) {
//                    VALUE_TYPE s = sharemem[threadid];

                    typename T::REDUCE_VALUE s = sharemem[threadid];

                    for (size_t i=1; i < threads_per_result; i++) {
                        size_t nth_in_seq = i * src_per_thread;
                        if (nth_in_seq >= src_per_result) {
                            break;
                        }

                        size_t n = threadid+i;
                        if (n >= num_threads) {
                            break;
                        }

                        T::reduce_share(sharemem[n], s);
//                        s = T::oper(s, sharemem[n]);
                    }
//                    result[idx_result] = s;
                    T::set_result(s, result[idx_result], &adapter);
                }
            }
        }


        template <typename T>
        void static reduce_array(
            size_t num_blocks, size_t num_threads,
            typename T::SRC_VALUE *src, size_t src_size,
            typename T::RESULT_VALUE *result, size_t result_size,
            size_t src_per_result,
            size_t sequence_stride,
            size_t num_axis,
            reduce_shape_infos *reduction_infos,
            reduce_shape_infos *seq_infos, const T &adapter) {

            cuda_reduce_array<T><<<num_blocks, num_threads>>> (
                num_blocks , num_threads, src, src_size, result, result_size, 
                src_per_result,
                sequence_stride, 
                num_axis, *reduction_infos, *seq_infos, adapter);

/*
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("kernel launch failed with error \"%s\".\n",
                       cudaGetErrorString(cudaerr));
           
*/
        }

        void thrust_reduce_sum(
            size_t num_blocks, size_t num_threads,
            VALUE_TYPE *src, size_t src_size,
            VALUE_TYPE *result, size_t result_size,
            size_t src_per_result,
            size_t sequence_stride,
            size_t num_axis,
            reduce_shape_infos *reduction_infos,
            reduce_shape_infos *seq_infos)
        {
            reduce_array<Reduce_Add>(num_blocks, num_threads, src, src_size, result, result_size,
                src_per_result, sequence_stride, num_axis, reduction_infos, seq_infos, Reduce_Add());
        }

        void thrust_reduce_min(
            size_t num_blocks, size_t num_threads,
            VALUE_TYPE *src, const size_t src_size,
            VALUE_TYPE *result, const size_t result_size,
            size_t src_per_result,
            size_t sequence_stride,
            size_t num_axis,
            reduce_shape_infos *reduction_infos,
            reduce_shape_infos *seq_infos)
        {
            reduce_array<Reduce_Min>(num_blocks, num_threads, src, src_size, result, result_size,
                src_per_result, sequence_stride, num_axis, reduction_infos, seq_infos, Reduce_Min());
        }

        void thrust_reduce_argmin(
            size_t num_blocks, size_t num_threads,
            VALUE_TYPE *src, const size_t src_size,
            size_t *result, const size_t result_size,
            size_t src_per_result,
            size_t sequence_stride,
            size_t num_axis,
            reduce_shape_infos *reduction_infos,
            reduce_shape_infos *seq_infos,
            size_t mod, size_t div)
        {
            reduce_array<Reduce_ArgMin>(num_blocks, num_threads, src, src_size, result, result_size,
                src_per_result, sequence_stride, num_axis, reduction_infos, seq_infos, Reduce_ArgMin(mod, div));
        }

        void thrust_reduce_max(
            size_t num_blocks, size_t num_threads,
            VALUE_TYPE *src, const size_t src_size,
            VALUE_TYPE *result, const size_t result_size,
            size_t src_per_result,
            size_t sequence_stride,
            size_t num_axis,
            reduce_shape_infos *reduction_infos,
            reduce_shape_infos *seq_infos)
        {

            reduce_array<Reduce_Max>(num_blocks, num_threads, src, src_size, result, result_size,
                src_per_result, sequence_stride, num_axis, reduction_infos, seq_infos, Reduce_Max());
        }

        void thrust_reduce_argmax(
            size_t num_blocks, size_t num_threads,
            VALUE_TYPE *src, const size_t src_size,
            size_t *result, const size_t result_size,
            size_t src_per_result,
            size_t sequence_stride,
            size_t num_axis,
            reduce_shape_infos *reduction_infos,
            reduce_shape_infos *seq_infos,
            size_t mod, size_t div)
        {

            reduce_array<Reduce_ArgMax>(num_blocks, num_threads, src, src_size, result, result_size,
                src_per_result, sequence_stride, num_axis, reduction_infos, seq_infos, Reduce_ArgMax(mod, div));
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
	};

    struct sign_function
    {
        __host__ __device__
            VALUE_TYPE operator()(const VALUE_TYPE& x, const VALUE_TYPE& b) const {
                return x > 0 ? 1 : (x<0 ? -1 : 0);
            }
    };

    void thrust_sign(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    {
        thrust::device_ptr<VALUE_TYPE> dev_a(a);
        thrust::device_ptr<VALUE_TYPE> dev_b(b);
        thrust::transform(dev_a, dev_a+size, dev_b, dev_b, sign_function());
    };


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

    __global__ void cuda_forward_roi_pool2d(int N, VALUE_TYPE * x, float spatial_scale, int channels,
            int height, int width, int outh, int outw, VALUE_TYPE * rois, VALUE_TYPE *z,
            VALUE_TYPE *argmax_data)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= (N*channels*outw*outh)) return;
        int pw = idx % outw;   //idxw
        int ph = (idx / outw) % outh;  //idxh
        int c = (idx/outw/outh) % channels;
        int num = idx/outw/outh/channels;

        int roi_batch_idx = rois[num * 5 + 0];
        int roi_start_w = round(rois[num * 5 + 1] * spatial_scale);  //xmin
        int roi_start_h = round(rois[num * 5 + 2] * spatial_scale);  //ymin
        int roi_end_w = round(rois[num * 5 + 3] * spatial_scale); //xmax
        int roi_end_h = round(rois[num * 5 + 4] * spatial_scale); //ymax

        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(outh); // strideh
        float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(outw);  // stridew

        int hstart = static_cast<int>(floor(static_cast<float>(ph)*bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<float>(pw)*bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<float>(ph+1)*bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<float>(pw+1)*bin_size_w));
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        float maxval = is_empty ? 0: -1E+37;
        int maxidx = -1;
        int data_offset = (roi_batch_idx * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h)
        {
            for(int w = wstart; w< wend; ++w)
            {
                int bottom_idx = h*width + w;
                if (x[data_offset + bottom_idx] > maxval)
                {
                    maxval = x[data_offset + bottom_idx];
                    maxidx = bottom_idx;
                }

            }
        }

        z[idx] = maxval;
        argmax_data[idx] = maxidx;
    }

    void thrust_forward_roi_pool2d(int N, VALUE_TYPE *x, float spatial_scale, int channels,
            int height, int width, int outh, int outw, VALUE_TYPE *rois, VALUE_TYPE *z,
            VALUE_TYPE *argmax_data)
    {
        cuda_forward_roi_pool2d <<<ceil((N*channels*outh*outw)/256.0), 256>>>(N, x, spatial_scale, channels,
                 height, width, outh, outw, rois, z, argmax_data);
    }


    __global__ void cuda_backward_roi_pool2d(int N, VALUE_TYPE *du ,VALUE_TYPE *argmax, VALUE_TYPE *rois, float spatial_scale, int channels, int height, 
                                            int width, int outh, int outw, VALUE_TYPE *dx)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= (N*channels*outw*outh)) return;
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int num = idx / (width * height * channels);

        float gradient = 0;
        for (int roi_n=0; roi_n < N; ++roi_n ){
            if (num != static_cast<int>(rois[roi_n*5])){
                continue;
            }

            int roi_start_w = round(rois[roi_n*5 + 1]*spatial_scale);
            int roi_start_h = round(rois[roi_n*5 + 2]*spatial_scale);
            int roi_end_w = round(rois[roi_n*5 + 3]*spatial_scale);
            int roi_end_h = round(rois[roi_n*5 + 4]*spatial_scale);

            const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                    h >= roi_start_h && h <= roi_end_h);
            if (!in_roi){
                continue;
            }

            int offset = (roi_n*channels + c) * outh * outw;

            int roi_width = max(roi_end_w - roi_start_w + 1 , 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);

            float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(outh);
            float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(outw);
            int phstart = floor(static_cast<float>(h - roi_start_h)/bin_size_h);
            int phend = ceil(static_cast<float>(h - roi_start_h + 1)/bin_size_h);
            int pwstart = floor(static_cast<float>(w - roi_start_w)/bin_size_w);
            int pwend = ceil(static_cast<float>(w - roi_start_w + 1)/bin_size_w);

            phstart = min(max(phstart, 0), outh);
            phend= min(max(phend, 0), outh);
            pwstart = min(max(pwstart, 0), outw);
            pwend =  min(max(pwend, 0), outw);

            for (int ph=phstart; ph<phend; ++ph){
                for(int pw=pwstart; pw<pwend; ++pw){
                    int index = ph * outw + pw + offset;
                    if(argmax[index] == (h*width + w)){
                        gradient += du[index];
                    }
                }
            }
        }
        dx[idx] = gradient;

    }

    void thrust_backward_roi_pool2d(int N, VALUE_TYPE *du ,VALUE_TYPE *argmax, VALUE_TYPE *rois,
                                        float spatial_scale, int channels, int height, int width, int outh,
                                        int outw, VALUE_TYPE *dx)
    {
        cuda_backward_roi_pool2d <<<ceil((N*channels*height*width)/256.0), 256>>>(N, du, argmax, rois, spatial_scale, channels, height,
                                                                                width, outh, outw, dx);
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
            VALUE_TYPE *dwc // in place
        )
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
        dou[idx] += prefg[f4]*dot[idx];
        dou[idx] += drt[f4]*wc[f3];
        dou[idx] += drt[i4]*wc[i3];

        dwc[f3+row3] = drt[f4]*state[idx];
        dwc[i3+row3] = drt[i4]*state[idx];
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
            VALUE_TYPE *dwc
        )
    {
        cuda_backward_peephole_lstm <<<ceil((N*M/4)/256.0), 256>>> \
                (N, M/4, u, prestate, state, prefg, wc, dy, drt, dot, dr, dou, dwc);
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
    
    __global__ void cuda_get_fg_ary_forward(int N, int M, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
        if((idx/M)%2 == 0){
            ptr2[M*(idx/M/2) + (idx%M)] = ptr1[idx];
        }
    }

    void thrust_get_fg_ary_forward(int N, int M, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2){
        cuda_get_fg_ary_forward <<<ceil(N/256.0), 256>>> (N, M, ptr1, ptr2);
    }

    __global__ void cuda_get_fg_ary_backward(int N, int M, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
        if((idx/M)%2 == 0){
            ptr2[idx] = ptr1[M*(idx/M/2) + (idx%M)];
        }
    }

    void thrust_get_fg_ary_backward(int N, int M, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2){
        cuda_get_fg_ary_forward <<<ceil(N/256.0), 256>>> (N, M, ptr1, ptr2);
    }

    __global__ void cuda_get_ith_ary_forward(int N, int M, int i, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
        if(i * M <= idx && (i+1)*M){
            ptr2[idx%M] = ptr1[idx];
        }
    }

    void thrust_get_ith_ary_forward(int N, int M, int i, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2){
        cuda_get_ith_ary_forward <<<ceil(N/256.0), 256>>> (N, M, i, ptr1, ptr2);
    }

    __global__ void cuda_get_ith_ary_backward(int N, int M, int i, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
        if(i * M <= idx && (i+1)*M){
            ptr2[idx] = ptr1[idx%M];
        }
    }

    void thrust_get_ith_ary_backward(int N, int M, int i, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2){
        cuda_get_ith_ary_backward <<<ceil(N/256.0), 256>>> (N, M, i, ptr1, ptr2);
    }

    __global__ void cuda_get_nth_ary(int N, int M, int i, int j, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= N*M) return;
        if (idx %j == i){
            ptr2[idx/j] = ptr1[idx];
        }
    }
    void thrust_get_nth_ary(int N, int M, int i, int j, VALUE_TYPE *ptr1, VALUE_TYPE *ptr2){
        cuda_get_nth_ary <<<ceil((N*M)/256.0), 256.0>>> (N, M, i, j, ptr1, ptr2);
    }

    __global__ void cuda_assign_pred_box(int N, int M, VALUE_TYPE *x_ptr, VALUE_TYPE *y_ptr, VALUE_TYPE *h_ptr, VALUE_TYPE *w_ptr, VALUE_TYPE *ary_ptr)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= N*M) return;
        switch(idx % 4){
            case 0:
                ary_ptr[idx] = x_ptr[idx/4] - 0.5 * w_ptr[idx/4];
                break;
            case 1:
                ary_ptr[idx] = y_ptr[idx/4] - 0.5 * h_ptr[idx/4];
                break;
            case 2:
                ary_ptr[idx] = x_ptr[idx/4] + 0.5 * w_ptr[idx/4];
                break;
            case 3:
                ary_ptr[idx] = y_ptr[idx/4] + 0.5 * h_ptr[idx/4];
                break;
        };
    }

    void thrust_assign_pred_box(int N, int M, VALUE_TYPE *x_ptr, VALUE_TYPE *y_ptr, VALUE_TYPE *h_ptr, VALUE_TYPE *w_ptr, VALUE_TYPE *ary_ptr)
    {
        cuda_assign_pred_box <<<ceil((N*M)/256.0), 256.0>>> (N, M, x_ptr, y_ptr, h_ptr, w_ptr, ary_ptr);
    }

    __global__ void cuda_pred_ctr(int N, int M, VALUE_TYPE *arg_ptr, VALUE_TYPE *length_ptr, VALUE_TYPE *ctr_ptr, VALUE_TYPE *ary_ptr)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= M*N) return;
        ary_ptr[idx] = arg_ptr[idx] * length_ptr[idx/M] + ctr_ptr[idx/M];
    }

    void thrust_pred_ctr(int N, int M, VALUE_TYPE *arg_ptr, VALUE_TYPE *length_ptr,VALUE_TYPE *ctr_ptr, VALUE_TYPE *ary_ptr)
    {
        cuda_pred_ctr <<<ceil((N*M)/256.0), 256.0 >>> (N, M, arg_ptr, length_ptr, ctr_ptr, ary_ptr);
    }

    __global__ void cuda_generate_anchors(int A, int K, int N, VALUE_TYPE *shifts_ptr, VALUE_TYPE *ratios_ptr, VALUE_TYPE *scales_ptr, int ratio_size, int scale_size, int feat_stride, int base_size, VALUE_TYPE *anchors_ptr)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx>=A*K*N) return;
        int shift_idx = idx / (N * A);
        int base_idx = (idx-(shift_idx*N*A))/N;
        int cord_idx = idx % N;
        int i = base_idx / scale_size;
        int j = base_idx % scale_size;
        float h = float(base_size) * float(scales_ptr[j]) * std::sqrt(float(ratios_ptr[i]));
        float w = float(base_size) * float(scales_ptr[j]) * std::sqrt(1.0/float(ratios_ptr[i]));
        switch(cord_idx){
            case 0:
                anchors_ptr[idx] = float(shifts_ptr[shift_idx*4+cord_idx]) +  (float(base_size)/2.0 - float(w) / 2.0);
                break;
            case 1:
                anchors_ptr[idx] = float(shifts_ptr[shift_idx*4+cord_idx]) +  (float(base_size)/2.0 - float(h) / 2.0);
                break;
            case 2:
                anchors_ptr[idx] = float(shifts_ptr[shift_idx*4+cord_idx]) +  (float(base_size)/2.0 + float(w) / 2.0);
                break;
            case 3:
                anchors_ptr[idx] = float(shifts_ptr[shift_idx*4+cord_idx]) +  (float(base_size)/2.0 + float(h) / 2.0);
                break;
        }
    }

    void thrust_generate_anchors(int A, int K, int N, VALUE_TYPE *shifts_ptr, VALUE_TYPE *ratios_ptr, VALUE_TYPE *scales_ptr, int ratio_size, int scale_size, int feat_stride, int base_size, VALUE_TYPE *anchors_ptr)
    {
        cuda_generate_anchors <<<ceil(A*K*N/256.0), 256.0>>>(A, K, N, shifts_ptr, ratios_ptr, scales_ptr, ratio_size, scale_size, feat_stride, base_size, anchors_ptr);
    }

    __global__ void cuda_get_ith_bbox(int N, int M, VALUE_TYPE *bbox_ptr, int i, VALUE_TYPE *ary_ptr)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx>=N*M) return;
        if (idx%M==i){
            ary_ptr[idx/M] = bbox_ptr[idx];
        }


    }
    void thrust_get_ith_bbox(int N, int M, VALUE_TYPE *bbox_ptr, int i, VALUE_TYPE *ary_ptr)
    {
        cuda_get_ith_bbox <<<ceil(N*M/256.0), 256.0>>>(N, M, bbox_ptr, i, ary_ptr);
    }

    __global__ void cuda_clip_roi(int N, int M, VALUE_TYPE *roi_ptr, int start, int end, int step, int min_v, int max_v, VALUE_TYPE *ary_ptr)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx>=N*M) return;
        if ((idx/N)%step == start){
            ary_ptr[idx] = fmax(float(min_v), fmin(float(roi_ptr[idx]), float(max_v)));
        } else {
            ary_ptr[idx] = roi_ptr[idx];
        }

    }
    void thrust_clip_roi(int N, int M, VALUE_TYPE *roi_ptr, int start, int end, int step, int min_v, int max_v, VALUE_TYPE *ary_ptr)
    {
        cuda_clip_roi <<<ceil(N*M/256.0), 256.0>>>(N, M, roi_ptr, start, end, step, min_v, max_v, ary_ptr);
    }
}

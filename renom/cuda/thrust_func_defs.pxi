from libcpp cimport bool

cdef extern from * namespace "renom":
    cdef void thrust_negate(VALUE_TYPE* first, VALUE_TYPE *last, VALUE_TYPE *output)
    cdef void thrust_relu_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_relu_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_sigmoid(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_tanh(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_copy_memory_stride(VALUE_TYPE *dest, VALUE_TYPE *src, const size_t src_elems,
                             const size_t size_stride, const size_t size_srcblock)
    cdef void thrust_fill(VALUE_TYPE value, VALUE_TYPE *a, int size)
    cdef void thrust_loge(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_exp(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_sqrt(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_cross_entropy(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, int size)
    cdef void thrust_abs_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_abs_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef VALUE_TYPE thrust_all_reduce(VALUE_TYPE* a, int size)
    cdef void thrust_strided_reduce(VALUE_TYPE* a, VALUE_TYPE* b, int stride, int axis_size, int step, int size);
    cdef void thrust_create_mask(VALUE_TYPE *a, int size)
    cdef void thrust_min(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_max(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size);

    cdef struct binop_strides:
        size_t size
        size_t result_strides[16]
        size_t lhs_strides[16]
        size_t rhs_strides[16]

    void thrust_add(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_mul(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_sub(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_div(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_rdiv(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_pow(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);
    void thrust_rpow(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, size_t size, binop_strides *strides);

    cdef struct reduce_shape_infos:
        size_t out_size[16]
        size_t in_size[16]
        size_t group_size[16]

    cdef void thrust_reduce_sum(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reduction_infos,
        reduce_shape_infos *seq_infos)

    cdef void thrust_reduce_min(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        VALUE_TYPE *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reduction_infos,
        reduce_shape_infos *seq_infos)

    cdef void thrust_reduce_argmin(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, size_t src_size,
        size_t *result, size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reduction_infos,
        reduce_shape_infos *seq_infos,
        size_t mod, size_t div)

    cdef void thrust_reduce_max(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, const size_t src_size,
        VALUE_TYPE *result, const size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reduction_infos,
        reduce_shape_infos *seq_infos)

    cdef void thrust_reduce_argmax(
        size_t num_blocks, size_t num_threads,
        VALUE_TYPE *src, const size_t src_size,
        size_t *result, const size_t result_size,
        size_t src_per_result,
        size_t sequence_stride,
        size_t num_axis,
        reduce_shape_infos *reduction_infos,
        reduce_shape_infos *seq_infos,
        size_t mod, size_t div);

    cdef void thrust_transpose(
        size_t size, size_t shapesize,
        VALUE_TYPE *src, const size_t src_strides[16],
        VALUE_TYPE *result, const size_t result_strides[16]);

    cdef void thrust_concat_blocks(VALUE_TYPE *a, const size_t nsize, VALUE_TYPE *b, const size_t block_len, const size_t copy_len)

    cdef void thrust_leaky_relu_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_leaky_relu_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_elu_forward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_elu_backward(VALUE_TYPE s, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_forward_lstm_activate(int N, int M, VALUE_TYPE *u);
    cdef void thrust_forward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *s, VALUE_TYPE *ps, VALUE_TYPE *z);
    cdef void thrust_backward_lstm(int N, int M, VALUE_TYPE *u, VALUE_TYPE *du, VALUE_TYPE *s, VALUE_TYPE *ps,
            VALUE_TYPE *e, VALUE_TYPE *pfg, VALUE_TYPE *dou, VALUE_TYPE *next_dou);

    cdef void thrust_forward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *wc, VALUE_TYPE *prestate, VALUE_TYPE *state, VALUE_TYPE *z)

    cdef void thrust_backward_peephole_lstm\
        (int N, int M, VALUE_TYPE *u, VALUE_TYPE *prestate, VALUE_TYPE *state, VALUE_TYPE *prefg, VALUE_TYPE *wc,\
             VALUE_TYPE *dy, VALUE_TYPE *drt, VALUE_TYPE *dot, VALUE_TYPE *dr, VALUE_TYPE *dou, VALUE_TYPE *dwc);

    cdef void thrust_binarize(VALUE_TYPE *a, VALUE_TYPE prob, int size, VALUE_TYPE *b);
    cdef void thrust_embedding_forward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *w, VALUE_TYPE *y);

    cdef void thrust_embedding_backward(int N, int K, int M, VALUE_TYPE *a, VALUE_TYPE *dy, VALUE_TYPE *dx);
    cdef void thrust_add_bias(int size, int n, int wh, VALUE_TYPE *bias, VALUE_TYPE *a);


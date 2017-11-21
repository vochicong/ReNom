from libcpp cimport bool

cdef extern from * namespace "renom":
    ctypedef enum Operation:
        MUL, ADD, DIV, RDIV, SUB, POW, RPOW
    cdef void thrust_negate(VALUE_TYPE* first, VALUE_TYPE *last, VALUE_TYPE *output)
    cdef void thrust_relu_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_relu_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_sigmoid(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_tanh(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_operation(Operation op, VALUE_TYPE value, int elem_size_a, VALUE_TYPE *a, int elem_size_b, VALUE_TYPE *b, VALUE_TYPE *c)
    cdef void thrust_copy_memory_stride(VALUE_TYPE *dest, VALUE_TYPE *src, const size_t src_elems,
                             const size_t size_stride, const size_t size_srcblock)
    cdef void thrust_fill(VALUE_TYPE value, VALUE_TYPE *a, int size)
    cdef void thrust_loge(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_exp(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_sqrt(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_cross_entropy(VALUE_TYPE *a, VALUE_TYPE *b, VALUE_TYPE *c, int size)
    cdef void thrust_broad_cast(int elem_size_a, VALUE_TYPE *a, int elem_size_b, VALUE_TYPE *b)
    cdef void thrust_abs_forward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef void thrust_abs_backward(VALUE_TYPE *a, VALUE_TYPE *b, int size)
    cdef VALUE_TYPE thrust_all_reduce(VALUE_TYPE* a, int size)
    cdef void thrust_strided_reduce(VALUE_TYPE* a, VALUE_TYPE* b, int stride, int axis_size, int step, int size);
    cdef void thrust_create_mask(VALUE_TYPE *a, int size)
    cdef void thrust_min(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_max(VALUE_TYPE v, VALUE_TYPE *a, VALUE_TYPE *b, int size);
    cdef void thrust_reduce_sum(VALUE_TYPE *a, const size_t nsize,
                                 const size_t axis_size, const size_t elem_size,
                                 const size_t child_size, VALUE_TYPE *b,
                                 const size_t result_size)

    cdef void thrust_reduce_min(VALUE_TYPE *a, const size_t nsize,
                                 const size_t axis_size, const size_t elem_size,
                                 const size_t child_size, VALUE_TYPE *b,
                                 const size_t result_size)
    cdef void thrust_reduce_max(VALUE_TYPE *a, const size_t nsize,
                                 const size_t axis_size, const size_t elem_size,
                                 const size_t child_size, VALUE_TYPE *b,
                                 const size_t result_size)

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

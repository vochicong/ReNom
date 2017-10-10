# -*- coding: utf-8 -*.t-
from __future__ import print_function, division

import numpy as np
from renom.core import Node, get_gpu, GPUValue, BinOp, UnaryOp, to_value
from renom.config import precision

try:
    from renom.cuda.cublas import *
    from renom.cuda.cuda_base import *
    if precision == np.float32:
        from renom.cuda.thrust_float import *
    else:
        from renom.cuda.thrust_double import *
except ImportError:
    pass


class reshape(Node):
    """This function reshapes matrix shape.

    Args:
        array (Variable): Input array.
        shape (tuple): Shape.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> x = rm.Variable(np.arange(6))
        >>> x.shape
        (6,)
        >>> y = rm.reshape(x, (2, 3))
        >>> y.shape
        (2, 3)
    """

    @classmethod
    def _oper_cpu(cls, array, shape):
        return array.reshape(shape).copy()

    @classmethod
    def _oper_gpu(cls, array, shape):
        return get_gpu(array).reshape(shape)

    def __new__(cls, array, shape):
        value = cls.calc_value(array, shape)
        ret = super(reshape, cls).__new__(cls, value)
        ret.attrs._array = array
        ret.attrs._shape = array.shape
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._array, Node):
            self.attrs._array._update_diff(context, dy.reshape(self.attrs._shape), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._array, Node):
            self.attrs._array._update_diff(context, get_gpu(
                dy).reshape(self.attrs._shape), **kwargs)


class sum(Node):
    '''
    This function sums up matrix elements.
    In the current version(2.0), summation along 1st axis and
    summation of all elements are supported.

    Args:
        array (Variable): Input array.
        axis (int): Summing up along specified axis.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> z = rm.sum(x)
        >>> z
        sum(3.21392822265625, dtype=float32)
    '''

    @classmethod
    def _oper_cpu(cls, arg, axis=None):
        return np.sum(arg, axis=axis)

    @classmethod
    def _oper_gpu(cls, arg, axis=None):
        return cusum(get_gpu(arg), axis=axis)

    def __new__(cls, arg, axis=None):
        assert not hasattr(axis, "__getitem__"), "The argument axis only accepts integer."
        assert (axis is None) or (axis < 1)
        value = cls.calc_value(arg, axis)
        ret = super(sum, cls).__new__(cls, value)
        ret.attrs._axis = axis
        ret.attrs._arg = arg
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = np.ones_like(self.attrs._arg) * dy
            self.attrs._arg._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = get_gpu(self.attrs._arg).ones_like_me() * get_gpu(dy)
            self.attrs._arg._update_diff(context, dx, **kwargs)


class dot(BinOp):
    '''
    This function executes dot product of the two matrixes.

    Args:
        lhs (Variable,ndarray): Input array.
        rhs (Variable,ndarray): Input array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> y = np.random.rand(2, 2)
        >>> z = rm.dot(y, x)
        >>> z
        dot([[ 0.10709135,  0.15022227,  0.12853521],
             [ 0.30557284,  0.32320538,  0.26753256]], dtype=float32)
    '''

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.dot(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        new_shape = (lhs.shape[0], rhs.shape[1])
        ret = GPUValue(shape=new_shape)
        cublas_gemm(get_gpu(lhs), 0,
                    get_gpu(rhs), 0,
                    get_gpu(ret))
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, np.dot(dy, to_value(self.attrs._rhs).T), **kwargs)

        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, np.dot(to_value(self.attrs._lhs).T, dy), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        lhs = self.attrs._lhs
        rhs = self.attrs._rhs
        if isinstance(self.attrs._lhs, Node):
            new_shape = lhs.shape
            ldx = GPUValue(shape=new_shape)
            cublas_gemm(get_gpu(dy), 0,
                        get_gpu(rhs), 1,
                        get_gpu(ldx))
            self.attrs._lhs._update_diff(context, ldx, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            new_shape = rhs.shape
            rdx = GPUValue(shape=new_shape)
            cublas_gemm(get_gpu(lhs), 1,
                        get_gpu(dy), 0,
                        get_gpu(rdx))
            self.attrs._rhs._update_diff(context, rdx, **kwargs)


class concat(BinOp):
    """
    Join a sequence of arrays along an existing axis.
    In the current version(2.0), concatenation along 2nd axis is
    only supported.

    Args:
        lhs (Variable,ndarray): Input array.
        rhs (Variable,ndarray): Input array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> y = np.random.rand(2, 2)
        >>> z = rm.concat(x, y)
        >>> z.shape
        (2, 5)
        >>> z
        concat([[ 0.56989014,  0.50372809,  0.40573129,  0.17601326,  0.07233092],
                [ 0.09377897,  0.8510806 ,  0.78971916,  0.52481949,  0.06913455]], dtype=float32)

    """

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.hstack((lhs, rhs))

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        axis = 1
        newshape = lhs.shape[:axis] + (lhs.shape[axis] + rhs.shape[axis],) + lhs.shape[axis + 1:]

        ret = GPUValue(shape=newshape)
        cuconcat(get_gpu(lhs), get_gpu(rhs), ret, axis)
        return ret

    def __new__(cls, lhs, rhs):
        ret = super(concat, cls).__new__(cls, lhs, rhs)
        ret.attrs._index = lhs.shape[1]
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        ldy, rdy = np.hsplit(to_value(dy), [self.attrs._index])
        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, ldy, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, rdy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        ldy, rdy = np.hsplit(get_gpu(dy).new_array(), [self.attrs._index])
        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, GPUValue(ldy), **kwargs)

        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, GPUValue(rdy), **kwargs)


class where(Node):
    """
    Return elements, either from a or b, depending on condition.

    Args:
        condition (Variable,ndarray): Condition array.
        a (Variable,ndarray): Input array.
        b (Variable,ndarray): Input array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> x
        array([[ 0.56989017,  0.50372811,  0.4057313 ],
               [ 0.09377897,  0.85108059,  0.78971919]])
        >>> z = rm.where(x > 0.5, x, 0)
        >>> z
        where([[ 0.56989014,  0.50372809,  0.        ],
               [ 0.        ,  0.8510806 ,  0.78971916]], dtype=float32)

    """

    @classmethod
    def _oper_cpu(cls, condition, a, b):
        return np.where(condition, a, b)

    @classmethod
    def _oper_gpu(cls, condition, a, b):
        a_cpu = getattr(get_gpu(a), "new_array()", a)
        b_cpu = getattr(get_gpu(b), "new_array()", b)
        ret = GPUValue(np.where(condition, a_cpu, b_cpu))
        return ret

    def __new__(cls, condition, a, b):
        value = cls.calc_value(condition, a, b)
        ret = super(where, cls).__new__(cls, value)
        ret.attrs._condition = condition
        ret.attrs._a, ret.attrs._b = a, b
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._a, Node):
            ldy = np.zeros_like(self.attrs._a)
            ldy[self.attrs._condition] = dy[self.attrs._condition]
            self.attrs._a._update_diff(context, ldy, **kwargs)

        if isinstance(self.attrs._b, Node):
            rdy = np.zeros_like(self.attrs._b)
            rdy[- self.attrs._condition] = dy[- self.attrs._condition]
            self.attrs._b._update_diff(context, rdy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._a, Node):
            ldy = get_gpu(self.attrs._a).zeros_like_me()
            ldy[self.attrs._condition] = dy[self.attrs._condition]
            self.attrs._a._update_diff(context, ldy, **kwargs)

        if isinstance(self.attrs._b, Node):
            rdy = get_gpu(self.attrs._b).zeros_like_me()
            rdy[- self.attrs._condition] = dy[- self.attrs._condition]
            self.attrs._b._update_diff(context, rdy, **kwargs)


class sqrt(UnaryOp):
    """
    Square root operation.

    Args:
        arg (Variable,ndarray): Input array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> x
        array([[ 0.56989017,  0.50372811,  0.4057313 ],
               [ 0.09377897,  0.85108059,  0.78971919]])
        >>> z = rm.sqrt(x)
        >>> z
        sqrt([[ 0.75491071,  0.70973808,  0.6369704 ],
              [ 0.30623353,  0.92254031,  0.88866144]], dtype=float32)
    """

    @classmethod
    def _oper_cpu(cls, arg):
        return np.sqrt(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = GPUValue(shape=arg.shape)
        cusqrt(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = np.power(self, -0.5) / 2
            self.attrs._arg._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = (self**-0.5) * 0.5
            self.attrs._arg._update_diff(context, dx, **kwargs)


class log(UnaryOp):
    """
    Log operation.

    Args:
        arg (Variable,ndarray): Input array.
    """

    @classmethod
    def _oper_cpu(cls, arg):
        return np.log(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        culoge(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy / self.attrs._arg, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy / get_gpu(self.attrs._arg), **kwargs)


class exp(UnaryOp):
    """
    Exponential operation.

    Args:
        arg (Variable,ndarray): Input array.
    """

    @classmethod
    def _oper_cpu(cls, arg):
        return np.exp(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cuexp(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy * self, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy * get_gpu(self), **kwargs)


# Not implemented.
class mean(Node):
    pass

# Not implemented.


class max(Node):
    pass

# -*- coding: utf-8 -*.t-
from __future__ import print_function, division

import numpy as np
from renom.core import Node, get_gpu, GPUValue, BinOp, UnaryOp, to_value, Reshape, Amin, Amax
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


def reshape(array, shape):
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
    return Reshape(array, shape)


class sum(Node):
    '''
    This function sums up matrix elements.
    If the argument 'axis' is passed, this function performs
    sum along specified axis.

    Args:
        array (Variable): Input array.
        axis (int): Summing up along this axis.

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
        value = cls.calc_value(arg, axis)
        ret = super(sum, cls).__new__(cls, value)
        ret.attrs._axis = axis
        ret.attrs._arg = arg
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            arg = self.attrs._arg
            axis = self.attrs._axis
            if axis is None or axis == 0:
                dx = np.ones_like(arg) * dy
            else:
                dx = np.ones_like(arg) * np.expand_dims(dy, axis)
            arg._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            arg = self.attrs._arg
            axis = self.attrs._axis
            if axis is None or axis == 0:
                dx = get_gpu(arg).ones_like_me() * get_gpu(dy)
            else:
                dy = get_gpu(dy).new_array()
                dx = np.ones(arg.shape, dtype=arg.dtype) * np.expand_dims(dy, axis=axis)
            arg._update_diff(context, get_gpu(dx), **kwargs)


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


class concat(Node):
    """
    Join a sequence of arrays along specified axis.

    Args:
        args (*Variable, tuple): Input arrays or tuple of input arrays.
        axis (int): Concatenation will be performed along this axis.

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
    def _oper_cpu(cls, args, axis):
        return np.concatenate(args, axis=axis).copy()

    @classmethod
    def _oper_gpu(cls, args, axis):
        newshape = args[0].shape[:axis] + \
            (np.sum([a.shape[axis] for a in args]), ) + args[0].shape[axis + 1:]

        ret = GPUValue(shape=newshape)
        cuconcat([get_gpu(a) for a in args], ret, axis)
        return ret

    def __new__(cls, *args, axis=1):
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        assert all([len(args[0].shape) == len(args[i].shape) for i in range(1, len(args))]), \
            "All arguments must have same number of dimension."
        assert np.sum(np.sum(np.array([list(map(lambda x, y: x != y,
                                                args[0].shape, args[i].shape)) for i in range(1, len(args))]), axis=0).astype(np.bool)) < 2, \
            "All dimensions must have same size except specified axis."

        val = cls.calc_value(args, axis)
        ret = super(concat, cls).__new__(cls, val)
        tmp = 0
        index = []
        for a in args[:-1]:
            tmp += a.shape[axis]
            index.append(tmp)
        ret.attrs._index = index
        ret.attrs._axis = axis
        for i, v in enumerate(args):
            setattr(ret.attrs, "_arg%d" % i, args[i])
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        axis = self.attrs._axis
        args = np.split(to_value(dy), self.attrs._index, axis=axis)
        for i in range(len(self.attrs._index) + 1):
            arg = getattr(self.attrs, "_arg%d" % i)
            if isinstance(arg, Node):
                arg._update_diff(context, args[i], **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        axis = self.attrs._axis
        args = get_gpu(dy).split(self.attrs._index, axis=axis)
        for i in range(len(self.attrs._index) + 1):
            arg = getattr(self.attrs, "_arg%d" % i)
            if isinstance(arg, Node):
                arg._update_diff(context, args[i], **kwargs)


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

class sign(UnaryOp):
    """
    Sign operation

    Args:
        arg (Variable, ndarray): Input array.
    """

    @classmethod
    def _oper_cpu(cls, arg):
        return np.sign(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cusign(get_gpu(arg), ret)
        return ret

    #def _backward_cpu(self, context, dy, **kwargs):
    #    if isinstance(self.attrs._lhs, Node):
    #        zero = np.zeros_like(np.array(self.attrs._lhs))
   #         self.attrs._lhs._update_diff(context, zero, **kwargs)
   # def _backward_gpu(self, context, dy, **kwargs):
   #     if isinstance(self.attrs._lhs, Node):
   #         zero = get_gpu(self.attrs._lhs, Node)
   #         self.attrs._lhs._update_diff(context, zero, **kwargs)


class amin(Amin):
    pass


class amax(Amax):
    pass

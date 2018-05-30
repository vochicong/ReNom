# -*- coding: utf-8 -*-
from __future__ import print_function, division
import collections
import weakref
from . config import precision
import contextlib
import numpy as np
from numbers import Number

from renom.cuda import *
from renom.cuda.gpuvalue import *
from .debug_graph import *


class Grads:
    '''Grads class. This class contains gradients of each Node object.

    When the function ``grad`` which is instance of Node class is called,
    an instance of Grads class will be returned.

    For getting the gradient with respect to a Variable object 'x' which is on a
    computational graph, call the 'get' function of Grads object (An example is bellow).

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> a = rm.Variable(np.random.rand(2, 3))
        >>> b = rm.Variable(np.random.rand(2, 3))
        >>> c = rm.sum(a + 2*b)
        >>> grad = c.grad()
        >>> grad.get(a)
        Mul([[ 1.,  1.,  1.],
             [ 1.,  1.,  1.]], dtype=float32)
        >>> grad.get(b)
        RMul([[ 2.,  2.,  2.],
              [ 2.,  2.,  2.]], dtype=float32)
    '''

    def __init__(self, root=None):
        self.stroage = {}
        self.variables = {}
        self._auto_updates = []

        if root is not None:
            self._build_refcounts(root)

    def _build_refcounts(self, root):
        self._refcounts = collections.Counter()
        self._backwards = collections.Counter()

        q = collections.deque([root])

        while q:
            t = q.pop()
            if isinstance(t, Node):
                nodeid = id(t)
                seen = nodeid in self._refcounts
                self._refcounts[nodeid] += 1

                if not seen and not getattr(t, '_no_backward', False):
                    for c in t._args:
                        q.append(c)

    @contextlib.contextmanager
    def unlock_node(self, node):
        if hasattr(node, "setflags") and not node.flags.writeable:
            node.setflags(write=True)
            yield
            node.setflags(write=False)
        else:
            yield

    def store(self, node, dy):
        selfid = id(node)
        self.stroage[selfid] = Node(dy)  # if cuda active, dy must be GPUValue type.

    def restore(self, node, default=None):
        selfid = id(node)
        return self.stroage.get(selfid, default)

    def add(self, node, dy, caller=None):
        selfid = id(node)
        if selfid in self.variables:
            v = self.variables[selfid]
            with self.unlock_node(v):
                if isinstance(dy, GPUValue):
                    diff = v.get_gpu() + dy
                    v.set_gpu(diff)
                else:
                    v[...] += dy
        else:
            if isinstance(dy, GPUValue):
                dy = Variable(dy)
            self.variables[selfid] = dy
            if node._auto_update:
                self._auto_updates.append(node)

        self._backwards[selfid] += 1

        return self._refcounts[selfid] <= self._backwards[selfid]

    _omit = object()

    def get(self, node, default=_omit):
        '''This function returns the gradient with respect to the given node.
        In the case of there are not any gradient of the given node, this function
        returns 'None'.

        Args:
            node (Node): Returns a gradient with respect to this argument.

        Return:
            ndarray, Node, None: Gradient of given node object.
        '''
        if default is self._omit:
            return self.variables[id(node)]
        else:
            return self.variables.get(id(node), default)

    def set(self, node, diff):
        self.variables[id(node)] = diff

    def update_node(self, node, opt=None):
        import time
        if node.prevent_update:
            return

        with self.unlock_node(node):
            dy = self.get(node) if opt is None else opt(self.get(node), node)
            if node._auto_update:
                if callable(node.auto_update):
                    node.auto_update(dy)
                else:
                    if is_cuda_active():
                        ngpu = get_gpu(node)
                        ngpu -= get_gpu(dy)
                    else:
                        node[...] -= dy
            node.detach_graph()

    def update(self, opt=None, models=()):
        '''This function updates variable objects on the computational graph
        using obtained gradients.

        If an optimizer instance is passed, gradients are rescaled
        with regard to the optimization algorithm before updating.

        Args:
            opt (Optimizer): Algorithm for rescaling gradients.
            models: List of models to update variables. When specified,
                    variables which does not belong to one of the models
                    are not updated.
        '''

        if not models:
            for node in self._auto_updates:
                self.update_node(node, opt)
        else:
            for model in models:
                for node in model.params.values():
                    if id(node) in self.variables:
                        self.update_node(node, opt)


def to_value(array):
    if isinstance(array, Node):
        array.to_cpu()
        return array.view(np.ndarray)
    else:
        return array


class GraphAttrs(object):

    def __init__(self):
        object.__setattr__(self, 'v__attrs', {})

    def clear(self):
        self.v__attrs.clear()

    def get_names(self):
        return self.v__attrs.keys()

    def get_attrs(self):
        return self.v__attrs.values()

    def __setattr__(self, name, value):
        self.v__attrs[name] = value

    def __getattr__(self, name):
        try:
            return self.v__attrs[name]
        except KeyError:
            raise AttributeError('%r has no attribute %r' % (self, name))


class Node(np.ndarray):
    '''This is the base class of all operation function.
    Node class inherits numpy ndarray class.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> vx = rm.Variable(np.random.rand(3, 2))
        >>> isinstance(vx, rm.Node)
        True
    '''

    _gpu = None
    attrs = None
    _model = None
    _auto_update = False
    _no_backward = False
    _args = ()

    def __new__(cls, value):
        ret = cls._create_node(value)
        return ret

    @classmethod
    def _create_node(cls, value):
        if isinstance(value, np.ndarray):
            ret = value.astype(precision).view(cls)
        elif isinstance(value, GPUValue):
            ret = super(Node, cls).__new__(
                cls, shape=value.shape, dtype=value.dtype)
            ret._gpu = value

        elif isinstance(value, Number):
            ret = np.array(value, dtype=precision).view(cls)
        else:
            raise ValueError('Invalid Node value: %r' % value)

        assert ret.dtype == precision, (
            "Type miss matched. Required is {}, actual is {}".format(
                precision().dtype, ret.dtype))

        ret.attrs = GraphAttrs()
        if GET_ACTIVE_NODE() is not None:
            SET_NODE_DICT(id(ret), ret)
        return ret

    @classmethod
    def calc_value(cls, *args, **kwargs):
        if is_cuda_active():
            value = cls._oper_gpu(*args, **kwargs)
        else:
            value = cls._oper_cpu(*args, **kwargs)
        return value

    def __init__(self, *args, **kwargs):
        self.setflags(write=False)
        self._args = []
        q = collections.deque([args])
        while q:
            a = q.pop()
            if isinstance(a, Node):
                self._args.append(a)
            elif isinstance(a, list) or isinstance(a, tuple):
                q.extend(a)
            elif isinstance(a, dict):
                q.extend(a.values())
        self._args.extend(a for a in kwargs.values() if isinstance(a, Node))
        self._reduce_graph()
        return

    @property
    def auto_update(self):
        if self._auto_update:
            if self._model:
                if not self._model.auto_update:
                    return False
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        raise Exception()

    @property
    def prevent_update(self):
        if self._model:
            if self._model._prevent_update:
                return True
        return False

    @prevent_update.setter
    def prevent_update(self, value):
        raise Exception()

    @property
    def device_id(self):
        if self._gpu:
            return self._gpu.device_id

        if self._model:
            return self._model._device_id

        return 0

    def set_model(self, model):
        self._model = model

    def get_gpu(self):
        if not self._gpu:
            self._gpu = GPUValue(self)
        return self._gpu

    def set_gpu(self, gpu):
        self.release_gpu()
        self._gpu = gpu

    def to_cpu(self):
        '''Send the data on GPU device to CPU.'''
        if self._gpu:
            self._gpu.to_cpu(self)

    def to_gpu(self):
        '''Send the data on CPU to GPU device.'''
        if self._gpu:
            self._gpu.to_gpu(self)
        else:
            self._gpu = GPUValue(self)

    def copy(self):
        if self._gpu:
            return self.__class__(self._gpu.copy())
        else:
            return np.ndarray.copy(self)

    def copy_from(self, other):
        assert self.shape == other.shape
        assert self.dtype == other.dtype
        if self._gpu:
            if other._gpu:
                self._gpu.copy_from(other._gpu)
                return

        if hasattr(self, "setflags"):
            writable = self.flags.writeable
            self.setflags(write=True)

        try:
            self[...] = other
        finally:
            if hasattr(self, "setflags"):
                self.setflags(write=writable)

    def as_ndarray(self):
        '''This method returns itself as ndarray object.'''
        self.to_cpu()
        if self._gpu:
            return self._gpu.new_array()
        if isinstance(self, Number):
            return np.array(self, dtype=precision)
        else:
            ret = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self)
            ret.setflags(write=True)
            return np.array(ret)

    def release_gpu(self):
        '''This method releases memory on GPU.'''
        if self._gpu:
            self._gpu = None

    def grad(self, initial=None, detach_graph=True, **kwargs):
        '''This method follows computational graph and returns the gradients of
        Variable object.

        Args:
            initial (ndarray): Initial value of following the graph.
            detach_graph (boolean): If it's True, the computational graph will be destroyed.
        '''
        if not self._has_autoupdate():
            return Grads()

        if initial is None:
            if self.size > 1:
                raise ValueError("Initial diff is required for scalar value.")
            initial = np.ones_like(self).astype(precision)
            if is_cuda_active():
                initial = Node(initial)
                initial.to_gpu()

        context = Grads(self)
        self._update_diff(context, initial, **kwargs)

        if detach_graph:
            self.detach_graph()
        return context

    def _update_diff(self, context, dy, **kwargs):
        ready = context.add(self, dy)
        if ready:
            diff = context.get(self)
            self.backward(context, diff, **kwargs)

    def _get_graph(self):
        if self.attrs:
            return self.attrs.get_attrs()
        return []

    def _has_autoupdate(self):
        '''Check if the graph to witch this node belongs need to update.'''

        for v in self._get_graph():
            if isinstance(v, Node):
                if v.auto_update:
                    return True

                if any((o is not None) for o in v._get_graph()):
                    return True

    def _reduce_graph(self):
        if self.attrs:
            if not self._has_autoupdate():
                self._no_backward = True
                self.attrs.clear()
                self._args = []
        return False

    def detach_graph(self):
        '''This method destroys computational graph.'''

        for v in self._get_graph():
            if isinstance(v, Node):
                v.detach_graph()
        if self.attrs:
            self.attrs.clear()

        self._args = []

    def backward(self, context, dy, **kwargs):
        if self._no_backward:
            return

        if is_cuda_active():
            if self._gpu:
                with use_device(self._gpu.device_id):
                    return self._backward_gpu(context, dy, **kwargs)
            else:
                return self._backward_gpu(context, dy, **kwargs)
        else:
            return self._backward_cpu(context, dy, **kwargs)

    def __neg__(self):
        return Neg(self)

    def __pos__(self):
        return Pos(self)

    def __abs__(self):
        return Abs(self)

    def __invert__(self):
        return Invert(self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return RAdd(other, self)

    def __iadd__(self, other):
        assert getattr(self, "shape", (1,)) == getattr(self, "shape", (1,))
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return RSub(other, self)

    def __isub__(self, other):
        assert getattr(self, "shape", (1,)) == getattr(self, "shape", (1,))
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return RMul(other, self)

    def __imul__(self, other):
        assert getattr(self, "shape", (1,)) == getattr(self, "shape", (1,))
        return Mul(self, other)

    def __div__(self, other):
        return Div(self, other)

    def __rdiv__(self, other):
        return RDiv(other, self)

    def __idiv__(self, other):
        assert getattr(self, "shape", (1,)) == getattr(self, "shape", (1,))
        return Div(self, other)

    def __floordiv__(self, other):
        return Div(self, other)

    def __rfloordiv__(self, other):
        return RDiv(other, self)

    def __ifloordiv__(self, other):
        return Div(self, other)

    def __truediv__(self, other):
        return TrueDiv(self, other)

    def __rtruediv__(self, other):
        return RTrueDiv(other, self)

    def __itruediv__(self, other):
        return TrueDiv(self, other)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return RMod(other, self)

    def __imod__(self, other):
        return Mod(self, other)

    def __divmod__(self, other):
        return DivMod(self, other)

    def __rdivmod__(self, other):
        return RDivMod(other, self)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return RPow(other, self)

    def __ipow__(self, other):
        return Pow(self, other)

    def __lshift__(self, other):
        return Lshift(self, other)

    def __rlshift__(self, other):
        return RLshift(other, self)

    def __ilshift__(self, other):
        return Lshift(self, other)

    def __rshift__(self, other):
        return Rshift(self, other)

    def __rrshift__(self, other):
        return RRshift(other, self)

    def __irshift__(self, other):
        return Rshift(self, other)

    def __and__(self, other):
        return And(self, other)

    def __rand__(self, other):
        return RAnd(other, self)

    def __iand__(self, other):
        return And(self, other)

    def __xor__(self, other):
        return Xor(self, other)

    def __rxor__(self, other):
        return RXor(other, self)

    def __ixor__(self, other):
        return Xor(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __ror__(self, other):
        return ROr(other, self)

    def __ior__(self, other):
        return Or(self, other)

    def __getitem__(self, index):
        return GetItem(self, index)

    def __setitem__(self, index, value):
        if self._gpu is not None:
            self._gpu[index] = value
        else:
            np.ndarray.__setitem__(self, index, value)

    def __getslice__(self, i, j):
        return GetSlice(self, i, j)

    def __lt__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__lt__(self, other)

    def __le__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__le__(self, other)

    def __eq__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__eq__(self, other)

    def __ne__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__ne__(self, other)

    def __ge__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__ge__(self, other)

    def __gt__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__gt__(self, other)

    def __not__(self):
        self.to_cpu()
        return np.ndarray.__not__(self)

    def __str__(self):
        self.to_cpu()
        return np.ndarray.__str__(self)

    def __repr__(self):
        self.to_cpu()
        return np.ndarray.__repr__(self)

    def __float__(self):
        self.to_cpu()
        return np.ndarray.__float__(self)

    def __int__(self):
        self.to_cpu()
        return np.ndarray.__int__(self)

    def __complex__(self):
        self.to_cpu()
        return np.ndarray.__complex__(self)

    def __bool__(self):
        self.to_cpu()
        return np.ndarray.__bool__(self)

    def __index__(self):
        self.to_cpu()
        return np.ndarray.__index__(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # move gpu values of input arrays to cpu
        new_inputs = []
        for item in inputs:
            if isinstance(item, Node):
                item.to_cpu()
                item.release_gpu()
                new_inputs.append(item.view(np.ndarray))
            else:
                new_inputs.append(item)

        # move gpu values of output arrays to cpu
        outs = kwargs.get('out', None)
        if isinstance(outs, tuple):
            new_outs = []
            for item in outs:
                if isinstance(item, Node):
                    item.to_cpu()
                    item.release_gpu()
                    new_outs.append(item.view(np.ndarray))
                else:
                    new_outs.append(item)

            kwargs['out'] = tuple(new_outs)

        elif outs is not None:
            kwargs['out'] = outs.view(np.ndarray)
            outs.to_cpu()
            outs.release_gpu()

        ret = getattr(ufunc, method)(*new_inputs, **kwargs)
        return ret

    @property
    def T(self):
        return Transpose2d(self)

    def transpose(self, *axis):
        ax = axis
        if isinstance(ax[0], (tuple, list)):
            ax = ax[0]
        else:
            ax = tuple(axis)

        assert len(self.shape) == len(ax), "Axis must be same size to matrix dim size."
        return Transpose(self, ax)

    def reshape(self, *shape):
        if isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return Reshape(self, shape)


class Variable(Node):
    '''Variable class.

    The gradient of this object will be calculated.
    Variable object is created from ndarray object or Number object.

    Args:
        value (Variable,ndarray): Input array.
        auto_update (bool): Auto update flag.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.array([1. -1])
        >>> rm.Variable(x)
        Variable([ 1., -1.], dtype=float32)
    '''

    def __new__(cls, value, auto_update=True):
        ret = super(Variable, cls).__new__(cls, value)
        ret._auto_update = auto_update
        return ret

    def backward(self, context, dy, **kwargs):
        pass


class UnaryOp(Node):
    def __new__(cls, arg, *args, **kwargs):
        value = cls.calc_value(arg, *args, **kwargs)
        ret = super(UnaryOp, cls).__new__(cls, value)
        ret.attrs._arg = arg
        return ret


class Transpose(UnaryOp):
    @classmethod
    def _oper_cpu(cls, arg):
        assert(len(arg.shape) < 3)
        return arg.T

    @classmethod
    def _oper_gpu(cls, arg):
        return get_gpu(arg).T

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy.T, **kwargs)

    def _backward_gpu(self, context, dy, *kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context. get_gpu(dy).T, **kwargs)


class Pos(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__pos__(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        return +get_gpu(arg.get_gpu())

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, get_gpu(dy), **kwargs)


class Neg(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__neg__(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        return -(get_gpu(arg))

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, -dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, -get_gpu(dy), **kwargs)


class Abs(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__abs__(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        new_ptr = get_gpu(arg).empty_like_me()
        cuabs_forward(get_gpu(arg), new_ptr)
        return new_ptr

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            arg = to_value(self.attrs._arg)
            mask = np.where(arg > 0, 1, -1)
            self.attrs._arg._update_diff(context, mask * dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            new_ptr = get_gpu(dy).empty_like_me()
            cuabs_backward(get_gpu(self.attrs._arg), new_ptr)
            self.attrs._arg._update_diff(context, new_ptr, **kwargs)


class Invert(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__invert__(arg)

    def _backward_cpu(self, context, dy, **kwargs):
        self.attrs._arg._update_diff(context, dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        return self.attrs._backward_cpu(context, dy, **kwargs)


class BinOp(Node):
    GRAPH = ['_lhs', '_rhs']

    def __new__(cls, lhs, rhs, *args, **kwargs):
        value = cls.calc_value(lhs, rhs, *args, **kwargs)
        ret = super(BinOp, cls).__new__(cls, value)
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        return ret


def broad_cast(hs, dy):
    if isinstance(hs, np.ndarray):
        shape = list(hs.shape)
        if hs.shape != dy.shape:
            axis = []
            while len(shape) != len(dy.shape):
                if len(shape) < len(dy.shape):
                    shape.insert(0, 1)
            for i, s in enumerate(shape):
                if s == 1:
                    axis.append(i)
            if axis:
                dy = np.sum(dy, axis=tuple(axis))
        dy = dy.reshape(hs.shape)
    return dy


def cu_broad_cast(hs, dy):
    if isinstance(hs, GPUValue):
        shape = list(hs.shape)
        if hs.shape != dy.shape:
            axis = []
            while len(shape) != len(dy.shape):
                if len(shape) < len(dy.shape):
                    shape.insert(0, 1)
            for i, s in enumerate(shape):
                if s == 1:
                    axis.append(i)
            if axis:
                dy = cusum(dy, axis=tuple(axis))
            dy = dy.reshape(hs.shape)
    return dy


class Add(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__add__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) + get_gpu(rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._rhs, Node):
            r_dx = broad_cast(self.attrs._rhs, dy)
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(self.attrs._lhs, dy)
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._rhs, Node):
            rhs = get_gpu(self.attrs._rhs)

            r_dx = cu_broad_cast(rhs, get_gpu(dy))
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            l_dx = cu_broad_cast(lhs, get_gpu(dy))
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)


class RAdd(Add):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__radd__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(rhs) + get_gpu(lhs)


class Sub(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__sub__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) - get_gpu(rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(self.attrs._lhs, dy)
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            r_dx = broad_cast(self.attrs._rhs, -dy)
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)

            new_l_dx = cu_broad_cast(lhs, get_gpu(dy))
            self.attrs._lhs._update_diff(context, new_l_dx, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)

            new_r_dx = cu_broad_cast(rhs, -1 * get_gpu(dy))

            self.attrs._rhs._update_diff(context, new_r_dx, **kwargs)


class RSub(Sub):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rsub__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return - get_gpu(rhs) + get_gpu(lhs)


class Mul(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__mul__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) * get_gpu(rhs)

    def _backward_cpu(self, context, dy, **kwargs):

        lhs = to_value(self.attrs._lhs)
        rhs = to_value(self.attrs._rhs)

        if isinstance(self.attrs._rhs, Node):
            r_dx = broad_cast(rhs, lhs * dy)
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(lhs, rhs * dy)
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._rhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)
            dxr = cu_broad_cast(rhs, lhs * get_gpu(dy))

            self.attrs._rhs._update_diff(context, dxr, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)
            dxl = cu_broad_cast(lhs, rhs * get_gpu(dy))

            self.attrs._lhs._update_diff(context, dxl, **kwargs)


class RMul(Mul):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rmul__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(rhs) * get_gpu(lhs)


class Div(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__div__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) / get_gpu(rhs)

    def _backward_cpu(self, context, dy, **kwargs):

        lhs = to_value(self.attrs._lhs)
        rhs = to_value(self.attrs._rhs)

        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(lhs, dy / rhs)
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            n = (-1) * (rhs ** (-2))
            r_dx = broad_cast(rhs, lhs * n * dy)
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):

        if isinstance(self.attrs._lhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)

            dxl = cu_broad_cast(lhs, get_gpu(dy) / rhs)
            self.attrs._lhs._update_diff(context, dxl, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)

            v = rhs ** (-2.0) * -1.0 * lhs * get_gpu(dy)
            dxr = cu_broad_cast(rhs, v)
            self.attrs._rhs._update_diff(context, dxr, **kwargs)


class RDiv(Div):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rdiv__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) / get_gpu(rhs)


class TrueDiv(Div):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        ret = np.ndarray.__truediv__(lhs, rhs)
        return ret

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) / get_gpu(rhs)


class RTrueDiv(TrueDiv):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rtruediv__(rhs, lhs)


class Mod(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__mod__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RMod(Mod):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rmod__(rhs, lhs)


class DivMod(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        d, m = np.ndarray.__divmod__(lhs, rhs)
        return np.array([d, m])

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RDivMod(DivMod):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        d, m = np.ndarray.__rdivmod__(rhs, lhs)
        return np.array([d, m])


class Pow(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__pow__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) ** get_gpu(rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        lhs = to_value(self.attrs._lhs)
        rhs = to_value(self.attrs._rhs)

        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, dy * (np.power(lhs, rhs - 1) * rhs), **kwargs)

        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, dy * self * np.log(lhs), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):

        if isinstance(self.attrs._lhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)

            v = get_gpu(dy) * rhs * (GPUValue.__pow__(lhs, rhs - 1))

            dxl = cu_broad_cast(lhs, v)
            self.attrs._lhs._update_diff(context, dxl, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            lhs = get_gpu(self.attrs._lhs).empty_like_me()
            culoge(get_gpu(self.attrs._lhs), lhs)
            new_r_dx = get_gpu(dy) * get_gpu(self) * lhs
            self.attrs._rhs._update_diff(context, new_r_dx, **kwargs)


class RPow(Pow):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rpow__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) ** get_gpu(rhs)


class Lshift(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__lshift__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RLshift(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rlshift__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)


class Rshift(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rshift__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RRshift(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rrshift__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)


class And(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__and__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RAnd(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rand__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class Xor(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__xor__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RXor(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rxor__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)


class Or(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__or__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class ROr(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__ror__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)


class GetItem(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__getitem__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs)[rhs]

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            zero = np.zeros_like(to_value(self.attrs._lhs))
            np.add.at(zero, self.attrs._rhs, to_value(dy))
            self.attrs._lhs._update_diff(context, zero, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            if self._is_advanced_indexing(self.attrs._lhs, self.attrs._rhs):
                self._backward_cpu(context, to_value(dy), **kwargs)
            else:
                zero = get_gpu(self.attrs._lhs).zeros_like_me()
                zero[self.attrs._rhs] = dy
                self.attrs._lhs._update_diff(context, zero, **kwargs)

    def _is_advanced_indexing(self, array, index):
        if isinstance(index, (int, slice, type(None), type(Ellipsis))):
            return False
        elif isinstance(index, tuple):
            if all([isinstance(o, (int, slice, type(None), type(Ellipsis))) for o in index]):
                return False
        elif isinstance(index, np.ndarray):
            if index.dtype == np.bool:
                return False
        return True


class GetFgAry(Node):
    @classmethod
    def _oper_cpu(cls, arg):
        return arg[:, :, 1, :, :]

    @classmethod
    def _oper_gpu(cls, arg):
        shape = arg.shape
        fg_ary = GPUValue(shape=(shape[0], shape[1], 1, shape[3], shape[4]))
        arg = get_gpu(arg)
        cu_get_fg_ary_forward(arg, fg_ary)
        return fg_ary

    def __new__(cls, arg):
        value = cls.calc_value(arg)
        ret = super(GetFgAry, cls).__new__(cls, value)
        ret.attrs._arg = arg
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            zero = np.zeros_like(np.array(self.attrs._arg))
            zero[:, :, 1, :, :] = np.array(dy)
            self.attrs._arg._update_diff(context, zero, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            zero = get_gpu(self.attrs._lhs).zeros_like_me()
            cu_get_fg_ary_backward(dy, zero)
            self.attrs._arg._update_diff(context, zero, **kwargs)


class GetIthAry(Node):
    @classmethod
    def _oper_cpu(cls, arg, i):
        return arg[i]

    @classmethod
    def _oper_gpu(cls, arg, i):
        shape = arg.shape
        ith_ary = GPUValue(shape=(shape[1:]))
        arg = get_gpu(arg)
        cu_get_ith_ary_forward(arg, ith_ary, i)
        return ith_ary

    def __new__(cls, arg, i):
        value = cls.calc_value(arg, i)
        ret = super(GetIthAry, cls).__new__(cls, value)
        ret.attrs._arg = arg
        ret._index = i
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            zero = np.zeros_like(np.array(self.attrs._arg))
            zero[self.attrs._index] = np.array(dy)
            self.attrs._arg._update_diff(context, zero, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            zero = get_gpu(self.attrs._lhs).zeros_like_me()
            cu_get_ith_ary_backward(dy, zero, self.attrs._index)
            self.attrs._arg._update_diff(context, zero, **kwargs)


class GetNthAry(Node):
    def __new__(cls, arg, i, j):
        value = cls.calc_value(arg, i, j)
        ret = super(GetNthAry, cls).__new__(cls, value)
        return ret

    @classmethod
    def _oper_gpu(cls, arg, i, j):
        ary = GPUValue(shape=(arg.shape[0], ((arg.shape[1] - (i + 1)) // j) + 1))
        arg = get_gpu(arg)
        cu_get_every_nth_ary(arg, ary, i, j)
        return ary


class GetSlice(Node):
    @classmethod
    def _oper_cpu(cls, arg, i, j):
        return np.ndarray.__getslice__(arg, i, j)

    @classmethod
    def _oper_gpu(cls, arg, i, j):
        return cls._oper_cpu(arg, i, j)

    def __new__(cls, arg, i, j):
        value = cls.calc_value(arg, i, j)
        ret = super(GetSlice, cls).__new__(cls, value)
        ret.attrs._arg = arg
        ret.attrs._i, ret.attrs._j = i, j
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            zero = np.zeros_like(np.array(self.attrs._arg))
            zero[self.attrs._i:self.attrs._j] = np.array(dy)
            self.attrs._arg._update_diff(context, zero, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class AssignPredBox(Node):
    def __new__(cls, arg, x, y, h, w):
        ary = GPUValue(shape=arg.shape)
        x = get_gpu(x)
        y = get_gpu(y)
        h = get_gpu(h)
        w = get_gpu(w)
        value = cls.calc_value(ary, x, y, h, w)
        ret = super(AssignPredBox, cls).__new__(cls, value)
        return ret

    @classmethod
    def _oper_gpu(cls, ary, x, y, h, w):
        cu_assign_pred_box(x, y, h, w, ary)
        return ary


class PredCtr(Node):
    def __new__(cls, arg, length, ctr):
        ary = GPUValue(shape=arg.shape)
        arg = get_gpu(arg)
        length = get_gpu(length)
        ctr = get_gpu(ctr)
        value = cls.calc_value(arg, length, ctr, ary)
        ret = super(PredCtr, cls).__new__(cls, value)
        return ret

    @classmethod
    def _oper_gpu(cls, arg, length, ctr, ary):
        cu_pred_ctr(arg, length, ctr, ary)
        return ary


class GetIthBbox(Node):
    def __new__(cls, arg, i):
        arg = get_gpu(arg)
        ary = GPUValue(shape=(arg.shape[0], 1))
        value = cls.calc_value(arg, i, ary)
        ret = super(GetIthBbox, cls).__new__(cls, value)
        return ret

    @classmethod
    def _oper_gpu(cls, arg, i, ary):
        cu_get_ith_bbox(arg, i, ary)
        return ary


class Reshape(Node):

    @classmethod
    def _oper_cpu(cls, array, shape):
        return np.reshape(array, shape).copy()

    @classmethod
    def _oper_gpu(cls, array, shape):
        return get_gpu(array).reshape(shape)

    def __new__(cls, array, shape):
        value = cls.calc_value(array, shape)
        ret = super(Reshape, cls).__new__(cls, value)
        ret.attrs._array = array
        ret.attrs._shape = array.shape
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._array, Node):
            self.attrs._array._update_diff(context, to_value(
                dy).reshape(self.attrs._shape), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._array, Node):
            self.attrs._array._update_diff(context, get_gpu(
                dy).reshape(self.attrs._shape), **kwargs)


class Transpose2d(UnaryOp):
    @classmethod
    def _oper_cpu(cls, arg):
        assert(len(arg.shape) < 3)
        return to_value(arg).T

    @classmethod
    def _oper_gpu(cls, arg):
        return get_gpu(arg).T

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, to_value(dy).T, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, get_gpu(dy).T, **kwargs)


class Transpose(Node):

    @classmethod
    def _oper_cpu(cls, arg, axis):
        return np.transpose(to_value(arg), axis)

    @classmethod
    def _oper_gpu(cls, arg, axis):
        return get_gpu(arg).transpose(axis)

    def __new__(cls, arg, axis):
        value = cls.calc_value(arg, axis)
        ret = super(Transpose, cls).__new__(cls, value)
        rev = [-1] * len(axis)
        for i, a in enumerate(axis):
            rev[a] = i
        ret.attrs._arg = arg
        ret.attrs._axis = rev
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            axis = self.attrs._axis
            self.attrs._arg._update_diff(context, to_value(dy).transpose(axis), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            axis = self.attrs._axis
            self.attrs._arg._update_diff(context, get_gpu(dy).transpose(axis), **kwargs)


class Abase(Node):

    def __new__(cls, arg, axis=None, keepdims=False):
        assert isinstance(axis, (type(None), int)), 'This function only accepts int or None.'
        value, index = cls.calc_value(arg, axis, keepdims)
        ret = super(Abase, cls).__new__(cls, value)
        ret.attrs._arg = arg
        ret.attrs._axis = axis
        ret.attrs._index = index
        ret.attrs._keepdims = keepdims
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            axis = self.attrs._axis
            index = self.attrs._index
            dx = np.zeros(self.attrs._arg.shape, dtype=dy.dtype)

            if axis is None:
                dxx = dx.reshape(-1)
                dxx[index] = dy
            else:
                axis_list = list(range(len(dx.shape)))
                axis_list.pop(axis)
                axis_list.append(axis)
                rev = [-1] * len(axis_list)
                for i, a in enumerate(axis_list):
                    rev[a] = i
                dxx = np.transpose(dx, axis_list)
                if(not self.attrs._keepdims):
                    dyy = dy
                else:
                    axis_list = list(range(len(dy.shape)))
                    axis_list.pop(axis)
                    axis_list.append(axis)
                    rev = [-1] * len(axis_list)
                    for i, a in enumerate(axis_list):
                        rev[a] = i
                    dyy = np.transpose(dy, axis_list)
                for i in np.ndindex(index.shape):
                    dxx[i][index[i]] = dyy[i]

            # dxx is a representation of the same memory as dx

            self.attrs._arg._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            axis = self.attrs._axis
            index = self.attrs._index.new_array()
            dx = np.zeros(self.attrs._arg.shape, dy.dtype)

            if axis is None:
                dxx = dx.reshape(-1)
                dxx[index] = dy
            else:
                axis_list = list(range(len(dx.shape)))
                axis_list.pop(axis)
                axis_list.append(axis)
                rev = [-1] * len(axis_list)
                for i, a in enumerate(axis_list):
                    rev[a] = i
                dxx = np.transpose(dx, axis_list)
                if(not self.attrs._keepdims):
                    dyy = dy
                else:
                    dyy = np.transpose(dy, axis_list)
                for i in np.ndindex(index.shape):
                    dxx[i][index[i]] = dyy[i]
            self.attrs._arg._update_diff(context, get_gpu(dx), **kwargs)


class Amax(Abase):
    """This function performs max calculation.

    Args:
        arg (Variable, ndarray): Input matrix.
        axis (int): Perform calculation along this argument.
        keepdims (bool): If `True` is passed, reduced dimensions remain.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> # Forward Calculation
        >>> a = np.arange(4).reshape(2, 2)
        >>> a
        [[0 1]
         [2 3]]
        >>> rm.amax(a, axis=1)
        [ 1.  3.]
        >>>
        >>> rm.amax(a, axis=0)
        [ 2.  3.]
        >>> rm.amax(a, axis=0, keepdims=True)
        [[ 2.  3.]]
        >>>
        >>> # Calculation of differentiation
        >>> va = rm.Variable(a)
        >>> out = rm.amax(va)
        >>> grad = out.grad()
        >>> grad.get(va) # Getting the gradient of 'va'.
        [[ 0.,  0.],
         [ 0.,  1.]]
    """

    @classmethod
    def _oper_cpu(cls, arg, axis, keepdims):
        array = to_value(arg)
        # Max is calculated twice, update?
        return np.amax(array, axis, keepdims=keepdims), np.argmax(array, axis)

    @classmethod
    def _oper_gpu(cls, arg, axis, keepdims):
        array = get_gpu(arg)
        value = cu_reduce_max(array, axis, keepdims)
        index = cu_reduce_argmax(array, axis)
        return value, index


class Amin(Abase):
    """This function performs min calculation.

    Args:
        arg (Variable, ndarray): Input matrix.
        axis (int): Perform calculation along this argument.
        keepdims (bool): If `Ture` is passed, reduced dimentions remain.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> # Forward Calculation
        >>> a = np.arange(4).reshape(2, 2)
        >>> a
        [[0 1]
         [2 3]]
        >>> rm.amin(a, axis=1)
        [ 0.  2.]
        >>>
        >>> rm.amin(a, axis=0)
        [ 0.  1.]
        >>> rm.amin(a, axis=0, keepdims=True)
        [[ 0.  1.]]
        >>>
        >>> # Calculation of differentiation
        >>> va = rm.Variable(a)
        >>> out = rm.amin(va)
        >>> grad = out.grad()
        >>> grad.get(va) # Getting the gradient of 'va'.
        [[ 1.,  0.],
         [ 0.,  0.]]
    """

    @classmethod
    def _oper_cpu(cls, arg, axis, keepdims):
        array = to_value(arg)
        return np.amin(array, axis, keepdims=keepdims), np.argmin(array, axis)

    @classmethod
    def _oper_gpu(cls, arg, axis, keepdims):
        array = get_gpu(arg)
        value = cu_reduce_min(array, axis, keepdims)
        index = cu_reduce_argmin(array, axis)
        return value, index

# -*- coding: utf-8 -*-
from __future__ import print_function, division
import collections
import weakref
from . config import precision
import contextlib
import numpy as np
import itertools
from numbers import Number

from renom.cuda import *


class Grads:
    '''This class contains gradients of each Node object.

    When the function ``grad`` which is instance of Node class called,
    an instance of Grads is returned.

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

    def __init__(self):
        self.stroage = {}
        self.variables = {}
        self._auto_updates = []

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

    def add(self, node, dy):
        selfid = id(node)
        if selfid in self.variables:
            node = self.variables[selfid]
            with self.unlock_node(node):
                if isinstance(dy, GPUValue):
                    diff = node.get_gpu() + dy
                    node.set_gpu(diff)
                else:
                    node[...] += dy
        else:
            if isinstance(dy, GPUValue):
                dy = Variable(dy)
            self.variables[selfid] = dy
            if node._auto_update:
                self._auto_updates.append(node)

    _omit = object()

    def get(self, node, default=_omit):
        '''This function returns the gradient of the given node.

        Args:
            node (Node):

        Return:
            ndarray,Node,None: This method returns gradient of passed node object.
        '''
        if default is self._omit:
            return self.variables[id(node)]
        else:
            return self.variables.get(id(node), default)

    def set(self, node, diff):
        self.variables[id(node)] = diff

    def update_node(self, node, opt=None):
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
        '''Updates variables using earned gradients.

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

    def join(self, model, others):
        """Merge gradients of other models.
        Merged models should have same structure with model."""

        values = dict(model.flatten_values())
        for o in others:
            for (name, attrname), diff in o.items():
                obj = values[name][attrname]
                curdiff = self.get(obj, None)
                if curdiff is not None:
                    if isinstance(curdiff, GPUValue):
                        with use_device(curdiff.device_id):
                            if diff.device_id != curdiff.device_id:
                                diff = diff.copy()
                            newdiff = curdiff + diff
                    else:
                        newdiff = curdiff + diff

                    self.set(obj, newdiff)


# todo: move this method to Cython
def calc_broadcast_shape(s1, s2):
    # silly, but works
    if all([isinstance(s, (np.ndarray, Number, GPUValue)) for s in (s1, s2)]):
        return np.broadcast(np.empty(getattr(s1, "shape", 1), dtype=np.bool),
                            np.empty(getattr(s2, "shape", 1), dtype=np.bool)).shape
    else:
        raise Exception("Not supported data type.")


class GPUValue(object):
    ACTIVE_GPU = None

    def __init__(self, array=None, shape=None, ptr=None):
        if shape is not None:
            self.shape = shape
        else:
            self.shape = getattr(array, "shape", None) or ()

        self.dtype = precision
        self.itemsize = np.dtype(self.dtype).itemsize
        self.nbytes = np.prod(self.shape) * self.itemsize
        self.size = np.prod(self.shape) if self.shape else 1
        self._ptr = ptr
        if array is not None:
            self.to_gpu(array)
        elif not self._ptr:
            self.alloc()
        else:
            self.device_id = cuGetDevice()

        if self.ACTIVE_GPU is not None:
            self.ACTIVE_GPU[id(self)] = self

    def __del__(self):
        self.free()

    def alloc(self):
        if self._ptr:
            gpu_allocator.free(self._ptr)
        self._ptr = gpu_allocator.malloc(self.nbytes)
        self.device_id = cuGetDevice()

    def free(self):
        if self._ptr:
            gpu_allocator.free(self._ptr)
        self._ptr = None

    def reshape(self, *shape):
        clone = self.copy()
        a = np.empty(self.shape).reshape(*shape)
        clone.shape = a.shape
        return clone

    def attach(self, value):
        ptr = value._ptr
        self.detach()
        self._ptr = ptr
        self.device_id = value.device_id

    def detach(self):
        if self._ptr:
            ret = GPUValue(shape=self.shape, ptr=self._ptr)
            self._ptr = None
            return ret

    def get_gpu(self):
        return self

    def copy(self):
        if cuGetDevice() == self.device_id:
            ret = GPUValue(shape=self.shape)
            self._ptr.memcpyD2D(ret._ptr, self.nbytes)
        else:
            with use_device(self.device_id):
                arr = self.new_array()
            ret = GPUValue(arr)
        return ret

    def empty_like_me(self):
        ret = GPUValue(shape=self.shape)
        return ret

    def zeros_like_me(self):
        ret = self.empty_like_me()
        cufill(0., ret)
        return ret

    def ones_like_me(self):
        ret = self.empty_like_me()
        cufill(1., ret)
        return ret

    def new_array(self):
        em = np.empty(self.shape, dtype=self.dtype)
        self._ptr.memcpyD2H(em, em.nbytes)
        return em

    def to_cpu(self, value):
        assert self._ptr
        assert tuple(value.shape) == tuple(self.shape), "{} {}".format(value.shape, self.shape)
        assert value.dtype == self.dtype
        self._ptr.memcpyD2H(value, value.nbytes)
        return value

    def to_gpu(self, value):
        if value.dtype.type is not self.dtype:
            value = value.astype(self.dtype)

        assert value.shape == self.shape, "{} {}".format(value.shape, self.shape)

        if self._ptr:
            ptr = self._ptr
        else:
            ptr = gpu_allocator.malloc(value.nbytes)
            self.device_id = cuGetDevice()

        # todo: value.flatten() copies buffer
        with use_device(self.device_id):
            ptr.memcpyH2D(value.flatten(), value.nbytes)

        self._ptr = ptr

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        ret = self.copy()
        cumul(self, -1, ret)
        return ret

    def __add__(self, other):
        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            # Only data type float32 is acceptable.
            cuadd(self, other, ret)
            return ret

    def __iadd__(self, other):
        with use_device(self.device_id):
            assert getattr(self, "shape", (1,)) == getattr(self, "shape", (1,))
            cublas_axpy(get_gpu(other), get_gpu(self))
            return self

    def __mul__(self, other):
        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            cumul(self, other, ret)
            return ret

    def __rmul__(self, other):
        with use_device(self.device_id):
            return self.__mul__(other)

    def __div__(self, other):
        with use_device(self.device_id):
            return self.__truediv__(other)

    def __rdiv__(self, other):
        with use_device(self.device_id):
            return self.__rtruediv__(other)

    def __idiv__(self, other):
        with use_device(self.device_id):
            return self.__itruediv__(other)

    def __truediv__(self, other):
        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            cudiv(self, other, ret)
            return ret

    def __rtruediv__(self, other):
        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            curdiv(self, other, ret)
            return ret

    def __itruediv__(self, other):
        with use_device(self.device_id):
            assert getattr(self, "shape", (1,)) == getattr(self, "shape", (1,))
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            cudiv(self, other, ret)
            return ret

    def __sub__(self, other):
        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            cusub(self, other, ret)
            return ret

    def __isub__(self, other):
        with use_device(self.device_id):
            assert getattr(self, "shape", (1,)) == getattr(self, "shape", (1,))
            cublas_axpy(-get_gpu(other), get_gpu(self))
            return self

    def __pow__(self, other):
        with use_device(self.device_id):
            s_shape = getattr(self, "shape", 1)
            o_shape = getattr(other, "shape", 1)
            new_shape = (np.empty(s_shape, dtype=np.bool) ** np.zeros(o_shape)).shape
            ret = GPUValue(shape=new_shape)
            cupow(self, other, ret)
            return ret

    def __rpow__(self, other):
        with use_device(self.device_id):
            s_shape = getattr(self, "shape", 1)
            o_shape = getattr(other, "shape", 1)
            new_shape = (np.empty(s_shape, dtype=np.bool) ** np.zeros(o_shape)).shape
            ret = GPUValue(shape=new_shape)
            curpow(self, other, ret)
            return ret

    def __getitem__(self, index):
        with use_device(self.device_id):
            arry = self.new_array()
            arry = arry[index]
            ret = GPUValue(arry)
            return ret

    def __setitem__(self, index, value):
        with use_device(self.device_id):
            arry = self.new_array()
            arry[index] = get_gpu(value).new_array()
            self.free()
            self.to_gpu(arry)

    @property
    def T(self):
        with use_device(self.device_id):
            n = len(self.shape)
            assert n < 3
            clone = self.zeros_like_me()
            if n == 2:
                new_shape = list(clone.shape)
                with cublas_handler() as cublas_handle:
                    cublas_transpose(cublas_handle, self, clone)
                new_shape[0] = clone.shape[1]
                new_shape[1] = clone.shape[0]
                clone.shape = tuple(new_shape)
            return clone


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
    ACTIVE_NODE = None
    _model = None
    _auto_update = False
    _no_backward = False

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
        if cls.ACTIVE_NODE is not None:
            cls.ACTIVE_NODE[id(ret)] = ret
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
            self._gpu.free()
            self._gpu = None

    def grad(self, initial=None, detach_graph=True):
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

        context = Grads()
        self._update_diff(context, initial)

        if detach_graph:
            self.detach_graph()
        return context

    def _update_diff(self, context, dy):
        context.add(self, dy)
        self.backward(context, dy)

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

        return False

    def detach_graph(self):
        '''This method destroys computational graph.'''

        for v in self._get_graph():
            if isinstance(v, Node):
                v.detach_graph()
        if self.attrs:
            self.attrs.clear()

    def backward(self, context, dy):
        if self._no_backward:
            return

        if is_cuda_active():
            if self._gpu:
                with use_device(self._gpu.device_id):
                    return self._backward_gpu(context, dy)
            else:
                return self._backward_gpu(context, dy)
        else:
            return self._backward_cpu(context, dy)

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
        if is_cuda_active():
            if self._gpu is None:
                ret = Variable(super(Variable, self).T)
                ret.get_gpu()
            else:
                ret = Variable(get_gpu(self).T)
        else:
            ret = Variable(super(Node, self).T)
        return ret


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

    def backward(self, context, dy):
        pass


class UnaryOp(Node):
    def __new__(cls, arg, *args, **kwargs):
        value = cls.calc_value(arg, *args, **kwargs)
        ret = super(UnaryOp, cls).__new__(cls, value)
        ret.attrs._arg = arg
        return ret


class Pos(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__pos__(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        return +get_gpu(arg.get_gpu())

    def _backward_cpu(self, context, dy):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy)

    def _backward_gpu(self, context, dy):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, get_gpu(dy))


class Neg(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__neg__(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        return -(get_gpu(arg))

    def _backward_cpu(self, context, dy):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, -dy)

    def _backward_gpu(self, context, dy):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, -get_gpu(dy))


class Abs(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__abs__(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        new_ptr = get_gpu(arg).empty_like_me()
        cuabs_forward(get_gpu(arg), new_ptr)
        return new_ptr

    def _backward_cpu(self, context, dy):
        if isinstance(self.attrs._arg, Node):
            arg = to_value(self.attrs._arg)
            # TODO: 原点における劣微分の定義
            mask = np.where(arg > 0, 1, -1)
            self.attrs._arg._update_diff(context, mask * dy)

    def _backward_gpu(self, context, dy):
        if isinstance(self.attrs._arg, Node):
            new_ptr = get_gpu(dy).empty_like_me()
            cuabs_backward(get_gpu(self.attrs._arg), new_ptr)
            self.attrs._arg._update_diff(context, new_ptr)


class Invert(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__invert__(arg)

    def _backward_cpu(self, context, dy):
        self.attrs._arg._update_diff(context, dy)

    def _backward_gpu(self, context, dy):
        return self.attrs._backward_cpu(context, dy)


class BinOp(Node):
    GRAPH = ['_lhs', '_rhs']

    def __new__(cls, lhs, rhs, *args, **kwargs):
        value = cls.calc_value(lhs, rhs, *args, **kwargs)
        ret = super(BinOp, cls).__new__(cls, value)
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        return ret


# TODO:cython化
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


def get_gpu(array):
    if isinstance(array, Number) or isinstance(array, GPUValue):
        return array
    elif isinstance(array, Node):
        return array.get_gpu()
    elif isinstance(array, np.ndarray):
        return GPUValue(array=array)
    else:
        raise Exception("Gpu not supported data type.")


class Add(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__add__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        # TODO:scal vs scalの定義
        return get_gpu(lhs) + get_gpu(rhs)

    def _backward_cpu(self, context, dy):
        # 次元が異なる場合の足し算を定義

        if isinstance(self.attrs._rhs, Node):
            r_dx = broad_cast(self.attrs._rhs, dy)
            self.attrs._rhs._update_diff(context, r_dx)

        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(self.attrs._lhs, dy)
            self.attrs._lhs._update_diff(context, l_dx)

    def _backward_gpu(self, context, dy):
        gdy = get_gpu(dy)

        if isinstance(self.attrs._rhs, Node):
            grhs = get_gpu(self.attrs._rhs)
            if grhs.shape == gdy.shape:
                new_r_dx = gdy.copy()
            else:
                new_r_dx = grhs.zeros_like_me()
                cubroadcast(gdy, new_r_dx)
            self.attrs._rhs._update_diff(context, new_r_dx)

        if isinstance(self.attrs._lhs, Node):
            glhs = get_gpu(self.attrs._lhs)
            if glhs.shape == gdy.shape:
                new_l_dx = gdy.copy()
            else:
                new_l_dx = glhs.zeros_like_me()
                cubroadcast(gdy, new_l_dx)
            self.attrs._lhs._update_diff(context, new_l_dx)


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

    def _backward_cpu(self, context, dy):
        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(self.attrs._lhs, dy)
            self.attrs._lhs._update_diff(context, l_dx)

        if isinstance(self.attrs._rhs, Node):
            r_dx = broad_cast(self.attrs._rhs, -dy)
            self.attrs._rhs._update_diff(context, r_dx)

    def _backward_gpu(self, context, dy):
        if isinstance(self.attrs._lhs, Node):
            new_l_dx = get_gpu(self.attrs._lhs).zeros_like_me()
            dxl = get_gpu(dy)
            cubroadcast(dxl, new_l_dx)
            self.attrs._lhs._update_diff(context, new_l_dx)

        if isinstance(self.attrs._rhs, Node):
            new_r_dx = get_gpu(self.attrs._rhs).zeros_like_me()
            dxr = -get_gpu(dy)
            cubroadcast(dxr, new_r_dx)
            self.attrs._rhs._update_diff(context, new_r_dx)


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

    def _backward_cpu(self, context, dy):
        lhs = to_value(self.attrs._lhs)
        rhs = to_value(self.attrs._rhs)

        if isinstance(self.attrs._rhs, Node):
            r_dx = broad_cast(rhs, lhs * dy)
            self.attrs._rhs._update_diff(context, r_dx)

        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(lhs, rhs * dy)
            self.attrs._lhs._update_diff(context, l_dx)

    def _backward_gpu(self, context, dy):

        if isinstance(self.attrs._rhs, Node):
            new_r_dx = get_gpu(self.attrs._rhs).zeros_like_me()
            dxr = get_gpu(dy) * get_gpu(self.attrs._lhs)
            cubroadcast(dxr, new_r_dx)

            self.attrs._rhs._update_diff(context, new_r_dx)

        if isinstance(self.attrs._lhs, Node):
            new_l_dx = get_gpu(self.attrs._lhs).zeros_like_me()
            dxl = get_gpu(dy) * get_gpu(self.attrs._rhs)
            cubroadcast(dxl, new_l_dx)
            self.attrs._lhs._update_diff(context, new_l_dx)


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

    def _backward_cpu(self, context, dy):

        lhs = to_value(self.attrs._lhs)
        rhs = to_value(self.attrs._rhs)

        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(lhs, dy / rhs)
            self.attrs._lhs._update_diff(context, l_dx)

        if isinstance(self.attrs._rhs, Node):
            n = (-1) * (rhs ** (-2))
            r_dx = broad_cast(rhs, lhs * n * dy)
            self.attrs._rhs._update_diff(context, r_dx)

    def _backward_gpu(self, context, dy):

        if isinstance(self.attrs._lhs, Node):
            new_l_dx = get_gpu(self.attrs._lhs).zeros_like_me()
            dxl = get_gpu(dy) / get_gpu(self.attrs._rhs)
            cubroadcast(dxl, new_l_dx)
            self.attrs._lhs._update_diff(context, new_l_dx)

        if isinstance(self.attrs._rhs, Node):
            new_r_dx = get_gpu(self.attrs._rhs).zeros_like_me()
            dxr = (get_gpu(self.attrs._rhs) ** (-2.0)) * - \
                1.0 * get_gpu(self.attrs._lhs) * get_gpu(dy)
            cubroadcast(dxr, new_r_dx)
            self.attrs._rhs._update_diff(context, new_r_dx)


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

# TODO:微分定義


class RTrueDiv(TrueDiv):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rtruediv__(rhs, lhs)


# TODO:微分定義


class Mod(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__mod__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy):
        pass

    def _backward_gpu(self, context, dy):
        self._backward_cpu(context, dy)


class RMod(Mod):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rmod__(rhs, lhs)

# TODO:微分定義


class DivMod(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        d, m = np.ndarray.__divmod__(lhs, rhs)
        return np.array([d, m])

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy):
        pass

    def _backward_gpu(self, context, dy):
        self._backward_cpu(context, dy)


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

    def _backward_cpu(self, context, dy):
        lhs = to_value(self.attrs._lhs)
        rhs = to_value(self.attrs._rhs)

        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, dy * (np.power(lhs, rhs - 1) * rhs))

        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, dy * self * np.log(lhs))

    def _backward_gpu(self, context, dy):

        if isinstance(self.attrs._lhs, Node):
            rhs = get_gpu(self.attrs._rhs)
            new_l_dx = get_gpu(self.attrs._lhs).zeros_like_me()
            dxl = get_gpu(dy) * rhs * (GPUValue.__pow__(get_gpu(self.attrs._lhs), rhs - 1))
            cubroadcast(dxl, new_l_dx)
            self.attrs._lhs._update_diff(context, new_l_dx)

        if isinstance(self.attrs._rhs, Node):
            lhs = get_gpu(self.attrs._lhs).empty_like_me()
            culoge(get_gpu(self.attrs._lhs), lhs)
            new_r_dx = get_gpu(dy) * get_gpu(self) * lhs
            self.attrs._rhs._update_diff(context, new_r_dx)


class RPow(Pow):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rpow__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) ** get_gpu(rhs)

# TODO:微分定義


class Lshift(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__lshift__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy):
        pass

    def _backward_gpu(self, context, dy):
        self._backward_cpu(context, dy)


class RLshift(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rlshift__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

# TODO:微分定義


class Rshift(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rshift__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy):
        pass

    def _backward_gpu(self, context, dy):
        self._backward_cpu(context, dy)


class RRshift(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rrshift__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

# TODO:微分定義


class And(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__and__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy):
        pass

    def _backward_gpu(self, context, dy):
        self._backward_cpu(context, dy)


class RAnd(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rand__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_gpu(self, context, dy):
        self._backward_cpu(context, dy)

# TODO:微分定義


class Xor(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__xor__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy):
        pass

    def _backward_gpu(self, context, dy):
        self._backward_cpu(context, dy)


class RXor(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rxor__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)


# TODO:微分定義
class Or(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__or__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy):
        pass

    def _backward_gpu(self, context, dy):
        self._backward_cpu(context, dy)


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

    def _backward_cpu(self, context, dy):
        if isinstance(self.attrs._lhs, Node):
            zero = np.zeros_like(np.array(self.attrs._lhs))
            zero[self.attrs._rhs] = np.array(dy)
            self.attrs._lhs._update_diff(context, zero)

    def _backward_gpu(self, context, dy):
        if isinstance(self.attrs._lhs, Node):
            zero = get_gpu(self.attrs._lhs).zeros_like_me()
            zero[self.attrs._rhs] = dy
            self.attrs._lhs._update_diff(context, zero)


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

    def _backward_cpu(self, context, dy):
        if isinstance(self.attrs._arg, Node):
            zero = np.zeros_like(np.array(self.attrs._arg))
            zero[self.attrs._i:self.attrs._j] = np.array(dy)
            self.attrs._arg._update_diff(context, zero)

    def _backward_gpu(self, context, dy):
        self._backward_cpu(context, dy)


def _plot_graph(objs):
    g = Digraph('G', filename='graphviz_output')

    s = set()
    for n in objs:
        g.node(str(id(n)), str(type(n)))
        s.add(id(n))

        def add_edge(node):
            if not hasattr(node, "attrs"):
                return

            nodeid = str(id(node))
            if not node.attrs:
                return
            for name in node.attrs.get_names():
                val = getattr(node.attrs, name)
                valid = str(id(val))

                g.node(valid, label=str(type(val)))
                g.edge(valid, nodeid, label=name)

            for o in node.attrs.get_attrs():
                if id(o) not in s:
                    add_edge(o)
                    s.add(id(o))

        add_edge(n)

    g.view()


try:
    from graphviz import Digraph
except ImportError:
    def plot_graph(n):   # NOQA
        pass


def DEBUG_GRAPH_INIT(active):
    if active:
        GPUValue.ACTIVE_GPU = weakref.WeakValueDictionary()
        Node.ACTIVE_NODE = weakref.WeakValueDictionary()
    else:
        GPUValue.ACTIVE_GPU = None
        Node.ACTIVE_NODE = None


def DEBUG_GPU_STAT():
    if GPUValue.ACTIVE_GPU is None:
        return

    print('Num of GPUValue: %d' % len(GPUValue.ACTIVE_GPU))
    print('Bytes of GPU   : %d' % sum(g.nbytes for g in GPUValue.ACTIVE_GPU))


def DEBUG_GET_ROOTS():
    if Node.ACTIVE_NODE is None:
        return []

    forwards = collections.defaultdict(set)
    for o in Node.ACTIVE_NODE.values():
        for ref in o.attrs.get_attrs():
            forwards[id(ref)].add(id(o))
    rootids = set(Node.ACTIVE_NODE.keys()) - set(forwards.keys())
    roots = [Node.ACTIVE_NODE[o] for o in rootids]

    return roots


def DEBUG_NODE_STAT():
    if Node.ACTIVE_NODE is None:
        return

    print('Num of Node: %d' % len(Node.ACTIVE_NODE))

    print('')
    print('Num of Node by types:')

    c = collections.Counter(str(o.__class__) for o in Node.ACTIVE_NODE.values())

    print('-----------------------------------------------------')
    print(' #\t class')
    print('-----------------------------------------------------')
    for name, n in c.most_common():
        print('%d \t%s' % (n, name))

    length = collections.Counter()

    def walk(o, n):
        if not isinstance(o, Node):
            length[n + 1] += 1
            return

        if not o.attrs:
            return
        attrs = o.attrs.get_attrs()
        if not attrs:
            length[n + 1] += 1
        else:
            for attr in attrs:
                walk(attr, n + 1)

    for root in DEBUG_GET_ROOTS():
        walk(root, 0)

    print('')
    print('Num of terminal node by graph length:')

    print('-----------------------------------------------------')
    print('#\t length')
    print('-----------------------------------------------------')
    for length, n in length.most_common():
        print('%d \t%s' % (n, length))


def DEBUG_NODE_GRAPH():
    if Node.ACTIVE_NODE is None:
        return
    roots = DEBUG_GET_ROOTS()
    _plot_graph(roots)

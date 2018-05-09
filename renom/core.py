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

        return self._refcounts[selfid] <= self._backwards[selfid], GradsWithCaller(node, self)

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
        f = time.time()
        if node.prevent_update:
            return

        with self.unlock_node(node):
            f1 = time.time()

            dy = self.get(node) if opt is None else opt(self.get(node), node)

            f2 = time.time()

            if node._auto_update:
                if callable(node.auto_update):
                    node.auto_update(dy)
                else:
                    if is_cuda_active():
                        ngpu = get_gpu(node)
                        ngpu -= get_gpu(dy)
                    else:
                        node[...] -= dy
            f3 = time.time()
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


class GradsWithCaller(object):
    def __init__(self, caller, grad):
        self.grad__ = getattr(grad, 'grad__', grad)
        self.caller = caller

    def add(self, node, dy):
        return self.grad__.add(node, dy, self.caller)

    def __getattr__(self, name):
        return getattr(self.grad__, name)

# todo: move this method to Cython


def calc_broadcast_shape(*args):
    # silly, but works
    if all([isinstance(s, (np.ndarray, Number, GPUValue)) for s in args]):
        arrays = [np.empty(getattr(s, 'shape', 1), dtype=np.bool) for s in args]
        return np.broadcast(*arrays).shape
    else:
        raise Exception("Not supported data type.")


class _AdvIndex:
    def __init__(self, index):
        if isinstance(index, (list, tuple)):
            index = np.array(index)

        isbool = index.dtype.name == 'bool'
        if isbool:
            if isinstance(index, GPUValue):
                index = index.new_array()

            elems = []
            for j, v in enumerate(index.reshape(-1)):
                if v:
                    elems.append(j)

            index = np.array(elems, dtype='int64')
        elif isinstance(index, np.ndarray):
            index = index.astype('int64')

        self.org_index = index
        if not isinstance(index, GPUValue):
            index = index.reshape(-1)
            index = GPUValue(index.astype('int64'), dtype='int64')

        if index.dtype.type is not np.int64:
            raise IndexError("Invalid index type: %r" % index.dtype)

        self.shape = index.shape
        self.index = index


def _parse_index(arr, indexes):
    if not isinstance(indexes, tuple):
        indexes = [indexes]
    else:
        indexes = list(indexes)

    ellipsis = None
    num_values = 0

    # calc number of slice or int
    for i, s in enumerate(indexes):
        if s is None:
            continue

        elif s is Ellipsis:
            if ellipsis is not None:
                assert 0
            ellipsis = i
            continue

        num_values += 1

    # expand Ellipsis or append slices at tail
    if num_values != len(arr.shape):
        if ellipsis is None:
            ellipsis = len(indexes)

        f, b = indexes[:ellipsis], indexes[ellipsis + 1:]
        rest = len(arr.shape) - num_values
        mid = list(slice(0, arr.shape[i + ellipsis], 1) for i in range(rest))
        indexes = f + mid + b

    if len([i for i in indexes if i is not None]) != len(arr.shape):
        raise IndexError()

    # build slices
    slices = []
    dest_shapes = []
    result_shapes = []

    for i, index in enumerate(indexes):
        shape = arr.shape[len(slices)]

        if isinstance(index, slice):
            start, stop, step = index.indices(shape)
            slices.append((start, stop, step))

            dest_shape = 0
            if step < 0:
                if stop < start:
                    dest_shape = (start - stop - 1) // (-step) + 1
            else:
                if start < stop:
                    dest_shape = (stop - start - 1) // step + 1

            dest_shapes.append(dest_shape)
            result_shapes.append(dest_shape)

        elif isinstance(index, int):
            if index < 0:
                index = index + shape

            if not (0 <= index < shape):
                raise IndexError()

            slices.append((index, index + 1, 1))
            dest_shapes.append(1)

        else:
            # None(newaxis)
            result_shapes.append(1)

    strides = calc_strides(arr.shape)
    dest_strides = calc_strides(arr.shape)

    return slices, strides, dest_strides, result_shapes, dest_shapes


def build_shapes(arr, indexes):
    strides = calc_strides(arr.shape)

    # make indexes a list
    if isinstance(indexes, bool):
        slices = [[0, s, 1, None, st, st] for s, st in zip(arr.shape, strides)]
        return slices, [1 if indexes else 0] + list(arr.shape), list(arr.shape)

    elif isinstance(indexes, list):
        # if indexes is in form of `[[1]]`, then unwrap the outer list.
        for elem in indexes:
            if isinstance(elem, (list, tuple, np.ndarray, GPUValue)):
                indexes = indexes[:]
                break
        else:
            indexes = [indexes]
    elif isinstance(indexes, tuple):
        indexes = list(indexes)
    else:
        indexes = [indexes]

    # check if boolean index with same shape
    if len(indexes) == 1:
        elem = indexes[0]
        if isinstance(elem, (list, tuple, np.ndarray, GPUValue)):
            if not isinstance(elem, (np.ndarray, GPUValue)):
                elem = np.array(elem)
            if elem.dtype.name == 'bool':
                if elem.shape == arr.shape:
                    idxes = _AdvIndex(elem).index
                    slices = [[0, 0, 0, idxes, 1, 1]]
                    return slices, [idxes.size], [idxes.size]

    ellipsis = None
    num_values = 0
    is_advanced = False

    # calc number of slice or index
    for i, s in enumerate(indexes):
        # check if advanced index or not
        if isinstance(s, (list, tuple, np.ndarray, GPUValue)):
            is_advanced = True

        elif s is None:
            continue

        elif s is Ellipsis:
            if ellipsis is not None:
                assert 0
            ellipsis = i
            continue

        num_values += 1

    # expand Ellipsis or append slices at tail
    if num_values != len(arr.shape):
        if ellipsis is None:
            ellipsis = len(indexes)

        f, b = indexes[:ellipsis], indexes[ellipsis + 1:]
        rest = len(arr.shape) - num_values
        mid = list(slice(0, arr.shape[i + ellipsis], 1) for i in range(rest))
        indexes = f + mid + b

    if len([i for i in indexes if i is not None]) != len(arr.shape):
        raise IndexError()

    src_shape = arr.shape
    adv_shape = []
    if is_advanced:
        # convert int index to the advanced index
        # note that 1 in the [1, []] is an advanced index
        for i, elem in enumerate(indexes[:]):
            if isinstance(elem, int):
                indexes[i] = _AdvIndex([elem])
            elif isinstance(elem, (list, tuple, np.ndarray, GPUValue)):
                indexes[i] = _AdvIndex(elem)

        # collect advanced indexes
        advs = []
        stds = []
        num_advs = 0
        all = zip(indexes, strides, src_shape)
        for k, g in itertools.groupby(all, key=lambda e: isinstance(e[0], _AdvIndex)):
            if k:
                num_advs += 1
                advs.extend(g)
            else:
                stds.extend(g)

        # check if The advanced indexes are all next to each other.
        is_split_adv = (num_advs >= 2)

        if is_split_adv:
            # move adv indexes at topmost
            indexes = ([ind for ind, stride, shape in advs] +
                       [ind for ind, stride, shape in stds])
            strides = ([stride for ind, stride, shape in advs] +
                       [stride for ind, stride, shape in stds])
            src_shape = ([shape for ind, stride, shape in advs] +
                         [shape for ind, stride, shape in stds])

        adv_shape = calc_broadcast_shape(*(adv.org_index for adv, stride, shape in advs))

    # build slices
    # (start, stop, step, adv_indexes, stride, dest_stride)
    slices = []
    result_shapes = []
    dest_shapes = []
    adv_result_shapes = adv_shape[:]
    adv_ldxsize = calc_int_prod(adv_shape)
    adv_positions = []

    n_idx = 0
    for index in indexes:
        shape = src_shape[n_idx]
        stride = strides[n_idx]

        if isinstance(index, slice):
            start, stop, step = index.indices(shape)

            dest_shape = 0
            if step < 0:
                if stop < start:
                    dest_shape = (start - stop - 1) // (-step) + 1
            else:
                if start < stop:
                    dest_shape = (stop - start - 1) // step + 1

            slices.append([start, stop, step, None, stride])
            dest_shapes.append(dest_shape)
            result_shapes.append(dest_shape)
            n_idx += 1

        elif isinstance(index, int):
            if index < 0:
                index = index + shape

            if not (0 <= index < shape):
                raise IndexError()

            slices.append([index, index + 1, 1, None, stride])
            dest_shapes.append(1)
            n_idx += 1

        elif index is None:
            # None(newaxis)
            result_shapes.append(1)

        else:  # should be sequence
            adv_positions.append(len(slices))
            maxidx = cu_reduce_max(index.index)
            if maxidx.new_array() >= shape:
                raise IndexError()

            assert index.index
            slices.append([0, 0, 0, index.index, stride])
            if adv_result_shapes:
                dest_shapes.append(adv_ldxsize)
                result_shapes.extend(adv_result_shapes)
                adv_result_shapes = None

            n_idx += 1

    dest_strides = calc_strides(dest_shapes)
    adv_dest_stride = dest_strides[adv_positions[0]] if adv_positions else None

    j = 0
    # set dest_stride
    for i in range(len(slices)):
        s = slices[i]
        if s[3] is None:
            # slice
            s.append(dest_strides[j])
            j += 1
        else:
            # adv index
            s.append(adv_dest_stride)
            j = adv_positions[0] + 1

    return slices, result_shapes, dest_shapes


def _build_broadcast_mask(left, right):
    if len(right) > len(left):
        reminds = right[:-1 * len(left)]
        for r in reminds:
            if r != 1:
                raise ValueError("could not broadcast")
        right = right[-1 * len(left):]
    elif len(right) < len(left):
        right = (1,) * (len(left) - len(right)) + right

    mask = []
    for lft, rgt in zip(left, right):
        if lft != rgt:
            if rgt != 1:
                raise ValueError("could not broadcast")
            mask.append(0)
        else:
            mask.append(1)

    return mask, right


class GPUValue(object):
    ACTIVE_GPU = None
    _ptr = None

    def __init__(self, array=None, shape=None, ptr=None, dtype=None):
        if not is_cuda_active():
            raise ValueError('Cuda is not active. '
                             'Use renom.cuda.set_cuda_active() to activate.')

        if shape is not None:
            self.shape = tuple(shape)
        else:
            self.shape = getattr(array, "shape", None) or ()

        if not dtype:
            self.dtype = np.dtype(precision)
        else:
            self.dtype = np.dtype(dtype)

        self.itemsize = np.dtype(self.dtype).itemsize
        self.size = (calc_int_prod(self.shape) if self.shape else 1) or 1
        self.nbytes = self.size * self.itemsize

        self._ptr = ptr
        if array is not None:
            self.to_gpu(array)
        elif not self._ptr:
            self.alloc()
        else:
            self.device_id = cuGetDevice()

        if self.ACTIVE_GPU is not None:
            self.ACTIVE_GPU[id(self)] = self

        assert self._ptr

    def __del__(self):
        self._free()

    def alloc(self):
        self._free()

        self._ptr = gpu_allocator.malloc(self.nbytes)
        self.device_id = cuGetDevice()
        assert self._ptr

    def _free(self):
        if self._ptr:
            gpu_allocator.free(self._ptr)
        self._ptr = None

    def reshape(self, *shape):
        clone = self.copy()
        a = np.empty(self.shape, dtype=np.bool).reshape(*shape)
        clone.shape = a.shape
        return clone

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
            ptr.memcpyH2D(value.ravel(), value.nbytes)

        self._ptr = ptr

    def copy_from(self, other):
        self._ptr.copy_from(other._ptr, self.nbytes)

    def transpose(self, axis):
        return cu_transpose(self, axis)

    def split(self, indices_or_sections, axis=0):
        N = self.shape[axis]  # Raises IndexError if axis is invalid

        try:
            len(indices_or_sections)
        except TypeError:
            size, mod = divmod(N, indices_or_sections)
            if N % indices_or_sections:
                raise ValueError(
                    'array split does not result in an equal division')
            indices_or_sections = range(size, N, size)

        slices = []
        for s in self.shape:
            slices.append(slice(0, s, 1))

        ret = []
        pos = 0
        for to in indices_or_sections:
            slices[axis] = slice(pos, to, 1)
            v = self[tuple(slices)]
            ret.append(v)
            pos = to

        if to < N:
            slices[axis] = slice(to, N, 1)
            v = self[tuple(slices)]
            ret.append(v)

        return ret

    def hsplit(self, indices_or_sections):
        return self.split(indices_or_sections, 1)

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
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            cupow(self, other, ret)
            return ret

    def __rpow__(self, other):
        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            curpow(self, other, ret)
            return ret

    def __getitem__(self, indexes):
        with use_device(self.device_id):
            slices, result_shapes, dest_shapes = build_shapes(self, indexes)

            dest_size = calc_int_prod(dest_shapes)

            ret = cu_get_item(self, self.size, dest_size, slices)

            ret.shape = result_shapes
            return ret

    def __setitem__(self, indexes, value):
        with use_device(self.device_id):
            value = get_gpu(value)
            slices, result_shapes, dest_shapes = build_shapes(self, indexes)
            if calc_int_prod(result_shapes) == 0:
                return

            dest_strides = calc_strides(dest_shapes)
            mask, broadcasted = _build_broadcast_mask(dest_shapes, value.shape)

            broadcasted_strides = calc_strides(broadcasted)
            broadcasted_strides = [m * b for m, b in zip(mask, broadcasted_strides)]

            valuesize = calc_int_prod(dest_shapes)

            cu_set_item(value, valuesize, self, slices, dest_strides, broadcasted_strides)

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
        ready, context = context.add(self, dy)
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
            # TODO: 原点における劣微分の定義
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


def get_gpu(array):
    f = getattr(array, 'get_gpu', None)
    if f:
        return f()

    if isinstance(array, np.ndarray):
        return GPUValue(array=array)
    elif isinstance(array, Number):
        return array
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

    def _backward_cpu(self, context, dy, **kwargs):
        # 次元が異なる場合の足し算を定義

        if isinstance(self.attrs._rhs, Node):
            r_dx = broad_cast(self.attrs._rhs, dy)
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(self.attrs._lhs, dy)
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        gdy = get_gpu(dy)

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

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


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

# TODO:微分定義


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

# TODO:微分定義


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

# TODO:微分定義


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

# TODO:微分定義


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


# TODO:微分定義
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
            zero = np.zeros_like(np.array(self.attrs._lhs))
            zero[self.attrs._rhs] = np.array(dy)
            self.attrs._lhs._update_diff(context, zero, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            zero = get_gpu(self.attrs._lhs).zeros_like_me()
            zero[self.attrs._rhs] = dy
            self.attrs._lhs._update_diff(context, zero, **kwargs)


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
            for val in node._args:
                valid = str(id(val))
                name = ''
                g.node(valid, label=str(type(val)))
                g.edge(valid, nodeid, label=name)

            for o in node._args:
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
        for ref in o._args:
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

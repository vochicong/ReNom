#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import inspect
import weakref
import copy
import numpy as np
from renom.core import Node, Variable, GPUValue
from renom.operation import sum
import renom.cuda
from renom.cuda import use_device, is_cuda_active
from future.utils import with_metaclass


class ModelParams(dict):
    def __init__(self, model):
        self.__dict__['model'] = weakref.proxy(model)

    def update(self, map):
        super(ModelParams, self).update(map)
        for v in map.values():
            if isinstance(v, Node):
                v.set_model(self.model)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __setattr__(self, name, value):
        super(ModelParams, self).__setitem__(name, value)
        if isinstance(value, Node):
            value.set_model(self.model)
        else:
            raise ValueError('Attribute must be Node type, not %s' % type(value))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError('%r has no attribute %r' % (self, name))


class Model(with_metaclass(ABCMeta, object)):
    """Abstract class of neural network model."""

    auto_update = False
    _prevent_update = False
    _parameters = None
    _device_id = 0
    SERIALIZED = ()

    @property
    def params(self):
        if not self._parameters:
            self._parameters = ModelParams(self)
        return self._parameters

    @params.setter
    def params(self, map):
        self._parameters = ModelParams(self)
        self._parameters.update(map)

    @property
    def device_id(self):
        return self._device_id

    def __call__(self, *args, **kwargs):
        with use_device(self._device_id):
            return self.forward(*args, **kwargs)

    def set_gpu(self, device_id):
        self.set_models(_device_id=device_id)

    @abstractmethod
    def forward(self):
        pass

    def copy_params(self, model):
        value_list = model.flatten_values()
        with use_device(self._device_id):
            for names, values, attrs in value_list:
                layer = self
                for name in names[1:]:
                    layer = getattr(layer, name)

                for k, v in values.items():
                    if k in layer.params:
                        layer.params[k].copy_from(v)
                    else:
                        layer.params[k] = v.copy()

                    layer.params[k]._auto_update = v._auto_update

    def sync(self):
        if is_cuda_active():
            done = set()
            for m in self.iter_models():
                device_id = m._device_id
                if device_id not in done:
                    done.add(device_id)
                    with use_device(m._device_id):
                        renom.cuda.cuDeviceSynchronize()

    @contextmanager
    def train(self):
        """Context manager to control whether a computational graph
        will be created or not.

        Example:
            >>> import renom as rm
            >>> import numpy as np
            >>>
            >>> class Model(rm.Model):
            ...     def __init__(self):
            ...         self._layer1 = rm.Dense(3)
            ...         self._layer2 = rm.Dense(2)
            ...     def forward(self, x):
            ...         h = rm.relu(self._layer1(x))
            ...         z = self._layer2(h)
            ...         return z
            ...
            >>> x = rm.Variable(np.random.rand(3, 3))
            >>> y = np.random.rand(3, 2)
            >>> model = Model()
            >>> z = model(x)
            >>>
            >>> with model.train():
            ...     loss = rm.mean_squared_error(z, y)
            ...
            >>> dx1 = loss.grad().get(x)
            >>> print("Gradient1 of x is \\n{}".format(dx1))
            Gradient1 of x is
            array([[ 0.85432934,  0.51205811],
                   [ 0.20379112,  0.62481132],
                   [ 0.49004569,  0.35310219]])
            >>>
            >>> loss = rm.mean_squared_error(z, y)
            >>> dx2 = loss.grad().get(x)
            >>> print("Gradient2 of x is \\n{}".format(dx2))
            Gradient2 of x is
            None
        """
        self.detach_graph()
        self.set_auto_update(True)
        try:
            yield self
        finally:
            self.set_auto_update(False)

    @contextmanager
    def prevent_update(self):
        self.set_prevent_update(True)
        try:
            yield self
        finally:
            self.set_prevent_update(False)

    def iter_models(self):
        yield self

        for k, v in self.__dict__.items():
            if isinstance(v, Model):
                for c in v.iter_models():
                    yield c

    def _get_values(self, values):
        if self.params:
            for k, v in self.params.items():
                values[1][k] = v

        serialized = getattr(self, "SERIALIZED", ())
        for name in serialized:
            if hasattr(self, name):
                values[2][name] = getattr(self, name)

        for k, v in self.__dict__.items():
            if isinstance(v, Model):
                childvalues = ({}, {}, {})
                v._get_values(childvalues)
                values[0][k] = childvalues

    def values(self):
        """
        Generates nested tuple of underlying models and params of models.

        Each model generates tuple of two dictionary. The first dictionary
        contains child models, keyed by attribute name. The second dictionary
        contains parameters of the model, keyed by attribute name.

        Example:
            .. code-block:: python

                (
                    # child models of self
                    {
                        'layer1': (
                            {},     # child model of self.layer1
                            {       # params of layer1
                                'w': [1,2],   # self.layer1.params.w
                                'b': [3,4],   # self.layer1.params.b
                            }
                        ),
                        'layer2': (
                            {},     # child model of self.layer2
                            {       # params of layer2
                                'w': [1,2],   # self.layer2.params.w
                                'b': [3,4],   # self.layer2.params.b
                            }
                    },
                    # params of self
                    {}
                )
        """
        ret = ({}, {}, {})
        self._get_values(ret)
        return ret

    def flatten_values(self):
        values = self.values()
        value_list = []

        def flatten(names, values):
            value_list.append((names, values[1], values[2]))

            for name, child_values in values[0].items():
                flatten(names + (name,), child_values)

        flatten(('root',), values)
        return value_list

    def _get_grads(self, grads):
        "Get gradients of attribute of this model"
        value_list = self.flatten_values()

        d = {}
        for name, values, attrs in value_list:
            for k, v in values.items():
                diff = grads.get(v, None)
                if diff is not None:
                    d[(name, k)] = diff
        return d

    def join_grads(self, grads, others):
        """Merge gradients of other models.
        Others is a list of tuple of (model, grads) to be merged.
        Models listed in the others should have same structure with self."""

        values = {name: params for name, params, attrs in self.flatten_values()}
        for model, _grads in others:
            o = model._get_grads(_grads)

            for (name, attrname), diff in o.items():
                obj = values[name][attrname]
                curdiff = grads.get(obj, None)
                if curdiff is not None:
                    if not isinstance(curdiff, Node):
                        curdiff = Node(curdiff)
                    if not isinstance(diff, Node):
                        diff = Node(diff)
                    with use_device(curdiff.device_id):
                        if diff.device_id != curdiff.device_id:
                            diff = Node(diff.get_gpu().copy())
                        newdiff = curdiff + diff

                grads.set(obj, newdiff)

    def save(self, filename):
        """Save model attributes.
        For save attributes, please register attributes to the dictionary
        which is named as 'SERIALIZED'.

        Following example shows how to do it.

        Example:
            >>> import renom as rm
            >>> import numpy as np
            >>>
            >>> class MyModel(rm.Model):
            ...     SERIALIZED = ('_moving_avg', ) # Put any attributes for saving.
            ...     def __init__(self):
            ...         super(MyModel, self).__init__()
            ...         self._l1 = rm.Dense(2)
            ...         self._l2 = rm.Dense(1)
            ...         self._moving_avg = 0
            ...     def forward(self, x):
            ...         h = self._l1(x)
            ...         h = rm.relu(h)
            ...         h = self._l2(h)
            ...         self._moving_avg = np.float32(self._moving_avg*0.5 + rm.sum(h)*0.5)
            ...         return h
            ...
            >>> model = MyModel()
            >>> model(np.random.rand(12, 4))
            >>> print(model._moving_avg)
            1.95637
            >>> model.save("test.h5") # Save
            >>> model = MyModel() # Reset model object.
            >>> model.load("test.h5") # Load
            >>> print(model._moving_avg)
            1.95637

        Args:
            filename (str): File name to save model.

        """
        import h5py

        value_list = self.flatten_values()
        with h5py.File(filename, 'w') as f:
            values_grp = f.create_group('values')
            types_grp = f.create_group('types')

            for names, params, attrs in value_list:
                g = values_grp.create_group('.'.join(names))
                t = types_grp.create_group('.'.join(names))

                for propname, propvalue in params.items():
                    propvalue.to_cpu()
                    g[propname] = propvalue

                    if isinstance(propvalue, Variable):
                        # todo: move to Node/Variable
                        t[propname] = 'renom.Variable'
                        t[propname + '._auto_update'] = propvalue._auto_update

                    elif isinstance(propvalue, Node):
                        t[propname] = 'renom.Node'

                for propname, propvalue in attrs.items():
                    if isinstance(propvalue, GPUValue):
                        g['__dict__.' + propname] = propvalue.new_array()
                    else:
                        g['__dict__.' + propname] = propvalue

    def load(self, filename):
        """Load saved weights to model.

        Args:
            filename (str): File name of saved model.

        Example:
            >>> model = rm.Dense(2)
            >>> model.load("model.hd5")
        """
        import h5py
        f = h5py.File(filename, 'r')
        values = f['values']
        types = f['types']

        names = sorted(values.keys())

        def get_attr(root, names):
            names = names.split('.')[1:]
            ret = root
            for name in names:
                ret = getattr(ret, name)
            return ret

        target = self
        for name in names:
            target = get_attr(self, name)

            values_grp = values[name]
            types_grp = types[name]

            for k, v in values_grp.items():
                v = v.value
                if isinstance(v, np.ndarray):
                    type = types_grp.get(k, None)
                    if type:
                        if type.value == 'renom.Variable':
                            auto_update = types_grp[k + '._auto_update'].value
                            v = Variable(v, auto_update=auto_update)
                        else:
                            v = Node(v)

                if k.startswith('__dict__.'):
                    obj = target
                    name = k.split(".", 1)[1]
                else:
                    obj = target.params
                    name = k

                setattr(obj, name, v)

    def detach_graph(self):
        for c in self.iter_models():
            if c.params:
                for p in self.params.values():
                    if p is not None:
                        p.detach_graph()

    def set_auto_update(self, f):
        self.set_models(auto_update=f)

    def set_prevent_update(self, f):
        self.set_models(_prevent_update=f)

    def set_models(self, **kwargs):
        for c in self.iter_models():
            for k, v in kwargs.items():
                setattr(c, k, v)

    def truncate(self):
        for c in self.iter_models():
            if isinstance(c, Parametrized):
                c.truncate()


class Sequential(Model):
    """Sequential model.

    Args:
        layers (list): A list of layer objects.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>>
        >>> x = np.random.rand(32, 50)
        >>> sequential = rm.Sequential([
        ...         rm.Dense(100),
        ...         rm.Relu(),
        ...         rm.Dense(10),
        ...     ])
        ...
        >>> z = sequential(x)
        >>> z.shape
        (32, 10)
    """

    def __init__(self, layers, loss_function=None):
        self._layers = list(layers)
        for i, ly in enumerate(layers):
            setattr(self, "l%d" % (i), ly)

    def __call__(self, x):
        with use_device(self._device_id):
            return self.forward(x)

    def append(self, layer):
        setattr(self, "l%d" % (len(self._layers)), layer)
        self._layers.append(layer)

    def summary(self):
        print("---------------------------------")
        print("summary will be printed out soon.")

    def forward(self, x):
        t = x
        for ly in self._layers:
            t = ly(t)
        return t

    def __getitem__(self, i):
        return self._layers[i]


class Parametrized(Model):

    def __init__(self, input_size=None):
        if input_size is not None:
            self.weight_initiallize(input_size)

    def weight_initiallize(self, input_size):
        raise NotImplementedError

    def __call__(self, x, *args, **kwargs):
        with use_device(self._device_id):
            if not self.params:
                self.weight_initiallize(x.shape[1:])
            return super(Parametrized, self).__call__(x, *args, **kwargs)

    def truncate(self):
        pass

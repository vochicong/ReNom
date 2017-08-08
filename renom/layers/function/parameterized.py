#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import inspect
import weakref
import numpy as np
from renom.core import Node, Variable
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

    @property
    def params(self):
        if not self._parameters:
            self._parameters = ModelParams(self)
        return self._parameters

    @params.setter
    def params(self, map):
        self._parameters = ModelParams(self)
        self._parameters.update(map)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self):
        pass

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
                if isinstance(v, Node):
                    v.to_cpu()
                values[1][k] = v

        for k, v in self.__dict__.items():
            if isinstance(v, Model):
                childvalues = ({}, {})
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
        ret = ({}, {})
        self._get_values(ret)
        return ret

    def save(self, filename):
        """Save model weights.

        Args:
            filename (str): File name to save model.

        Example:
            >>> model = rm.Dense(2)
            >>> model.save("model.hd5")
        """
        import h5py

        values = self.values()
        value_list = []

        def flatten(names, values):
            value_list.append((names, values[1]))

            for name, child_values in values[0].items():
                flatten(names + (name,), child_values)

        flatten(('root',), values)

        with h5py.File(filename, 'w') as f:
            values_grp = f.create_group('values')
            types_grp = f.create_group('types')

            for names, values in value_list:
                g = values_grp.create_group('.'.join(names))
                t = types_grp.create_group('.'.join(names))

                for propname, propvalue in values.items():
                    g[propname] = propvalue

                    if isinstance(propvalue, Variable):
                        t[propname] = 'renom.Variable'
                        t[propname + '._auto_update'] = propvalue._auto_update

                    elif isinstance(propvalue, Node):
                        t[propname] = 'renom.Node'

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

                setattr(target.params, k, v)

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
        self._layers = layers
        for i, ly in enumerate(layers):
            setattr(self, "l%d" % (i), ly)

    def __call__(self, x):
        return self.forward(x)

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
        if not self.params:
            self.weight_initiallize(x.shape[1:])
        return super(Parametrized, self).__call__(x, *args, **kwargs)

    def truncate(self):
        pass

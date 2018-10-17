import collections
import onnx.helper
import onnx.numpy_helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
import numpy as np

import renom

def _to_param_name(obj):
    return str(id(obj))


MODELDEFS= {}
def modeldef(target):
    def wrapper(cls):
        MODELDEFS[target] = cls
        return cls

    return wrapper

class _ModelNode(renom.Node):
    def __new__(cls, model, output, input):
        ret = super(_ModelNode, cls).__new__(cls, output)
        ret.attrs.input = input
        ret.model = model
        ret.output = output

        return ret

@modeldef(renom.Dense)
class DenseModel(_ModelNode):
    def register_params(self, onnx_nodes, inputs, outputs, values):
        w = self.model.params["w"]
        inputs.add(w)
        values.add(w)

        b = self.model.params["b"]
        inputs.add(b)
        values.add(b)

        onnx_nodes.append(
            onnx.helper.make_node(
                'Gemm', 
                [_to_param_name(v) for v in (self.attrs.input, w, b)],
                [_to_param_name(self)]))

@modeldef(renom.Conv2d)
class Conv2dModel(_ModelNode):
    def register_params(self, onnx_nodes, inputs, outputs, values):
        w = self.model.params["w"]
        inputs.add(w)
        values.add(w)

        b = self.model.params["b"]
        inputs.add(b)
        values.add(b)

        onnx_nodes.append(
            onnx.helper.make_node(
                'Conv',
                [_to_param_name(v) for v in (self.attrs.input, w, b)],
                [_to_param_name(self)],
                kernel_shape=self.output.attrs._kernel,
                pads=self.output.attrs._padding,
                strides=self.output.attrs._stride,
                dilations=self.output.attrs._dilation))


NODEDEFS = {}

def nodedef(target):
    def wrapper(f):
        NODEDEFS[target] = f
        return f
    return wrapper

@nodedef(renom.relu)
def register_relu(onnx_nodes, inputs, outputs, values, node):
    onnx_nodes.append(
        onnx.helper.make_node(
            'Relu',
            [_to_param_name(node.attrs._arg)],
            [_to_param_name(node)]))


@nodedef(renom.max_pool2d)
def register_max_pool2d(onnx_nodes, inputs, outputs, values, node):
    onnx_nodes.append(
        onnx.helper.make_node(
            'MaxPool',
            [_to_param_name(node.attrs._x)],
            [_to_param_name(node)],
            kernel_shape=node._kernel))


@nodedef(renom.dropout)
def register_dropout(onnx_nodes, inputs, outputs, values, node):
    onnx_nodes.append(
        onnx.helper.make_node(
            'Dropout',
            [_to_param_name(node.attrs._x)],
            [_to_param_name(node)],
            ratio=node._ratio))


@nodedef(renom.Reshape)
def register_reshape(onnx_nodes, inputs, outputs, values, node):
    shape = np.array(node._shape_to)
    inputs.add(shape)
    values.add(shape)

    onnx_nodes.append(
        onnx.helper.make_node(
            'Reshape',
            [_to_param_name(node.attrs._array), _to_param_name(shape)],
            [_to_param_name(node)]))



def register_node(onnx_nodes, inputs, outputs, values, node):
    if isinstance(node, _ModelNode):
        node.register_params(onnx_nodes, inputs, outputs, values)
        return

    f = NODEDEFS.get(type(node))
    if f:
        f(onnx_nodes, inputs, outputs, values, node)
    elif isinstance(node, renom.Variable):
        pass
    else:
        raise ValueError('Unknown node: %s' % type(node))

class IdDict(dict):
    def add(self, obj):
        self[id(obj)] = obj


class OnnxHook:
    def call_enter(self, model, x, args, kwargs):
        return x, args, kwargs

    def call_leave(self, model, ret, x, args, kwargs):
        return ret

    def on_forward(self, model, forward, x, args, kwargs):
        output = forward(x, *args, **kwargs)
        conv = MODELDEFS.get(type(model), None)
        if conv:
            return conv(model, output, x)

        return output

    def leave_create(self, nodecls, ret):
        return ret


def value_info(value):
    return onnx.helper.make_tensor_value_info(
                _to_param_name(value),
                NP_TYPE_TO_TENSOR_TYPE[value.dtype],
                value.shape)

def export_onnx(name, model, x):

    hook = OnnxHook()
    renom.Model.set_hook(hook)
    renom.Node.set_hook(hook)

    x = renom.Variable(x)
    try:
        with model.train():
            ret = model(x)
    finally:
        renom.Model.set_hook(None)

    cur = [ret]
    parent_nodes = collections.defaultdict(set)
    child_nodes = collections.defaultdict(set)

    nodes = IdDict()

    # build tree
    while cur:
        node = cur.pop(0)
        if not isinstance(node, renom.Node):
            continue

        parents = [n for n in node._get_graph() if isinstance(n, renom.Node)]
        cur.extend(parents)

        nodes.add(node)
        for parent in parents:
            nodes.add(parent)
            parent_nodes[id(node)].add(id(parent))
            child_nodes[id(parent)].add(id(node))

    # sort tree
    sorted = []
    remains = [x]
    while remains:
        node = remains.pop(0)
        sorted.append(node)

        children = child_nodes[id(node)]
        for child in children:
            parents = parent_nodes[child]
            parents.remove(id(node))
            if not parents:
                remains.append(nodes[child])

    # sort extract params

    inputs = IdDict()
    inputs.add(x)

    outputs = IdDict()
    outputs.add(ret)

    onnx_nodes = []

    values = IdDict()
    for node in sorted:
        register_node(onnx_nodes, inputs, outputs, values, node)

    if id(x) in values:
        del values[id(x)]


    inputs = [value_info(v) for v in inputs.values()]
    outputs = [value_info(v) for v in outputs.values()]
    initializers = [onnx.numpy_helper.from_array(v, _to_param_name(v))
                    for v in values.values()]

    onnx_graph = onnx.helper.make_graph(
        onnx_nodes, name, inputs, outputs, initializer=initializers)

    model = onnx.helper.make_model(
        onnx_graph,
        producer_name='renom',
        producer_version=renom.__version__
    )

    with open(name+".onnx", 'wb') as f:
        f.write(model.SerializeToString())

    with open(name+".onnx.txt", 'w') as f:
        print(model, file=f)

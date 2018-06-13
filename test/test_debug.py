#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from renom.core import Variable, DEBUG_NODE_STAT, DEBUG_GRAPH_INIT, DEBUG_NODE_GRAPH
from renom import operation as O
import renom as R


def test_node_dump():
    DEBUG_GRAPH_INIT(True)

    a = Variable(np.array([1, 2, 3, 4, 5]))
    b = Variable(np.array([1, 2, 3, 4, 5]))
    c = a + b  # NOQA

    d = a + b * 2  # NOQA

    DEBUG_NODE_STAT()
    # DEBUG_NODE_GRAPH()

    DEBUG_GRAPH_INIT(False)


def test_node_clear():
    DEBUG_GRAPH_INIT(True)

    a = Variable(np.random.rand(2, 2).astype(np.float32))
    b = Variable(np.random.rand(2, 2).astype(np.float32))

    layer = R.Lstm(2)

    c = layer(O.dot(a, b))  # NOQA

    DEBUG_NODE_STAT()
#    DEBUG_NODE_GRAPH()
    DEBUG_GRAPH_INIT(False)


# test_node_clear()


from renom.core import Node
from renom.layers.function.parameterized import ModelMark
import collections
from graphviz import Digraph


class MNist(R.Model):
    def __init__(self):
        super(MNist, self).__init__()
        self.layer0 = R.Dense(output_size=50)
        self.layer1 = R.PeepholeLstm(output_size=50)
        self.layer2 = R.Dense(output_size=10)

    def forward(self, x):
        self.truncate()
        ret = 0
        for i in range(28):
            lstm = self.layer1(x[:, i])
            ret = self.layer0(lstm)
            ret = self.layer2(ret)
        return ret

    def forward(self, x):
        ret = self.layer0(x)
#        ret = self.layer2(ret)
#        print(2222222222222, type(ret))
        return ret





def test_model_graph():
    DEBUG_GRAPH_INIT(True)

    m = MNist()
    x = np.random.rand(2800).reshape((10,28,10))
    x = np.random.rand(100).reshape((10,10))

    nn = MNist()
    v = nn(x)

    models = {}
    ret = collections.defaultdict(set)

    modelnames = {id(m):(name, m) for name, m in nn.get_models('nn')}

    def label(modelid):
        if modelid in modelnames:
            name, model = modelnames[modelid]
            return '%s(%s)' % (name, type(model).__name__)
        else:
            model = models[modelid]
            return '(%s)' % type(model).__name__




    def walk(ret, models, seen, obj, model):
        import pdb;pdb.set_trace()
        if not isinstance(obj, Node):
            return

        if isinstance(obj, ModelMark):
            if model:
                models[id(obj.modelmark())] = obj.modelmark()

            if model is not None:
                ret[id(obj.modelmark())].add(id(model))

            model = obj.modelmark()

        for attr in obj.attrs.get_attrs():
            id_attr = id(attr)
            if id_attr not in seen:
                seen.add(id_attr)
                walk(ret, models, seen, attr, model)












    s = set()

    def addnode(modelid):
        if modelid not in s:
            g.node(str(modelid), label(modelid))
            s.add(modelid)
        return modelid



    walk(ret, models, set(), v, None)


    g = Digraph('G', filename='graphviz_output')


    for to, froms in ret.items():
        toid = addnode(to)

        for f in froms:
            fromid = addnode(f)
            g.edge(str(fromid), str(toid), label='')


    g.view()



    DEBUG_GRAPH_INIT(False)


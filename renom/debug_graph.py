import itertools
import collections
import weakref
import renom.core

try:
    from graphviz import Digraph, Graph
except ImportError:
    def plot_graph(n):   # NOQA
        pass


ACTIVE_GPU = None
ACTIVE_NODE = None


def DEBUG_GRAPH_INIT(active):
    global ACTIVE_GPU, ACTIVE_NODE
    if active:
        ACTIVE_GPU = weakref.WeakValueDictionary()
        ACTIVE_NODE = weakref.WeakValueDictionary()
    else:
        ACTIVE_GPU = None
        ACTIVE_NODE = None


def GET_ACTIVE_NODE():
    global ACTIVE_NODE
    return ACTIVE_NODE


def SET_NODE_DICT(id, val):
    global ACTIVE_NODE
    ACTIVE_NODE[id] = val


def GET_ACTIVE_GPU():
    global ACTIVE_GPU
    return ACTIVE_GPU


def SET_GPU_DICT(id, val):
    global ACTIVE_GPU
    ACTIVE_GPU[id] = val


def DEBUG_GPU_STAT():
    if ACTIVE_GPU is None:
        return

    print('Num of GPUValue: %d' % len(ACTIVE_GPU))
    print('Bytes of GPU   : %d' % sum(g.nbytes for g in ACTIVE_GPU))


def DEBUG_GET_ROOTS():
    if ACTIVE_NODE is None:
        return []

    forwards = collections.defaultdict(set)
    for o in ACTIVE_NODE.values():
        for ref in o._args:
            forwards[id(ref)].add(id(o))
    rootids = set(ACTIVE_NODE.keys()) - set(forwards.keys())
    roots = [ACTIVE_NODE[o] for o in rootids]

    return roots


def DEBUG_NODE_STAT():
    if ACTIVE_NODE is None:
        return

    print('Num of Node: %d' % len(ACTIVE_NODE))

    print('')
    print('Num of Node by types:')

    c = collections.Counter(str(o.__class__) for o in ACTIVE_NODE.values())

    print('-----------------------------------------------------')
    print(' #\t class')
    print('-----------------------------------------------------')
    for name, n in c.most_common():
        print('%d \t%s' % (n, name))

    length = collections.Counter()

    def walk(o, n):
        if not hasattr(o, "attrs"):
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
    if ACTIVE_NODE is None:
        return
    roots = DEBUG_GET_ROOTS()
    _plot_graph(roots)


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




class _Box:
    def __init__(self, obj):
        self.nexts = []
        self.obj = obj
        self.join = None

    def addnext(self, nextbox):
        self.nexts.append(nextbox)

    def joinnext(self):
        model = self.obj.modelmark()
        for c in self.nexts:
            if c.obj.modelmark() is model:
                c.join = self

    def create_node(self, context, graph):
        if self.join:
            return

        model = self.obj.modelmark()

        modelinfo = context.get_modelinfo(model)
        if modelinfo.children:
            shape='circle'
            color = 'gray'
            if isinstance(self.obj, renom.core.EnterModel):
                label = 'S'
            else:
                label = 'E'
        else:
            shape = 'box'
            color = 'white'
            name = context.get_modelinfo(model).name
            label = '%s(%s)' % (name, type(model).__name__)
            color = 'white'

        graph.node(str(id(self.obj)), label=label, shape=shape, style='filled', fillcolor=color, color='black')


    def nodelabel(self):
        if self.join:
            return str(id(self.join.obj))
        else:
            return str(id(self.obj))


    def create_edge(self, context):
        f = self.nodelabel()

        for c in self.nexts:
            if c.join is self:
                continue
            t = c.nodelabel()
            context.root.graph.edge(f, t)




class _ModelInfo:
    def __init__(self, parent, name, model):
        self.parent = parent
        self.children = weakref.WeakSet()
        if parent:
            self.parent.children.add(self)

        self.nodes = []

        self.name = name
        self.model = model
        self.graph = None

    def create_graph(self, context):
        self.graph = Digraph(name='cluster=' + self.name)
        self.graph.attr(label='%s(%s)' % (self.name, self.model.__class__.__name__),
            labelloc='top', labeljust='left')

        for node in self.nodes:
            node.create_node(context, self.graph)

    def addnode(self, node):
        self.nodes.append(node)

class ModelGraphContext:

    def __init__(self):
        pass
        
    def get_modelinfo(self, model):
        return self.models.get(id(model))

    def walk_model(self, model):
        self.models = {}

        models = [(None, 'root', model)]

        while models:
            parent, name, model = models.pop()
            p = _ModelInfo(parent, name, model)
            if not parent:
                self.root = p

            self.models[id(model)] = p

            models.extend((p, k, v) for k, v in model.get_model_children())


    def _getBox(self, node):
        nodeid = id(node)
        if nodeid not in self.boxes:
            modelinfo = self.get_modelinfo(node.modelmark())
            box = _Box(node)
            if modelinfo.children:
                modelinfo.addnode(box)
            if modelinfo.parent:
                modelinfo.parent.addnode(box)
            else:
                self.root.addnode(box)

            self.boxes[nodeid] = box
            
            
        return self.boxes[nodeid]

    def walk_node(self, node):
        self.boxes = {}
        self._walk_node(node, None, set())
        for box in self.boxes.values():
            box.joinnext()

    def _walk_node(self, node, nextbox, seen):
        if not isinstance(node, renom.core.Node):
            return

        if isinstance(node, renom.core.ModelMark):
            box = self._getBox(node)

            if nextbox is not None:
                box.addnext(nextbox)

            nextbox = box

        id_node = id(node)
        if id_node in seen:
            return
        seen.add(id_node)

        for attr in node.attrs.get_attrs():
            self._walk_node(attr, nextbox, seen)

    def build_subgraph(self):

        # build pathes from root to leaf
        leafs = []
        q = collections.deque([(self.root, [])])
        while q:
            model, path = q.pop()
            path = [model,] + path
            if not model.children:
                leafs.append(path)
            else:
                model.create_graph(self)
                for c in model.children:
                    q.append((c, path))


#        for c in leafs:
#            print('-'.join(m.name for m in c))

        # create sub graphs from leaf to root
        leafs.sort(key=len, reverse=True)
        seen = set()
        for leaf in leafs:
            while len(leaf) >= 2:
                child = leaf.pop(0)
                parent = leaf[0]
                if (child, parent) in seen:
                    break

                parent.graph.subgraph(child.graph)
                seen.add((child, parent))


        for box in self.boxes.values():
            box.create_edge(self)


    def build(self, nnmodel, value):
        self.walk_model(nnmodel)
        self.walk_node(value)
        self.build_subgraph()

        print(self.root.graph.source)
        self.root.graph.view()
#        self.graphs[id(nnmodel)].view()
#        for g in self.graphs.values():
#            print("--------------------")
#            print(g.source)

def DEBUG_MODELGRAPH(nnmodel, value):
    c = ModelGraphContext()
    c.build(nnmodel, value)

    

import collections
import weakref

try:
    from graphviz import Digraph
except ImportError:
    def plot_graph(n):   # NOQA
        pass


_ACTIVE_GPU = None
_ACTIVE_NODE = None


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
        if not hasattr(o,"attrs"):
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

from graphviz import Digraph
from cranberry import Tensor

def trace(root: Tensor):
    nodes, edges = set(), set()
    def build(v: Tensor):
        if v not in nodes:
            nodes.add(v)
            if v._prev is not None:
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
    build(root)
    return nodes, edges

def draw_graph(root: Tensor, format='svg', rankdir='TB'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        dot.node(name=str(n.__hash__()), label=n.__repr__(), shape='record')
    for n1, n2 in edges:
        dot.edge(str(n1.__hash__()), str(n2.__hash__()))

    return dot           
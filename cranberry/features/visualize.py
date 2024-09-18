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


def truncate(s: str, length=20):
  return s if len(s) <= length else s[:length] + "..."


def plot_graph(root: Tensor, fmt="svg", rankdir="LR"):
  """
  format: png | svg | ...
  rankdir: TB (top to bottom graph) | LR (left to right)
  """
  assert rankdir in ["LR", "TB"]
  nodes, edges = trace(root)
  dot = Digraph(format=fmt, graph_attr={"rankdir": rankdir})

  for n in nodes:
    label = f"data = {truncate(str(n.data.round(2)))}"
    if n.requires_grad:
      label += f"|grad = {truncate(str(n.grad.round(2)))}"
    # if n.shape: label += f"|shape = {n.shape}"
    if n.op is not None:
      label += f"|op = {n.op.__repr__()}"
    dot.node(name=str(n.__hash__()), label=label, shape="record")
  for n1, n2 in edges:
    dot.edge(str(n1.__hash__()), str(n2.__hash__()))

  return dot

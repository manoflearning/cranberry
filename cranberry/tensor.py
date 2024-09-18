from __future__ import annotations
import math
from cranberry.ops import Op, UnaryOps, BinaryOps, ReduceOps, MetaOps
import time
from typing import List, Optional, Tuple, Union
import numpy as np
from math import prod
from cranberry.view import View
from cranberry import StoragePtr


def list_to_size(x: List) -> int:
  size = 1
  while isinstance(x, List):
    size *= len(x)
    x = x[0]
  return size


def list_to_view(x: List) -> View:
  shape = []
  while isinstance(x, List):
    shape.append(len(x))
    x = x[0]
  return View.create(tuple(shape))


class Tensor:
  def __init__(
    self,
    data: Union[StoragePtr, float, int, List, np.ndarray, np.float32],
    view: Optional[Union[View, Tuple]] = None,
    requires_grad: bool = False,
    prev: Optional[Tuple[Tensor, ...]] = None,
    op: Optional[Op] = None,
    device: str = "cpu",
  ):
    match data:
      case StoragePtr():
        self._data = data
      case int() | float():
        self._data = StoragePtr.full(value=data, size=1, device=device)
      case list():
        self._data = StoragePtr.from_vec(vec=data, device=device)
      case np.ndarray():
        self._data = StoragePtr.from_vec(vec=data.flatten().astype(np.float32), device=device)
      case np.float32():
        self._data = StoragePtr.full(value=data, size=1, device=device)
      case _:
        raise ValueError(f"invalid data type {type(data)}")

    match requires_grad:
      case True:
        self._grad = Tensor(
          data=StoragePtr.full(value=0.0, size=self._data.size, device=device),
          view=self._view,
          requires_grad=False,
          prev=(),
          op=None,
          device=device,
        )
      case False:
        self._grad = None

    match view:
      case View():
        self._view = view
      case Tuple():
        self._view = View.create(view)
      case None:
        if isinstance(data, float, int):
          self._view = View.create(())
        elif isinstance(data, List):
          self._view = list_to_view(data)
        elif isinstance(data, np.ndarray):
          self._view = View.create(data.shape)
        elif type(data) is np.float32:
          self._view = View.create(data.shape)
        else:
          raise ValueError(f"cannot obtain view from {type(data)}")
      case _:
        raise ValueError(f"invalid view type {type(view)}")

    self._requires_grad: bool = requires_grad
    self._backward = lambda: None
    self._prev = prev
    self._op = op

    assert type(self._data) is StoragePtr
    assert type(self._grad) is Tensor or self._grad is None
    assert type(self._view) is View
    assert type(self._requires_grad) is bool
    assert callable(self._backward)
    assert type(self._prev) is tuple or self._prev is None
    assert type(self._op) is Op or self._op is None

    assert self._data.device == device and (self._grad is None or self._grad.device == device)
    assert self._requires_grad == (self._grad is not None)
    assert (self._op is not None) == (self._prev is not None)

  def assign(self, other: Tensor) -> Tensor:
    assert isinstance(other, Tensor)
    assert self._view == other._view, f"assign view mismatch {self._view} != {other._view}"
    assert self._data.device == other._data.device, f"assign device mismatch {self._data.device} != {other._data.device}"
    StoragePtr.assign(other._data, self._data)
    self

  # ********************************************************
  # ***************      backward prop       ***************
  # ********************************************************

  def backward(self):
    assert self._requires_grad, "cannot call backward on a tensor that doesn't require gradients"
    assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

    topo = []
    visited = set()

    def dfs(t: Tensor):
      visited.add(t)
      if t._prev is not None:
        for v in t._prev:
          if v not in visited and v._requires_grad:
            dfs(v)
      topo.append(t)

    dfs(self)

    self._grad.fill(1.0)
    for v in reversed(topo):
      v._backward()

  # ********************************************************
  # ***************       broadcasting       ***************
  # ********************************************************

  def _broadcasted(self, other: Tensor) -> Tuple[Tensor, Tensor]:
    shape1, shape2 = self._view.shape, other._view.shape
    while len(shape1) < len(shape2):
      shape1 = (1,) + shape1
    while len(shape2) < len(shape1):
      shape2 = (1,) + shape2
    shape = tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))
    if self.shape != shape:
      self = self.expand(*shape)
    if other.shape != shape:
      other = other.expand(*shape)
    return self, other

  # ********************************************************
  # ***************         unary ops        ***************
  # ********************************************************

  def _unary_op(self, op: UnaryOps) -> Tensor:
    out = Tensor(
      data=StoragePtr.full(value=0.0, size=self._data.size, device=self._device),
      view=self._view,
      requires_grad=self.requires_grad,
      prev=(self,),
      op=op,
      device=self._device,
    )

    idxs = View.unary_op_indexing(self._view, out._view)

    match op:
      case UnaryOps.NEG:
        for i, j, size in idxs:
          StoragePtr.neg(self._data, out._data, i, j, size)
        out._backward = lambda: self._grad.__iadd__(out._grad)
      case UnaryOps.SQRT:
        for i, j, size in idxs:
          StoragePtr.sqrt(self._data, out._data, i, j, size)
        out._backward = lambda: self._grad.__iadd__(0.5 * out._grad / out.detach())
      case UnaryOps.EXP:
        for i, j, size in idxs:
          StoragePtr.exp(self._data, out._data, i, j, size)
        out._backward = lambda: self._grad.__iadd__(out._grad * out.detach())
      case UnaryOps.LOG:
        for i, j, size in idxs:
          StoragePtr.log(self._data, out._data, i, j, size)
        out._backward = lambda: self._grad.__iadd__(out._grad / out.detach())
      case _:
        raise RuntimeError(f"invalid unary op {op}")
    return out

  def neg(self) -> Tensor: return self._unary_op(UnaryOps.NEG)
  def sqrt(self) -> Tensor: return self._unary_op(UnaryOps.SQRT)
  def exp(self) -> Tensor: return self._unary_op(UnaryOps.EXP)
  def log(self) -> Tensor: return self._unary_op(UnaryOps.LOG)

  def __neg__(self) -> Tensor: return self.neg()

  def sigmoid(self) -> Tensor: return 1 / (1 + self.neg().exp())
  def tanh(self) -> Tensor: return (self.exp() - self.neg().exp()) / (self.exp() + self.neg().exp())
  def relu(self) -> Tensor: return self.gt(0).mul(self)
  def gelu(self) -> Tensor: return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())

  # ********************************************************
  # ***************        binary ops        ***************
  # ********************************************************

  def _binary_op(self, other: Union[Tensor, float, int], reverse: bool, op: BinaryOps) -> Tensor:
    if isinstance(other, Union[float, int]): other = Tensor(other)
    self, other = self._broadcasted(other)
    if reverse: self, other = other, self

    assert self._data.device == other._data.device, "devices of operands must be the same"

    out = Tensor(
      data=StoragePtr.full(value=0.0, size=self._data.size, device=self._device),
      view=self._view,
      requires_grad=self.requires_grad or other.requires_grad,
      prev=(self, other),
      op=op,
      device=self._data.device,
    )

    idxs = View.binary_op_indexing(self._view, other._view, out._view)

    match op:
      case BinaryOps.ADD:
        for i, j, k, size in idxs:
          StoragePtr.add(self._data, other._data, out._data, i, j, k, size)
        out._backward = lambda: (
          self._grad.__iadd__(out._grad),
          other._grad.__iadd__(out._grad),
        )
      case BinaryOps.SUB:
        for i, j, k, size in idxs:
          StoragePtr.sub(self._data, other._data, out._data, i, j, k, size)
        out._backward = lambda: (
          self._grad.__iadd__(out._grad),
          other._grad.__isub__(out._grad),
        )
      case BinaryOps.MUL:
        for i, j, k, size in idxs:
          StoragePtr.mul(self._data, other._data, out._data, i, j, k, size)
        out._backward = lambda: (
          self._grad.__iadd__(out._grad * other.detach()),
          other._grad.__iadd__(out._grad * self.detach()),
        )
      case BinaryOps.DIV:
        for i, j, k, size in idxs:
          StoragePtr.div(self._data, other._data, out._data, i, j, k, size)
        out._backward = lambda: (
          self._grad.__iadd__(out._grad / other.detach()),
          other._grad.__isub__(out._grad * self.detach() / other.detach() ** 2),
        )
      case BinaryOps.CMPLT:
        for i, j, k, size in idxs:
          StoragePtr.cmplt(self._data, other._data, out._data, i, j, k, size)
        out._backward = None
      case _:
        raise RuntimeError(f"invalid binary op {op}")
    return out

  def add(self, other: Union[Tensor, float, int], reverse: bool = False) -> Tensor:
    return self._binary_op(other, reverse, BinaryOps.ADD)
  def sub(self, other: Union[Tensor, float, int], reverse: bool = False) -> Tensor:
    return self._binary_op(other, reverse, BinaryOps.SUB)
  def mul(self, other: Union[Tensor, float, int], reverse: bool = False) -> Tensor:
    return self._binary_op(other, reverse, BinaryOps.MUL)
  def div(self, other: Union[Tensor, float, int], reverse: bool = False) -> Tensor:
    return self._binary_op(other, reverse, BinaryOps.DIV)

  def __add__(self, other) -> Tensor: return self.add(other)
  def __sub__(self, other) -> Tensor: return self.sub(other)
  def __mul__(self, other) -> Tensor: return self.mul(other)
  def __truediv__(self, other) -> Tensor: return self.div(other)

  def __radd__(self, other) -> Tensor: return self.add(other, True)
  def __rsub__(self, other) -> Tensor: return self.sub(other, True)
  def __rmul__(self, other) -> Tensor: return self.mul(other, True)
  def __rtruediv__(self, other) -> Tensor: return self.div(other, True)

  def __iadd__(self, other) -> Tensor: return self.assign(self.add(other))
  def __isub__(self, other) -> Tensor: return self.assign(self.sub(other))
  def __imul__(self, other) -> Tensor: return self.assign(self.mul(other))
  def __itruediv__(self, other) -> Tensor: return self.assign(self.div(other))

  def lt(self, other: Union[Tensor, float, int], reverse: bool = False) -> Tensor:
    return self._binary_op(other, reverse, BinaryOps.CMPLT)
  def gt(self, other: Union[Tensor, float, int]) -> Tensor:
    return self.lt(other, True)

  def __lt__(self, other) -> Tensor: return self.lt(other)
  def __gt__(self, other) -> Tensor: return self.gt(other)
  # TODO: __ge__, __le__, __eq__, __ne__

  # ********************************************************
  # ***************        reduce ops        ***************
  # ********************************************************

  def _reduce_op(self, *args, op: ReduceOps) -> Tensor:
    out = Tensor._dummy(shape=View.create(()), requires_grad=self.requires_grad, prev=(self,), op=op)
    if op == ReduceOps.SUM:
      dim, keepdim = args
      out._data = self._data.sum(axis=dim, keepdims=keepdim)
      out._grad = np.zeros_like(out._data)
      # out._shape = Shape(out._data.shape) if isinstance(out._data, np.ndarray) else Shape(())

      def backward():
        if dim is None or keepdim:
          self._grad += out._grad
        else:
          o_new_shape = tuple(1 if i == dim else s for i, s in enumerate(self.shape))
          self._grad += out._grad.reshape(o_new_shape)

      out._backward = backward
    elif op == ReduceOps.MAX:
      dim, keepdim = args
      out._data = self._data.max(axis=dim, keepdims=keepdim)
      out._grad = np.zeros_like(out._data)
      # out._shape = Shape(out._data.shape) if isinstance(out._data, np.ndarray) else Shape(())

      def backward():
        if dim is None or keepdim:
          self._grad += (self._data == out._data) * out._grad
        else:
          o_new_shape = tuple(1 if i == dim else s for i, s in enumerate(self.shape))
          self._grad += (self._data == out._data.reshape(o_new_shape)) * out._grad.reshape(o_new_shape)

      out._backward = backward
    return out

  def sum(self, dim: Optional[int] = None, keepdim=False) -> Tensor:
    return self._reduce_op(dim, keepdim, op=ReduceOps.SUM)

  def max(self, dim: Optional[int] = None, keepdim=False) -> Tensor:
    # different return types than pytorch: https://pytorch.org/docs/stable/generated/torch.max.html
    return self._reduce_op(dim, keepdim, op=ReduceOps.MAX)

  def mean(self, dim: Optional[int] = None, keepdim=False):
    out = self.sum(dim=dim, keepdim=keepdim)
    return out.div(prod(self.shape) / prod(out.shape))

  def _softmax(self, dim: Optional[int]) -> Tuple[Tensor, Tensor, Tensor]:
    if len(self.shape) == 0:
      assert dim in [-1, 0], f"invalid dim {dim} for tensor {self}"
      dim = None
    # it makes the softmax more numerically stable
    m = self - self.max(dim=dim, keepdim=True)
    e = m.exp()
    return m, e, e.sum(dim=dim, keepdim=True)

  def softmax(self, dim: int = -1) -> Tensor:
    _, e, ss = self._softmax(dim)
    return e.div(ss)

  def log_softmax(self, dim: int = -1) -> Tensor:
    # x.log_softmax()
    # = x.softmax().log()
    # = log(exp(x_i) / sum(exp(x_k)))
    # = x_i - log(sum(exp(x_k)))
    m, _, ss = self._softmax(dim)
    return m - ss.log()

  # ********************************************************
  # ***************         meta ops         ***************
  # ********************************************************

  def _meta_op(self, view: View, op: MetaOps) -> Tensor:
    match op:
      case MetaOps.RESHAPE:
        out = Tensor(data=self._data, view=view, requires_grad=self._requires_grad, prev=(self,), op=op, device=self._data.device)
        out._backward = None
      case MetaOps.EXPAND:
        out = Tensor(
          data=self._data,
          view=view,
          requires_grad=self._requires_grad,
          prev=(self,),
          op=op,
          device=self._data.device,
        )
        out._backward = None
      case MetaOps.PERMUTE:
        out = Tensor(data=self._data, view=view, requires_grad=self._requires_grad, prev=(self,), op=op, device=self._data.device)
        out._backward = None
      case _: raise RuntimeError
    return out

  def reshape(self, *shape: int) -> Tensor:
    return self._meta_op(shape, op=MetaOps.RESHAPE)

  def expand(self, *shape: int) -> Tensor:
    # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
    # assert len(shape) >= len(
    #     self.shape
    # ), f"the expanded shape {shape} must have at least as many dimensions as the original shape {self.shape}"
    # assert all(
    #     s == 1 or s == e for s, e in zip(self.shape, shape[-len(self.shape) :])
    # ), "the expanded shape must be compatible with the original shape"
    return self._meta_op(shape, op=MetaOps.EXPAND)

  def permute(self, *dims: int) -> Tensor:
    # assert set(dims) == set(range(len(self.shape))), "invalid permutation {dims}"
    return self._meta_op(
      tuple(self.shape[dims[i]] for i in range(len(dims))),
      dims,
      op=MetaOps.PERMUTE,
    )

  def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    # if end_dim == -1:
    #     end_dim = len(self.shape)
    # assert 0 <= start_dim < end_dim <= len(self.shape), "invalid start_dim or end_dim"
    return self.reshape(*(self.shape[:start_dim] + (prod(self.shape[start_dim:end_dim]),) + self.shape[end_dim:]))

  def transpose(self, dim1: int, dim2: int) -> Tensor:
    dims = list(range(len(self.shape)))
    dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
    return self.permute(*dims)

  # ********************************************************
  # ***************      processing ops      ***************
  # ********************************************************

  def matmul_2d(self, other: Tensor) -> Tensor:
    assert len(self.shape) == 2 and len(other.shape) == 2, "matmul_2d only supports 2D tensors, but got shapes {self.shape} and {other.shape}"
    assert self.shape[1] == other.shape[0], f"matmul_2d shape mismatch: {self.shape} and {other.shape}"
    N, M, K = self.shape[0], self.shape[1], other.shape[1]
    return (self.reshape(N, 1, M) * other.permute(1, 0).reshape(1, K, M)).sum(dim=2)

  def matmul(self, other: Tensor) -> Tensor:
    # https://pytorch.org/docs/stable/generated/torch.matmul.html
    # if both tensors are 1-dimensional, the dot product (scalar) is returned
    if len(self.shape) == 1 and len(other.shape) == 1:
      assert self.shape[0] == other.shape[0], f"matmul shape mismatch: {self.shape} and {other.shape}"
      return self.mul(other).sum()
    # if both arguments are 2-dimensional, the matrix-matrix product is returned
    elif len(self.shape) == 2 and len(other.shape) == 2:
      return self.matmul_2d(other)
    # if the first argument is 1-dimensional and the second argument is 2-dimensional,
    # a 1 is prepended to its dimension for the purpose of the matrix multiply
    # after the matrix multiply, the prepended dimension is removed
    elif len(self.shape) == 1 and len(other.shape) == 2:
      return self.reshape(1, *self.shape).matmul_2d(other)
    # if the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned
    elif len(self.shape) == 2 and len(other.shape) == 1:
      return self.matmul_2d(other.reshape(*other.shape, 1)).reshape(-1)
    # if both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2),
    # then a batched matrix multiply is returned
    # if the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after
    # if the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after
    # the non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable)
    elif len(self.shape) >= 1 and len(other.shape) >= 1 and (len(self.shape) > 2 or len(other.shape) > 2):
      raise NotImplementedError("batched matrix multiply is not implemented yet: {self.shape} and {other.shape}")
    else:
      raise RuntimeError(f"Invalid matmul shapes {self.shape} and {other.shape}")

  def dot(self, other: Tensor) -> Tensor:
    return self.matmul(other)

  def __matmul__(self, other: Tensor) -> Tensor:
    return self.matmul(other)

  # ********************************************************
  # ***************     functional nn ops    ***************
  # ********************************************************

  def linear(self, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    x = self.matmul(weight)
    return x.add(bias) if bias is not None else x

  def sparse_categorical_crossentropy(self, Y: Tensor) -> Tensor:
    assert (
      len(self.shape) == 2 and len(Y.shape) == 1
    ), f"sparse_categorical_crossentropy only supports 2D tensor and 1D tensor, but got shapes {self.shape} and {Y.shape}"
    assert self.shape[0] == Y.shape[0], f"sparse_categorical_crossentropy shape mismatch: {self.shape} and {Y.shape}"

    Y_pred = self.log_softmax()
    # TODO: need more efficient implementation. currently, it's not possible to use Y as a tensor of indices
    Y_onehot_data = np.zeros_like(Y_pred._data)
    Y_onehot_data[np.arange(prod(Y._data.shape)), (Y._data + 1e-5).astype(np.int32)] = 1
    Y_onehot = Tensor(Y_onehot_data)
    return -(Y_onehot * Y_pred).sum() / prod(Y._data.shape)  # reduction="mean"

  # ********************************************************
  # ***************          random          ***************
  # ********************************************************

  _seed: int = int(time.time())

  @staticmethod
  def manual_seed(seed: int = 0):
    Tensor._seed = seed

  @staticmethod
  def _nxt_seed() -> int:
    Tensor._seed = (Tensor._seed * 1103515245 + 12345) & 0x7FFFFFFF
    return Tensor._seed

  @staticmethod
  def randn(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    np.random.seed(Tensor._nxt_seed())
    return Tensor(
      data=np.random.randn(*shape),
      shape=shape,
      requires_grad=requires_grad,
      prev=None,
      op=None,
    )

  @staticmethod
  def uniform(shape: Union[Tuple[int, ...], int], low=0.0, high=1.0) -> Tensor:
    if isinstance(shape, int):
      shape = (shape,)
    np.random.seed(Tensor._nxt_seed())
    return Tensor(
      data=np.random.uniform(low, high, prod(shape)).reshape(shape),
      shape=shape,
      requires_grad=True,
      prev=None,
      op=None,
    )

  @staticmethod
  def kaiming_uniform(shape: Union[Tuple[int, ...], int], a: float = 0.01) -> Tensor:
    if isinstance(shape, int):
      shape = (shape,)
    bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a**2)) / math.sqrt(prod(shape[1:]))
    return Tensor.uniform(shape, low=-bound, high=bound)

  # ********************************************************
  # ***************      helper functions    ***************
  # ********************************************************

  @staticmethod
  def zeros(shape: Tuple[int], device: str = "cpu", requires_grad: bool = False) -> Tensor:
    return Tensor(
      data=np.zeros(shape),
      view=View.create(shape),
      requires_grad=requires_grad,
      prev=None,
      op=None,
      device=device,
    )

  @staticmethod
  def ones(shape: Tuple[int], device: str = "cpu", requires_grad: bool = False) -> Tensor:
    return Tensor(
      data=np.ones(shape),
      view=View.create(shape),
      requires_grad=requires_grad,
      prev=None,
      op=None,
      device=device,
    )

  def detach(self) -> Tensor:
    return Tensor(
      data=self._data,
      view=self._view,
      requires_grad=False,
      prev=None,
      op=None,
      device=self._data.device,
    )

  def numpy(self) -> np.ndarray: return np.array(self._data.to_vec()).reshape(self._view.shape)
  @property
  def data(self) -> np.ndarray: return self.numpy()
  @property
  def grad(self) -> np.ndarray: return self._grad.numpy()
  @property
  def shape(self) -> Tuple[int, ...]: return self._view.shape
  @property
  def requires_grad(self) -> bool: return self._requires_grad
  @property
  def op(self): return self._op

  def __hash__(self): return id(self)

  def __repr__(self):
    out = f"Tensor({self.numpy().round(4) if self._shape != () else self.item()}"
    if self._op is not None:
      out += f", op={self._op.__repr__()}"
    return out + ")"

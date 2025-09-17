from __future__ import annotations
import math
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from .cranberry import StorageView
from cranberry.ops import BinaryOps, MovementOps, Op, ReduceOps, UnaryOps
from cranberry.shape import Shape


def shape_for_list(x: List) -> Tuple[int, ...]:
  shape: List[int] = []
  current = x
  while isinstance(current, list):
    shape.append(len(current))
    if len(current) == 0:
      break
    current = current[0]
  return tuple(shape)


def _flatten_list(x: List) -> List[float]:
  flattened: List[float] = []

  def _recurse(value):
    if isinstance(value, list):
      for item in value:
        _recurse(item)
    else:
      flattened.append(float(value))

  _recurse(x)
  return flattened


def _ensure_shape_tuple(shape: Optional[Union[Tuple[int, ...], Shape]]) -> Optional[Tuple[int, ...]]:
  if shape is None:
    return None
  if isinstance(shape, Shape):
    return tuple(shape.dims)
  return tuple(int(dim) for dim in shape)


def _prod(values: Iterable[int]) -> int:
  result = 1
  for value in values:
    result *= int(value)
  return result


def _shape_to_list(shape: Tuple[int, ...]) -> List[int]:
  return list(shape) if len(shape) > 0 else []


def _storage_shape(storage: StorageView) -> Tuple[int, ...]:
  shape = tuple(storage.shape())
  return shape


def _contiguous(storage: StorageView) -> StorageView:
  return storage.contiguous()


def _coerce_to_storage(
  data: Union["Tensor", StorageView, float, int, List, np.ndarray],
  shape: Optional[Union[Tuple[int, ...], Shape]],
  device: str,
) -> Tuple[StorageView, Tuple[int, ...]]:
  target_shape = _ensure_shape_tuple(shape)

  if isinstance(data, Tensor):
    storage = data._storage
    inferred_shape = data.shape
    if target_shape is not None and target_shape != inferred_shape:
      storage = storage.reshape(_shape_to_list(target_shape))
      inferred_shape = target_shape
    return storage, inferred_shape

  if isinstance(data, StorageView):
    inferred_shape = _storage_shape(data)
    storage = data
    if target_shape is not None and target_shape != inferred_shape:
      storage = storage.reshape(_shape_to_list(target_shape))
      inferred_shape = target_shape
    return storage, inferred_shape

  if isinstance(data, (float, int)):
    value = float(data)
    storage = StorageView.full(value, 1, device)
    if target_shape is None:
      inferred_shape: Tuple[int, ...] = ()
      storage = storage.reshape([])
    else:
      inferred_shape = target_shape
      storage = storage.reshape(_shape_to_list(inferred_shape))
    return storage, inferred_shape

  if isinstance(data, list):
    inferred = shape_for_list(data)
    flat = _flatten_list(data)
    storage = StorageView.from_vec(flat, device)
    if target_shape is None:
      target_shape = inferred
    if _prod(inferred) != _prod(target_shape):
      raise ValueError("provided shape is incompatible with data")
    storage = storage.reshape(_shape_to_list(target_shape))
    return storage, target_shape

  if isinstance(data, np.ndarray):
    arr = np.asarray(data, dtype=np.float32)
    inferred = tuple(int(dim) for dim in arr.shape)
    flat = arr.reshape(-1).tolist()
    storage = StorageView.from_vec(flat, device)
    if target_shape is None:
      target_shape = inferred
    if _prod(inferred) != _prod(target_shape):
      raise ValueError("provided shape is incompatible with data")
    storage = storage.reshape(_shape_to_list(target_shape))
    return storage, target_shape

  raise ValueError(f"invalid data type {type(data)}")


class Tensor:
  def __init__(
    self,
    data: Union[float, int, List, np.ndarray, StorageView, "Tensor"],
    grad: Optional[Union[List, np.ndarray, StorageView, "Tensor"]] = None,
    shape: Optional[Union[Tuple[int, ...], Shape]] = None,
    requires_grad: bool = False,
    prev: Optional[Tuple[Tensor, ...]] = None,
    op: Optional[Op] = None,
    device: str = "cpu",
  ):
    storage, inferred_shape = _coerce_to_storage(data, shape, device)
    self._storage = storage
    self._shape = inferred_shape
    self._device = device

    self._requires_grad: bool = requires_grad
    self._prev: Optional[Tuple[Tensor, ...]] = prev
    self._op = op
    self._backward = lambda: None

    if grad is not None:
      grad_storage, _ = _coerce_to_storage(grad, inferred_shape, device)
      self._grad_storage: Optional[StorageView] = grad_storage
    else:
      self._grad_storage = StorageView.zeros(_shape_to_list(self.shape), device) if requires_grad else None

  # ********************************************************
  # ***************    internal utilities    ***************
  # ********************************************************

  @classmethod
  def _from_storage(
    cls,
    storage: StorageView,
    *,
    requires_grad: bool,
    prev: Optional[Tuple[Tensor, ...]],
    op: Optional[Op],
    device: str,
  ) -> Tensor:
    obj = object.__new__(cls)
    obj._storage = storage
    obj._shape = _storage_shape(storage)
    obj._device = device
    obj._requires_grad = requires_grad
    obj._prev = prev
    obj._op = op
    obj._backward = lambda: None
    obj._grad_storage = StorageView.zeros(_shape_to_list(obj.shape), device) if requires_grad else None
    return obj

  def _shape_list(self) -> List[int]:
    return _shape_to_list(self._shape)

  def _numel(self) -> int:
    return _prod(self._shape) if len(self._shape) else 1

  def _ensure_grad(self) -> StorageView:
    if self._grad_storage is None:
      self._grad_storage = StorageView.zeros(self._shape_list(), self._device)
    return self._grad_storage

  def _add_grad(self, grad: StorageView):
    if not self._requires_grad:
      return
    grad_buf = self._ensure_grad()
    grad_c = _contiguous(grad)
    self._grad_storage = _contiguous(grad_buf).add(grad_c)

  def _constant_like(self, value: float) -> StorageView:
    storage = StorageView.full(float(value), max(self._numel(), 1), self._device)
    return storage.reshape(self._shape_list())

  # ********************************************************
  # ***************      backward prop       ***************
  # ********************************************************

  def backward(self):
    if not self._requires_grad:
      raise AssertionError("cannot call backward on a tensor that doesn't require gradients")
    if self.shape != ():
      raise AssertionError(f"backward can only be called for scalar tensors, but it has shape {self.shape}")

    topo: List[Tensor] = []
    visited = set()

    def dfs(t: Tensor):
      if t in visited:
        return
      visited.add(t)
      if t._prev is not None:
        for v in t._prev:
          dfs(v)
      topo.append(t)

    dfs(self)
    self._grad_storage = StorageView.ones([], self._device)

    for v in reversed(topo):
      v._backward()

  # ********************************************************
  # ***************         unary ops        ***************
  # ********************************************************

  def _unary_op(self, op: UnaryOps) -> Tensor:
    method = {
      UnaryOps.NEG: "neg",
      UnaryOps.SQRT: "sqrt",
      UnaryOps.RELU: "relu",
      UnaryOps.EXP: "exp",
      UnaryOps.LOG: "log",
    }[op]

    input_c = _contiguous(self._storage)
    storage = getattr(input_c, method)()
    out = Tensor._from_storage(storage, requires_grad=self._requires_grad, prev=(self,), op=op, device=self._device)

    if self._requires_grad:

      def backward():
        out_grad = out._grad_storage
        if out_grad is None:
          return
        out_grad_c = _contiguous(out_grad)
        if op is UnaryOps.NEG:
          self._add_grad(out_grad_c.neg())
        elif op is UnaryOps.SQRT:
          half = out._constant_like(0.5)
          contrib = out_grad_c.mul(half).div(_contiguous(out._storage))
          self._add_grad(contrib)
        elif op is UnaryOps.RELU:
          values = input_c.to_vec()
          mask = [1.0 if v > 0.0 else 0.0 for v in values]
          mask_storage = StorageView.from_vec(mask, self._device).reshape(self._shape_list())
          self._add_grad(out_grad_c.mul(mask_storage))
        elif op is UnaryOps.EXP:
          self._add_grad(out_grad_c.mul(_contiguous(out._storage)))
        elif op is UnaryOps.LOG:
          self._add_grad(out_grad_c.div(input_c))

      out._backward = backward

    return out

  def neg(self) -> Tensor:
    return self._unary_op(UnaryOps.NEG)

  def __neg__(self) -> Tensor:
    return self.neg()

  def sqrt(self) -> Tensor:
    return self._unary_op(UnaryOps.SQRT)

  def relu(self) -> Tensor:
    return self._unary_op(UnaryOps.RELU)

  def exp(self) -> Tensor:
    return self._unary_op(UnaryOps.EXP)

  def log(self) -> Tensor:
    return self._unary_op(UnaryOps.LOG)

  def sigmoid(self) -> Tensor:
    return 1 / (1 + self.neg().exp())

  def tanh(self) -> Tensor:
    return (self.exp() - self.neg().exp()) / (self.exp() + self.neg().exp())

  def gelu(self) -> Tensor:
    return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())

  # ********************************************************
  # ***************        binary ops        ***************
  # ********************************************************

  def _binary_op(self, other: Union[Tensor, int, float], reverse: bool, op: BinaryOps) -> Tensor:
    if not isinstance(other, Tensor):
      other = Tensor(other)

    left, right = self, other
    if reverse:
      left, right = right, left

    left_b, right_b = left._broadcasted(right)

    method = {
      BinaryOps.ADD: "add",
      BinaryOps.SUB: "sub",
      BinaryOps.MUL: "mul",
      BinaryOps.DIV: "div",
    }[op]

    left_c = _contiguous(left_b._storage)
    right_c = _contiguous(right_b._storage)
    storage = getattr(left_c, method)(right_c)
    requires_grad = left_b._requires_grad or right_b._requires_grad
    out = Tensor._from_storage(storage, requires_grad=requires_grad, prev=(left_b, right_b), op=op, device=self._device)

    if out._requires_grad:

      def backward():
        out_grad = out._grad_storage
        if out_grad is None:
          return
        if left_b._requires_grad:
          if op in (BinaryOps.ADD, BinaryOps.SUB):
            left_b._add_grad(_contiguous(out_grad))
          elif op is BinaryOps.MUL:
            left_b._add_grad(_contiguous(out_grad).mul(right_c))
          elif op is BinaryOps.DIV:
            left_b._add_grad(_contiguous(out_grad).div(right_c))

        if right_b._requires_grad:
          if op is BinaryOps.ADD:
            right_b._add_grad(_contiguous(out_grad))
          elif op is BinaryOps.SUB:
            right_b._add_grad(_contiguous(out_grad).neg())
          elif op is BinaryOps.MUL:
            right_b._add_grad(_contiguous(out_grad).mul(left_c))
          elif op is BinaryOps.DIV:
            num = _contiguous(out_grad).mul(left_c)
            denom = right_c.mul(right_c)
            right_b._add_grad(num.div(denom).neg())

      out._backward = backward

    return out

  def add(self, other: Union[Tensor, int, float], reverse: bool = False) -> Tensor:
    return self._binary_op(other, reverse, BinaryOps.ADD)

  def sub(self, other: Union[Tensor, int, float], reverse: bool = False) -> Tensor:
    return self._binary_op(other, reverse, BinaryOps.SUB)

  def mul(self, other: Union[Tensor, int, float], reverse: bool = False) -> Tensor:
    return self._binary_op(other, reverse, BinaryOps.MUL)

  def div(self, other: Union[Tensor, int, float], reverse: bool = False) -> Tensor:
    return self._binary_op(other, reverse, BinaryOps.DIV)

  def __add__(self, other) -> Tensor:
    return self.add(other)

  def __sub__(self, other) -> Tensor:
    return self.sub(other)

  def __mul__(self, other) -> Tensor:
    return self.mul(other)

  def __truediv__(self, other) -> Tensor:
    return self.div(other)

  def __radd__(self, x) -> Tensor:
    return self.add(x, True)

  def __rsub__(self, x) -> Tensor:
    return self.sub(x, True)

  def __rmul__(self, other) -> Tensor:
    return self.mul(other, True)

  def __rtruediv__(self, other) -> Tensor:
    return self.div(other, True)

  # ********************************************************
  # ***************        reduce ops        ***************
  # ********************************************************

  def _normalize_dim(self, dim: Optional[int]) -> Optional[int]:
    if dim is None:
      return None
    if dim < 0:
      dim += len(self.shape)
    if dim < 0 or dim >= len(self.shape):
      raise ValueError("dim out of range")
    return dim

  def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    axis = self._normalize_dim(dim)
    storage = self._storage.sum(axis if axis is not None else None, keepdim)
    out = Tensor._from_storage(storage, requires_grad=self._requires_grad, prev=(self,), op=ReduceOps.SUM, device=self._device)

    if self._requires_grad:

      def backward():
        out_grad = out._grad_storage
        if out_grad is None:
          return
        grad = _contiguous(out_grad)
        dims = list(self.shape)
        if axis is None:
          if not keepdim and dims:
            grad = grad.reshape([1] * len(dims))
          if dims:
            grad = grad.expand(dims)
        else:
          if not keepdim:
            reshape_shape = dims.copy()
            reshape_shape[axis] = 1
            grad = grad.reshape(reshape_shape)
          grad = grad.expand(dims)
        self._add_grad(grad)

      out._backward = backward

    return out

  def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    axis = self._normalize_dim(dim)
    storage = self._storage.max(axis if axis is not None else None, keepdim)
    out = Tensor._from_storage(storage, requires_grad=self._requires_grad, prev=(self,), op=ReduceOps.MAX, device=self._device)

    if self._requires_grad:

      def backward():
        out_grad = out._grad_storage
        if out_grad is None:
          return
        grad = _contiguous(out_grad)
        input_shape = self.shape or (1,)
        input_vals = np.array(_contiguous(self._storage).to_vec(), dtype=np.float32).reshape(input_shape)

        out_shape = out.shape or (1,)
        max_vals = np.array(_contiguous(out._storage).to_vec(), dtype=np.float32).reshape(out_shape)
        grad_vals = np.array(grad.to_vec(), dtype=np.float32).reshape(out_shape)

        if axis is None:
          if not keepdim and len(input_shape) > 0:
            grad_vals = grad_vals.reshape((1,) * len(input_shape))
            max_vals = max_vals.reshape((1,) * len(input_shape))
          grad_broadcast = np.broadcast_to(grad_vals, input_shape)
          max_broadcast = np.broadcast_to(max_vals, input_shape)
        else:
          axis_idx = axis
          if not keepdim:
            grad_vals = np.expand_dims(grad_vals, axis_idx)
            max_vals = np.expand_dims(max_vals, axis_idx)
          grad_broadcast = np.broadcast_to(grad_vals, input_shape)
          max_broadcast = np.broadcast_to(max_vals, input_shape)

        mask = np.isclose(input_vals, max_broadcast, rtol=1e-6, atol=1e-6).astype(np.float32)
        grad_result = (mask * grad_broadcast).astype(np.float32)
        grad_storage = StorageView.from_vec(grad_result.reshape(-1).tolist(), self._device).reshape(self._shape_list())
        self._add_grad(grad_storage)

      out._backward = backward

    return out

  def mean(self, dim: Optional[int] = None, keepdim: bool = False):
    out = self.sum(dim=dim, keepdim=keepdim)
    factor = _prod(self.shape) / _prod(out.shape or (1,))
    return out / factor

  def _softmax(self, dim: Optional[int]) -> Tuple[Tensor, Tensor, Tensor]:
    if len(self.shape) == 0:
      assert dim in [-1, 0], f"invalid dim {dim} for tensor {self}"
      dim = None
    m = self - self.max(dim=dim, keepdim=True)
    e = m.exp()
    return m, e, e.sum(dim=dim, keepdim=True)

  def softmax(self, dim: int = -1) -> Tensor:
    _, e, ss = self._softmax(dim)
    return e.div(ss)

  def log_softmax(self, dim: int = -1) -> Tensor:
    m, _, ss = self._softmax(dim)
    return m - ss.log()

  # ********************************************************
  # ***************       movement ops       ***************
  # ********************************************************

  def reshape(self, *shape: int) -> Tensor:
    shape_tuple = tuple(shape)
    if shape_tuple.count(-1) > 1:
      raise ValueError("can only specify one unknown dimension")
    if any(s == 0 for s in shape_tuple):
      raise ValueError("shape dimensions must be positive")
    total = self._numel()
    if -1 in shape_tuple:
      known = _prod(s for s in shape_tuple if s != -1)
      missing = total // known
      shape_tuple = tuple(missing if s == -1 else s for s in shape_tuple)
    if _prod(shape_tuple) != total:
      raise ValueError(f"cannot reshape tensor of size {total} into shape {shape_tuple}")
    storage = _contiguous(self._storage).reshape(list(shape_tuple))
    out = Tensor._from_storage(storage, requires_grad=self._requires_grad, prev=(self,), op=MovementOps.RESHAPE, device=self._device)

    if self._requires_grad:

      def backward():
        grad = out._grad_storage
        if grad is None:
          return
        self._add_grad(_contiguous(grad).reshape(self._shape_list()))

      out._backward = backward

    return out

  def expand(self, *shape: int) -> Tensor:
    if len(shape) < len(self.shape):
      raise ValueError("expanded shape must have at least the same number of dimensions")
    target = tuple(shape)
    storage = self._storage.expand(list(target))
    out = Tensor._from_storage(storage, requires_grad=self._requires_grad, prev=(self,), op=MovementOps.EXPAND, device=self._device)

    if self._requires_grad:

      def backward():
        grad = out._grad_storage
        if grad is None:
          return
        grad_reduced = _contiguous(grad)
        in_shape = self.shape
        out_shape = target
        padded_in = in_shape
        while len(padded_in) < len(out_shape):
          padded_in = (1,) + padded_in
        axes = [i for i, (s, t) in enumerate(zip(padded_in, out_shape)) if s == 1 and t > 1]
        for axis in axes:
          grad_reduced = grad_reduced.sum(axis, True)
        grad_reduced = grad_reduced.reshape(list(in_shape))
        self._add_grad(grad_reduced)

      out._backward = backward

    return out

  def permute(self, *dims: int) -> Tensor:
    if set(dims) != set(range(len(self.shape))):
      raise ValueError("invalid permutation")
    storage = self._storage.permute(list(dims))
    out = Tensor._from_storage(storage, requires_grad=self._requires_grad, prev=(self,), op=MovementOps.PERMUTE, device=self._device)

    if self._requires_grad:
      inv = [0] * len(dims)
      for i, dim in enumerate(dims):
        inv[dim] = i

      def backward():
        grad = out._grad_storage
        if grad is None:
          return
        self._add_grad(_contiguous(grad).permute(inv))

      out._backward = backward

    return out

  def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    if end_dim == -1:
      end_dim = len(self.shape)
    if not (0 <= start_dim < end_dim <= len(self.shape)):
      raise ValueError("invalid start_dim or end_dim")
    front = self.shape[:start_dim]
    middle = self.shape[start_dim:end_dim]
    back = self.shape[end_dim:]
    new_shape = front + (math.prod(middle),) + back
    return self.reshape(*new_shape)

  def transpose(self, dim1: int, dim2: int) -> Tensor:
    dims = list(range(len(self.shape)))
    dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
    return self.permute(*dims)

  def view(self, *shape: int) -> Tensor:
    return self.reshape(*shape)

  def _broadcasted(self, other: Tensor) -> Tuple[Tensor, Tensor]:
    shape1, shape2 = self.shape, other.shape
    while len(shape1) < len(shape2):
      shape1 = (1,) + shape1
    while len(shape2) < len(shape1):
      shape2 = (1,) + shape2
    target = tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))
    left = self if self.shape == target else self.expand(*target)
    right = other if other.shape == target else other.expand(*target)
    return left, right

  # ********************************************************
  # ***************      processing ops      ***************
  # ********************************************************

  def matmul_2d(self, other: Tensor) -> Tensor:
    if len(self.shape) != 2 or len(other.shape) != 2:
      raise ValueError("matmul_2d only supports 2D tensors")
    if self.shape[1] != other.shape[0]:
      raise ValueError("matmul_2d shape mismatch")
    n, m, k = self.shape[0], self.shape[1], other.shape[1]
    return (self.reshape(n, 1, m) * other.permute(1, 0).reshape(1, k, m)).sum(dim=2)

  def matmul(self, other: Tensor) -> Tensor:
    if len(self.shape) == 1 and len(other.shape) == 1:
      if self.shape[0] != other.shape[0]:
        raise ValueError("matmul shape mismatch")
      return self.mul(other).sum()
    if len(self.shape) == 2 and len(other.shape) == 2:
      return self.matmul_2d(other)
    if len(self.shape) == 1 and len(other.shape) == 2:
      return self.reshape(1, *self.shape).matmul_2d(other)
    if len(self.shape) == 2 and len(other.shape) == 1:
      return self.matmul_2d(other.reshape(other.shape[0], 1)).reshape(self.shape[0])
    raise NotImplementedError("batched matmul not implemented")

  def dot(self, other: Tensor) -> Tensor:
    return self.matmul(other)

  def __matmul__(self, other: Tensor) -> Tensor:
    return self.matmul(other)

  def linear(self, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    x = self.matmul(weight)
    return x.add(bias) if bias is not None else x

  def sparse_categorical_crossentropy(self, Y: Tensor) -> Tensor:
    if not (len(self.shape) == 2 and len(Y.shape) == 1):
      raise ValueError("sparse_categorical_crossentropy requires 2D logits and 1D labels")
    if self.shape[0] != Y.shape[0]:
      raise ValueError("shape mismatch between predictions and labels")
    y_pred = self.log_softmax()
    num_classes = self.shape[1]
    onehot = [[0.0] * num_classes for _ in range(Y.shape[0])]
    labels = Y.numpy().astype(np.int32)
    for i, label in enumerate(labels):
      onehot[i][int(label)] = 1.0
    y_onehot = Tensor(onehot)
    return -(y_onehot * y_pred).sum() / Y.shape[0]

  # ********************************************************
  # ***************          random          ***************
  # ********************************************************

  _seed: int = 1337

  @staticmethod
  def manual_seed(seed: int = 0):
    Tensor._seed = seed

  @staticmethod
  def _nxt_seed() -> int:
    Tensor._seed = (Tensor._seed * 1103515245 + 12345) & 0x7FFFFFFF
    return Tensor._seed

  @staticmethod
  def randn(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    storage = StorageView.randn(list(shape), "cpu", seed=Tensor._nxt_seed())
    return Tensor._from_storage(storage, requires_grad=requires_grad, prev=None, op=None, device="cpu")

  @staticmethod
  def uniform(shape: Union[Tuple[int, ...], int], low=0.0, high=1.0) -> Tensor:
    if isinstance(shape, int):
      shape = (shape,)
    storage = StorageView.uniform(list(shape), float(low), float(high), "cpu", seed=Tensor._nxt_seed())
    return Tensor._from_storage(storage, requires_grad=True, prev=None, op=None, device="cpu")

  @staticmethod
  def kaiming_uniform(shape: Union[Tuple[int, ...], int], a: float = 0.01) -> Tensor:
    if isinstance(shape, int):
      shape = (shape,)
    bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a**2)) / math.sqrt(math.prod(shape[1:]) if len(shape) > 1 else 1.0)
    return Tensor.uniform(shape, low=-bound, high=bound)

  # ********************************************************
  # ***************      helper functions    ***************
  # ********************************************************

  @staticmethod
  def _dummy(shape: Tuple[int, ...], requires_grad: bool, prev: Optional[Tuple[Tensor, ...]], op: Op) -> Tensor:
    storage = StorageView.zeros(list(shape), "cpu")
    return Tensor._from_storage(storage, requires_grad=requires_grad, prev=prev, op=op, device="cpu")

  @staticmethod
  def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    storage = StorageView.zeros(list(shape), "cpu")
    return Tensor._from_storage(storage, requires_grad=requires_grad, prev=None, op=None, device="cpu")

  @staticmethod
  def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    storage = StorageView.ones(list(shape), "cpu")
    return Tensor._from_storage(storage, requires_grad=requires_grad, prev=None, op=None, device="cpu")

  def detach(self) -> Tensor:
    return Tensor._from_storage(self._storage, requires_grad=False, prev=None, op=None, device=self._device)

  def numpy(self) -> np.ndarray:
    contig = _contiguous(self._storage)
    arr = np.array(contig.to_vec(), dtype=np.float32)
    return arr.reshape(self.shape) if self.shape else arr.reshape(())

  @property
  def data(self) -> np.ndarray:
    return self.numpy()

  @property
  def grad(self) -> np.ndarray:
    grad_storage = self._ensure_grad()
    contig = _contiguous(grad_storage)
    arr = np.array(contig.to_vec(), dtype=np.float32)
    return arr.reshape(self.shape) if self.shape else arr.reshape(())

  @property
  def shape(self) -> Tuple[int, ...]:
    return self._shape

  @property
  def requires_grad(self) -> bool:
    return self._requires_grad

  @property
  def device(self) -> str:
    return self._device

  def zero_grad(self):
    if self._requires_grad:
      self._grad_storage = StorageView.zeros(self._shape_list(), self._device)
    else:
      self._grad_storage = None

  def grad_storage(self) -> Optional[StorageView]:
    return self._grad_storage

  def data_storage(self) -> StorageView:
    return self._storage

  def set_data_storage(self, storage: StorageView):
    self._storage = storage

  def num_elements(self) -> int:
    return self._numel()

  @property
  def op(self):
    return self._op

  def size(self, dim: Optional[int] = None):
    return self.shape if dim is None else self.shape[dim]

  def item(self) -> float:
    if self.shape != ():
      raise ValueError("item() only supports tensors with a single element")
    return float(self.numpy().item())

  def __hash__(self):
    return id(self)

  def __repr__(self):
    out = f"Tensor({self.numpy().round(4) if self.shape != () else self.item()}"
    if self._op is not None:
      out += f", op={self._op.__repr__()}"
    return out + ")"

from __future__ import annotations
import math
from cranberry.ops import Op, UnaryOps, BinaryOps, ReduceOps, MovementOps
import time
from typing import List, Optional, Tuple, Union
import numpy as np
from math import prod
from cranberry.shape import Shape


def flatten_list(list: List) -> List:
    flat = []
    for item in list:
        if isinstance(item, List):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


def shape_for_list(list: List) -> Shape:
    shape = []
    while isinstance(list, List):
        shape.append(len(list))
        list = list[0]
    return Shape(tuple(shape))


class Tensor:
    def __init__(
        self,
        data: Union[float, int, List, np.ndarray, np.float32],
        grad: Optional[np.ndarray] = None,
        # TODO: remove shape, and replace with View
        shape: Optional[Union[Tuple, Shape]] = None,
        requires_grad: bool = False,
        prev: Optional[Tuple[Tensor, ...]] = None,
        op: Optional[Op] = None,
    ):  # TODO: dtype, device
        # self._data
        if isinstance(data, (float, int)):
            self._data = np.array(data, dtype=np.float32)
        elif isinstance(data, List):
            self._data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        elif type(data) is np.float32:
            self._data = np.array(data, dtype=np.float32)
        else:
            raise ValueError(f"Invalid data type {type(data)}")

        # self._grad
        # TODO: if requires_grad is False, then grad should be None
        self._grad: np.ndarray = grad if grad is not None else np.zeros_like(self._data)

        # self._shape
        if shape is not None:
            if isinstance(shape, Shape):
                self._shape = shape
            elif isinstance(shape, Tuple):
                self._shape = Shape(shape)
        elif isinstance(data, (float, int)):
            self._shape = Shape(())
        elif isinstance(data, List):
            self._shape = shape_for_list(data)
        elif isinstance(data, np.ndarray):
            self._shape = Shape(data.shape)
        elif type(data) is np.float32:
            self._shape = Shape(data.shape)
        else:
            raise ValueError(f"Invalid data type {type(data)}")

        assert self._shape == Shape(
            self._data.shape
        ), f"shape {self._shape} must match data shape {self._data.shape}"
        assert self._shape == Shape(
            self._grad.shape
        ), f"shape {self._shape} must match grad shape {self._grad.shape}"

        # self._requires_grad
        self._requires_grad: bool = requires_grad
        # self._backward
        self._backward = lambda: None
        # self._prev
        self._prev: Optional[Tuple[Tensor, ...]] = prev
        # self._op
        self._op = op

    # ********************************************************
    # ***************      backward prop       ***************
    # ********************************************************

    def backward(self):
        assert (
            self._requires_grad
        ), "cannot call backward on a tensor that doesn't require gradients"
        assert (
            self.shape == tuple()
        ), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

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
        shape1, shape2 = self.shape, other.shape
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
        out = Tensor._dummy(
            shape=self._shape, requires_grad=self.requires_grad, prev=(self,), op=op
        )
        if op == UnaryOps.NEG:
            out._data -= self._data

            def backward():
                self._grad -= out._grad

            out._backward = backward
        elif op == UnaryOps.SQRT:
            out._data = np.sqrt(self._data)

            def backward():
                self._grad += 0.5 * out._grad / out._data

            out._backward = backward
        elif op == UnaryOps.RELU:
            out._data = self._data * (self._data > 0)

            def backward():
                self._grad += out._grad * (self._data > 0)

            out._backward = backward
        elif op == UnaryOps.EXP:
            out._data = np.exp(self._data)

            def backward():
                self._grad += out._grad * out._data

            out._backward = backward
        elif op == UnaryOps.LOG:
            out._data = np.log(self._data)

            def backward():
                self._grad += out._grad / self._data

            out._backward = backward
        else:
            raise RuntimeError(f"Invalid unary op {op}")
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

    # 1 / (1 + exp(-x)) or exp(x) / (1 + exp(x))
    def sigmoid(self) -> Tensor:
        return 1 / (1 + self.neg().exp())

    def tanh(self) -> Tensor:
        return (self.exp() - self.neg().exp()) / (self.exp() + self.neg().exp())

    # TODO: gelu sanity check

    def gelu(self) -> Tensor:
        return (
            0.5
            * self
            * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
        )

    # ********************************************************
    # ***************        binary ops        ***************
    # ********************************************************

    def _binary_op(
        self, other: Union[Tensor, int, float], reverse: bool, op: BinaryOps
    ) -> Tensor:
        if isinstance(other, (int, float)):
            other = Tensor(other)
        self, other = self._broadcasted(other)
        if reverse:
            self, other = other, self

        out = Tensor._dummy(
            shape=self._shape,
            requires_grad=self.requires_grad or other.requires_grad,
            prev=(self, other),
            op=op,
        )
        if op == BinaryOps.ADD:
            out._data = self._data + other._data

            def backward():
                self._grad += out._grad
                other._grad += out._grad

            out._backward = backward
        elif op == BinaryOps.SUB:
            out._data = self._data - other._data

            def backward():
                self._grad += out._grad
                other._grad -= out._grad

            out._backward = backward
        elif op == BinaryOps.MUL:
            out._data = self._data * other._data

            def backward():
                self._grad += out._grad * other._data
                other._grad += out._grad * self._data

            out._backward = backward
        elif op == BinaryOps.DIV:
            out._data = self._data / other._data

            def backward():
                self._grad += out._grad / other._data
                other._grad -= out._grad * self._data / other._data**2

            out._backward = backward
        else:
            raise RuntimeError(f"Invalid binary op {op}")
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

    def __radd__(self, other) -> Tensor:
        return self.add(other, True)

    def __rsub__(self, other) -> Tensor:
        return self.sub(other, True)

    def __rmul__(self, other) -> Tensor:
        return self.mul(other, True)

    def __rtruediv__(self, other) -> Tensor:
        return self.div(other, True)

    # TODO: __lt__, __gt__, __ge__, __le__, __eq__, __ne__

    # ********************************************************
    # ***************        reduce ops        ***************
    # ********************************************************

    def _reduce_op(self, *args, op: ReduceOps) -> Tensor:
        out = Tensor._dummy(
            shape=Shape(()), requires_grad=self.requires_grad, prev=(self,), op=op
        )
        if op == ReduceOps.SUM:
            dim, keepdim = args
            out._data = self._data.sum(axis=dim, keepdims=keepdim)
            out._grad = np.zeros_like(out._data)
            out._shape = Shape(out._data.shape) if isinstance(out._data, np.ndarray) else Shape(())

            def backward():
                if dim is None or keepdim:
                    self._grad += out._grad
                else:
                    o_new_shape = tuple(
                        1 if i == dim else s for i, s in enumerate(self.shape)
                    )
                    self._grad += out._grad.reshape(o_new_shape)

            out._backward = backward
        elif op == ReduceOps.MAX:
            dim, keepdim = args
            out._data = self._data.max(axis=dim, keepdims=keepdim)
            out._grad = np.zeros_like(out._data)
            out._shape = Shape(out._data.shape) if isinstance(out._data, np.ndarray) else Shape(())

            def backward():
                if dim is None or keepdim:
                    self._grad += (self._data == out._data) * out._grad
                else:
                    o_new_shape = tuple(
                        1 if i == dim else s for i, s in enumerate(self.shape)
                    )
                    self._grad += (
                        self._data == out._data.reshape(o_new_shape)
                    ) * out._grad.reshape(o_new_shape)

            out._backward = backward
        else:
            raise RuntimeError(f"Invalid reduce op {op}")
        return out

    # TODO: add keepdim

    def sum(self, dim: Optional[int] = None, keepdim=False) -> Tensor:
        return self._reduce_op(dim, keepdim, op=ReduceOps.SUM)

    # different return types than pytorch: https://pytorch.org/docs/stable/generated/torch.max.html

    def max(self, dim: Optional[int] = None, keepdim=False) -> Tensor:
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
    # ***************       movement ops       ***************
    # ********************************************************

    def _movement_op(self, *args, op: MovementOps) -> Tensor:
        out = Tensor._dummy(
            shape=Shape(args[0]), requires_grad=self.requires_grad, prev=(self,), op=op
        )
        if op == MovementOps.RESHAPE:
            out._data = self._data.reshape(args[0])
            out._grad = self._grad.reshape(args[0])

            def backward():
                if (
                    out._grad.base is None
                ):  # TODO: initially, out._grad is a view of self._grad, but some reason it becomes a copy
                    self._grad += out._grad.reshape(self.shape)

            out._backward = backward
        elif op == MovementOps.EXPAND:  # can we make this zero-copy?
            out._data = np.copy(np.broadcast_to(self._data, args[0]))
            out._grad = np.copy(np.broadcast_to(self._grad, args[0]))

            def backward():
                s_shape, o_shape = self.shape, out.shape
                while len(s_shape) < len(o_shape):
                    s_shape = (1,) + s_shape
                axis = tuple(i for i in range(len(o_shape)) if s_shape[i] == 1)
                self._grad += out._grad.sum(axis=axis, keepdims=True).reshape(
                    self.shape
                )

            out._backward = backward
        elif op == MovementOps.PERMUTE:
            out._data = self._data.transpose(args[1])
            out._grad = self._grad.transpose(args[1])
        else:
            raise RuntimeError(f"Invalid movement op {op}")
        return out

    def reshape(self, *shape: int) -> Tensor:
        assert shape.count(-1) <= 1, "can only specify one unknown dimension"
        assert all(
            s > 0 or s == -1 for s in shape
        ), "shape dimensions must be positive or -1"

        if shape.count(-1) == 1:
            assert (
                prod(self._shape) % -prod(shape) == 0
            ), f"cannot reshape tensor of size {prod(self._shape)} into shape {shape}"
            shape = tuple(
                s if s != -1 else prod(self._shape) // -prod(shape) for s in shape
            )
        assert prod(shape) == prod(
            self._shape
        ), f"cannot reshape tensor of size {prod(self._shape)} into shape {shape}"

        return self._movement_op(shape, op=MovementOps.RESHAPE)

    # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
    def expand(self, *shape: int) -> Tensor:
        assert (
            len(shape) >= len(self.shape)
        ), f"the expanded shape {shape} must have at least as many dimensions as the original shape {self.shape}"
        assert all(
            s == 1 or s == e for s, e in zip(self.shape, shape[-len(self.shape) :])
        ), "the expanded shape must be compatible with the original shape"
        return self._movement_op(shape, op=MovementOps.EXPAND)

    def permute(self, *dims: int) -> Tensor:
        assert set(dims) == set(range(len(self.shape))), "invalid permutation {dims}"
        return self._movement_op(
            tuple(self.shape[dims[i]] for i in range(len(dims))),
            dims,
            op=MovementOps.PERMUTE,
        )

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        if end_dim == -1:
            end_dim = len(self.shape)
        assert (
            0 <= start_dim < end_dim <= len(self.shape)
        ), "invalid start_dim or end_dim"
        return self.reshape(
            *(
                self.shape[:start_dim]
                + (prod(self.shape[start_dim:end_dim]),)
                + self.shape[end_dim:]
            )
        )

    def transpose(self, dim1: int, dim2: int) -> Tensor:
        dims = list(range(len(self.shape)))
        dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
        return self.permute(*dims)

    def view(self, *shape: int) -> Tensor:
        return self.reshape(*shape)

    # ********************************************************
    # ***************      processing ops      ***************
    # ********************************************************

    def matmul_2d(self, other: Tensor) -> Tensor:
        assert (
            len(self.shape) == 2 and len(other.shape) == 2
        ), "matmul_2d only supports 2D tensors, but got shapes {self.shape} and {other.shape}"
        assert (
            self.shape[1] == other.shape[0]
        ), f"matmul_2d shape mismatch: {self.shape} and {other.shape}"
        N, M, K = self.shape[0], self.shape[1], other.shape[1]
        return (self.reshape(N, 1, M) * other.permute(1, 0).reshape(1, K, M)).sum(dim=2)

    def matmul(self, other: Tensor) -> Tensor:
        # https://pytorch.org/docs/stable/generated/torch.matmul.html
        # if both tensors are 1-dimensional, the dot product (scalar) is returned
        if len(self.shape) == 1 and len(other.shape) == 1:
            assert (
                self.shape[0] == other.shape[0]
            ), f"matmul shape mismatch: {self.shape} and {other.shape}"
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
        # if both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned
        # if the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after
        # if the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after
        # the non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable)
        elif (
            len(self.shape) >= 1
            and len(other.shape) >= 1
            and (len(self.shape) > 2 or len(other.shape) > 2)
        ):
            raise NotImplementedError(
                "batched matrix multiply is not implemented yet: {self.shape} and {other.shape}"
            )
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
        assert (
            self.shape[0] == Y.shape[0]
        ), f"sparse_categorical_crossentropy shape mismatch: {self.shape} and {Y.shape}"

        Y_pred = self.log_softmax()
        # TODO: need more efficient implementation. currently, it's not possible to use Y as a tensor of indices
        Y_onehot_data = np.zeros_like(Y_pred._data)
        Y_onehot_data[
            np.arange(prod(Y._data.shape)), (Y._data + 1e-5).astype(np.int32)
        ] = 1
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
        bound = (
            math.sqrt(3.0) * math.sqrt(2.0 / (1 + a**2)) / math.sqrt(prod(shape[1:]))
        )
        return Tensor.uniform(shape, low=-bound, high=bound)

    # ********************************************************
    # ***************      helper functions    ***************
    # ********************************************************

    @staticmethod
    def _dummy(
        shape: Shape, requires_grad: bool, prev: Optional[Tuple[Tensor, ...]], op: Op
    ) -> Tensor:
        return Tensor(
            data=np.zeros(shape.dims),
            shape=shape,
            requires_grad=requires_grad,
            prev=prev,
            op=op,
        )

    @staticmethod
    def zeros(shape: Tuple[int], requires_grad: bool = False) -> Tensor:
        return Tensor(
            data=np.zeros(shape),
            shape=shape,
            requires_grad=requires_grad,
            prev=None,
            op=None,
        )

    @staticmethod
    def ones(shape: Tuple[int], requires_grad: bool = False) -> Tensor:
        return Tensor(
            data=np.ones(shape),
            shape=shape,
            requires_grad=requires_grad,
            prev=None,
            op=None,
        )

    def detach(self) -> Tensor:
        return Tensor(
            data=self._data, shape=self._shape, requires_grad=False, prev=None, op=None
        )

    def numpy(self) -> np.ndarray:
        return self._data

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def grad(self) -> np.ndarray:
        return self._grad

    @property
    def shape(self) -> Tuple:
        return self._shape.dims

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def op(self):
        return self._op

    def size(self, dim: Optional[int] = None):
        return self.shape if dim is None else self.shape[dim]

    def item(self) -> float:
        assert (
            self._shape == ()
        ), f"item() only supports tensors with a single element, but got shape {self.shape}"
        return self._data.item()

    def __hash__(self):
        return id(self)

    def __repr__(self):
        out = f"Tensor({self.numpy().round(4) if self._shape != () else self.item()}"
        if self._op is not None:
            out += f", op={self._op.__repr__()}"
        return out + ")"

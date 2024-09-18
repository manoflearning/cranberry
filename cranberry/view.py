from __future__ import annotations
from dataclasses import dataclass
from math import prod
from typing import List, Optional, Tuple

MAX_RANK: int = 6


def compute_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
  stride = [1] * len(shape)
  for i in range(len(shape) - 2, -1, -1):
    stride[i] = shape[i + 1] * stride[i + 1]
  return tuple(stride)


@dataclass(frozen=True)
class View:
  shape: Tuple[int, ...]
  stride: Tuple[int, ...]
  offset: int
  contiguous: bool

  @staticmethod
  def create(
    shape: Tuple[int, ...],
    stride: Optional[Tuple[int, ...]] = None,
    offset: int = 0,
    contiguous: bool = True,
  ) -> View:
    # TODO: add more assertions
    assert len(shape) <= MAX_RANK, f"{shape=} can't have more than {MAX_RANK} dimensions"

    if stride is None:
      stride = compute_stride(shape)
    return View(shape, stride, offset, contiguous)

  def reshape(self, shape: Tuple[int, ...]) -> Optional[View]:
    if self.shape == shape: return self

    assert all(x >= 0 for x in shape), f"{shape=} can't contain negative numbers"
    assert prod(self.shape) == prod(shape), f"size mismatched, can't reshape {self.shape=} -> {shape=}"
    assert len(shape) <= MAX_RANK, f"{shape=} can't have more than {MAX_RANK} dimensions"

    if self.contiguous: return View.create(shape)
    else:
      raise NotImplementedError("reshaping non-contiguous views is not supported yet")

  def expand(self, sizes: Tuple[int, ...]) -> View:
    assert all(x >= 0 for x in sizes), f"expand {sizes=} can't contain negative numbers"
    assert len(sizes) <= MAX_RANK, f"expand {sizes=} can't have more than {MAX_RANK} dimensions"
    assert len(sizes) >= len(self.shape), f"expand {sizes=} must have at least {len(self.shape)} dimensions"
    assert (
      (p == 1 and 1 < q) or p == q for p, q in zip(self.shape, sizes[: len(self.shape)])
    ), f"expand {sizes=} must be compatible with {self.shape=}"

    if self.contiguous:
      n_shape = list(sizes)
      n_stride = list(self.stride)

      for i in range(len(sizes)):
        if i < len(self.shape):
          if sizes[i] == self.shape[i]: continue
          if sizes[i] != self.shape[i] and self.shape[i] == 1:
            n_stride[i] = 0 # means this dimension is "broadcasted"
        else: n_stride.append(0)

      return View.create(
        shape=tuple(n_shape),
        stride=tuple(n_stride),
        offset=self.offset,
        contiguous=all(p == q or (p > q and p == 1 and q == 0) for p, q in zip(n_shape, n_stride)),
      )
    else:
      raise NotImplementedError("expanding non-contiguous views is not supported yet")

  def permute(self, dims: Tuple[int, ...]) -> View:
    if dims == tuple(range(len(self.shape))): return self

    assert set(dims) == set(range(len(self.shape))), f"{dims=} must be a permutation of {range(len(self.shape))}"

    if self.contiguous:
      return View.create(
        shape=tuple(self.shape[dim] for dim in dims),
        stride=tuple(self.stride[dim] for dim in dims),
        offset=self.offset,
        contiguous=False,
      )
    else:
      raise NotImplementedError("permuting non-contiguous views is not supported yet")

  @staticmethod
  def unary_indexing(in1: View, out: View) -> List[Tuple[int, int, int]]:
    if in1.contiguous and out.contiguous:
      assert (prod(in1.shape) == prod(out.shape))
      return [[0, 0, prod(in1.shape)]]

    raise NotImplementedError

  @staticmethod
  def binary_indexing(in1: View, in2: View, out: View) -> List[Tuple[int, int, int, int]]:
    if in1.contiguous and in2.contiguous and out.contiguous:
      assert (prod(in1.shape) == prod(in2.shape) and prod[in2.shape] == prod(out.shape))
      return [[0, 0, 0, prod(in1.shape)]]

    raise NotImplementedError

  def __repr__(self):
    return f"View({self.shape=}, {self.stride=}, {self.offset=}, {self.contiguous=})"

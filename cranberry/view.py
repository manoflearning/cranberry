from __future__ import annotations
from dataclasses import dataclass
from math import prod
from typing import Optional, Tuple

MAX_RANK: int = 4


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

    if not self.contiguous:
      raise NotImplementedError("reshaping non-contiguous views is not supported yet")
    return View.create(shape)

    # total_elements = prod(self.shape)
    # new_shape_with_minus_one: List[int] = []
    # inferred_index = None

    # for i, dim in enumerate(shape):
    #     if dim == -1:
    #         if inferred_index is not None:
    #             raise ValueError("Only one dimension can be inferred")
    #         inferred_index = i
    #         new_shape_with_minus_one.append(1)
    #     else:
    #         new_shape_with_minus_one.append(dim)

    # if inferred_index is not None:
    #     inferred_dim = total_elements // prod(new_shape_with_minus_one)
    #     new_shape_with_minus_one[inferred_index] = inferred_dim

    # new_total_elements = prod(new_shape_with_minus_one)

    # if total_elements != new_total_elements:
    #     raise ValueError("Reshape cannot change the total number of elements.")

    # self.shape = tuple(new_shape_with_minus_one)
    # self.stride = compute_stride(self.shape)

  def expand(self, sizes: Tuple[int, ...]) -> View:
    assert all(x >= 0 for x in sizes), f"expand {sizes=} can't contain negative numbers"
    assert len(sizes) <= MAX_RANK, f"expand {sizes=} can't have more than {MAX_RANK} dimensions"
    assert len(sizes) >= len(self.shape), f"expand {sizes=} must have at least {len(self.shape)} dimensions"
    assert (
      (p == 1 and 1 < q) or p == q for p, q in zip(self.shape, sizes[: len(self.shape)])
    ), f"expand {sizes=} must be compatible with {self.shape=}"

    if not self.contiguous:
      raise NotImplementedError("expanding non-contiguous views is not supported yet")

    n_shape = list(sizes)
    n_stride = list(self.stride)

    for i in range(len(sizes)):
      if i < len(self.shape):
        if sizes[i] == self.shape[i]: continue
        if sizes[i] != self.shape[i] and self.shape[i] == 1:
          n_stride[i] = 0  # means this dimension is "broadcasted"
      else:
        n_stride.append(0)

    return View.create(
      shape=tuple(n_shape),
      stride=tuple(n_stride),
      offset=self.offset,
      contiguous=False,
    )

  def permute(self, dims: Tuple[int, ...]) -> View:
    if dims == tuple(range(len(self.shape))): return self

    if not self.contiguous:
      raise NotImplementedError("permuting non-contiguous views is not supported yet")

    assert set(dims) == set(range(len(self.shape))), f"{dims=} must be a permutation of {range(len(self.shape))}"

    return View.create(
      shape=tuple(self.shape[dim] for dim in dims),
      stride=tuple(self.stride[dim] for dim in dims),
      offset=self.offset,
      contiguous=False,
    )

  def __repr__(self):
    return f"View({self.shape=}, {self.stride=}, {self.offset=}, {self.contiguous=})"

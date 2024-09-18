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
    assert all(
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
  def unary_op_indexing(in1: View, out: View) -> List[Tuple[int, int, int]]:
    assert in1.size == out.size
    return [[in1.offset, out.offset, in1.size]]

  @staticmethod
  def binary_op_indexing(in1: View, in2: View, out: View) -> List[Tuple[int, int, int, int]]:
    if in1.contiguous and in2.contiguous and out.contiguous:
      assert (prod(in1.shape) == prod(in2.shape) and prod(in2.shape) == prod(out.shape))
      return [[in1.offset, in2.offset, out.offset, prod(in1.shape)]]

    raise NotImplementedError

    # # Compute the broadcasted shape
    # def broadcast_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    #   result = []
    #   len1 = len(shape1)
    #   len2 = len(shape2)
    #   for i in range(1, max(len1, len2)+1):
    #     dim1 = shape1[-i] if i <= len1 else 1
    #     dim2 = shape2[-i] if i <= len2 else 1
    #     if dim1 == 1 or dim2 == 1 or dim1 == dim2:
    #       result.append(max(dim1, dim2))
    #     else:
    #       raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
    #   return tuple(reversed(result))

    # broadcasted_shape = broadcast_shapes(in1.shape, in2.shape)
    # assert broadcasted_shape == out.shape, f"Output shape {out.shape} does not match broadcasted shape {broadcasted_shape}"

    # # Pad shapes and strides
    # def pad_shape_and_stride(shape: Tuple[int, ...], stride: Tuple[int, ...], full_rank: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    #   pad_len = full_rank - len(shape)
    #   shape_padded = (1,) * pad_len + shape
    #   stride_padded = (0,) * pad_len + stride
    #   return shape_padded, stride_padded

    # full_rank = len(broadcasted_shape)
    # in1_shape_padded, in1_stride_padded = pad_shape_and_stride(in1.shape, in1.stride, full_rank)
    # in2_shape_padded, in2_stride_padded = pad_shape_and_stride(in2.shape, in2.stride, full_rank)
    # out_shape_padded, out_stride_padded = pad_shape_and_stride(out.shape, out.stride, full_rank)

    # # Precompute cumulative products for shape
    # cum_prod = [1] * full_rank
    # for i in range(full_rank - 1, -1, -1):
    #   cum_prod[i] = prod(broadcasted_shape[i + 1:]) if i + 1 < full_rank else 1

    # indices = []
    # total_size = prod(broadcasted_shape)
    # for idx in range(total_size):
    #   idx_remain = idx
    #   in1_offset = in1.offset
    #   in2_offset = in2.offset
    #   out_offset = out.offset
    #   for dim in range(full_rank):
    #     dim_size = broadcasted_shape[dim]
    #     stride1 = in1_stride_padded[dim]
    #     stride2 = in2_stride_padded[dim]
    #     stride_out = out_stride_padded[dim]
    #     dim_idx = idx_remain // cum_prod[dim] if cum_prod[dim] != 0 else 0
    #     idx_remain = idx_remain % cum_prod[dim] if cum_prod[dim] != 0 else 0
    #     idx1 = dim_idx if in1_shape_padded[dim] != 1 else 0
    #     idx2 = dim_idx if in2_shape_padded[dim] != 1 else 0
    #     idx_out = dim_idx
    #     in1_offset += idx1 * stride1
    #     in2_offset += idx2 * stride2
    #     out_offset += idx_out * stride_out
    #   indices.append((in1_offset, in2_offset, out_offset, 1))
    # return indices

  @property
  def size(self): return prod(self.shape) - self.offset

  def __repr__(self):
    return f"View({self.shape=}, {self.stride=}, {self.offset=}, {self.contiguous=})"

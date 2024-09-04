from math import prod
from typing import List, Optional, Tuple

MAX_RANK: int = 4


class View:
    def __init__(
        self,
        shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]] = None,
        offset: int = 0,
    ):
        if len(shape) > MAX_RANK:
            raise ValueError(f"Rank of the tensor cannot exceed {MAX_RANK}.")
        self.shape: Tuple[int, ...] = shape
        self.stride: Tuple[int, ...] = (
            stride if stride is not None else self._compute_stride(shape)
        )
        self.offset: int = offset

    def _compute_stride(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        stride = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            stride[i] = shape[i + 1] * stride[i + 1]
        return tuple(stride)

    def reshape(self, shape: Tuple[int, ...]):
        total_elements = prod(self.shape)
        new_shape_with_minus_one: List[int] = []
        inferred_index = None

        for i, dim in enumerate(shape):
            if dim == -1:
                if inferred_index is not None:
                    raise ValueError("Only one dimension can be inferred")
                inferred_index = i
                new_shape_with_minus_one.append(1)
            else:
                new_shape_with_minus_one.append(dim)

        if inferred_index is not None:
            inferred_dim = total_elements // prod(
                new_shape_with_minus_one
            )
            new_shape_with_minus_one[inferred_index] = inferred_dim

        new_total_elements = prod(new_shape_with_minus_one)

        if total_elements != new_total_elements:
            raise ValueError("Reshape cannot change the total number of elements.")

        self.shape = tuple(new_shape_with_minus_one)
        self.stride = self._compute_stride(self.shape)

    def expand(self, *sizes: int):
        if len(sizes) < len(self.shape):
            raise ValueError(f"Cannot expand to {sizes}, must have at least {len(self.shape)} dimensions.")

        expanded_shape = list(sizes)
        expanded_stride = list(self.stride)

        # Expand dimensions
        for i in range(len(sizes)):
            if i < len(self.shape):
                if sizes[i] == self.shape[i]:
                    # No expansion needed
                    continue
                elif self.shape[i] == 1:
                    # Expand by repeating along this dimension
                    expanded_stride[i] = 0  # Means this dimension is "broadcasted"
                else:
                    raise ValueError(f"Cannot expand dimension {i}, shape {self.shape[i]} to {sizes[i]}")
            else:
                # New dimensions beyond original shape
                expanded_stride.append(0)  # New dimension is broadcasted

        self.shape = tuple(expanded_shape)
        self.stride = tuple(expanded_stride)

    def permute(self, dims: Tuple[int, ...]):
        if len(dims) != len(self.shape):
            raise ValueError(
                f"Shape {self.shape} and permutation {dims} must have the same length."
            )

        self.shape = tuple(self.shape[dim] for dim in dims)
        self.stride = tuple(self.stride[dim] for dim in dims)

    def __repr__(self):
        return f"View(shape={self.shape}, stride={self.stride}, offset={self.offset})"

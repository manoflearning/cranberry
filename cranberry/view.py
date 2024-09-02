from typing import Optional, Tuple

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

    def _calculate_total_elements(self, shape: Tuple[int, ...]) -> int:
        return shape[0] if len(shape) == 1 else shape[0] * self._calculate_total_elements(shape[1:])

    def reshape(self, shape: Tuple[int, ...]):
        total_elements = self._calculate_total_elements(self.shape)
        new_shape_with_minus_one = []
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
            inferred_dim = total_elements // self._calculate_total_elements(
                new_shape_with_minus_one
            )
            new_shape_with_minus_one[inferred_index] = inferred_dim

        new_total_elements = self._calculate_total_elements(new_shape_with_minus_one)

        if total_elements != new_total_elements:
            raise ValueError("Reshape cannot change the total number of elements.")

        self.shape = tuple(new_shape_with_minus_one)
        self.stride = self._compute_stride(self.shape)

    def permute(self, dims: Tuple[int, ...]):
        if len(dims) != len(self.shape):
            raise ValueError(
                "Invalid permutation, dims should match the number of dimensions."
            )

        self.shape = tuple(self.shape[dim] for dim in dims)
        self.stride = tuple(self.stride[dim] for dim in dims)

    def __repr__(self):
        return f"View(shape={self.shape}, stride={self.stride}, offset={self.offset})"

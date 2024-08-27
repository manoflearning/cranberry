class View:
    def __init__(self, shape, stride=None, offset=0):
        self.shape = shape
        self.stride = stride if stride is not None else self._compute_stride(shape)
        self.offset = offset

    def _compute_stride(self, shape):
        stride = []
        for dim in reversed(shape):
            if stride:
                stride.insert(0, stride[0] * dim)
            else:
                stride.insert(0, 1)
        return tuple(stride)

    def _calculate_total_elements(self, shape):
        total = 1
        for dim in shape:
            total *= dim
        return total

    def reshape(self, new_shape):
        total_elements = self._calculate_total_elements(self.shape)
        new_shape_with_minus_one = []
        inferred_index = None

        for i, dim in enumerate(new_shape):
            if dim == -1:
                if inferred_index is not None:
                    raise ValueError("Only one dimension can be inferred")
                inferred_index = i
                new_shape_with_minus_one.append(1)
            else:
                new_shape_with_minus_one.append(dim)

        if inferred_index is not None:
            inferred_dim = total_elements // self._calculate_total_elements(new_shape_with_minus_one)
            new_shape_with_minus_one[inferred_index] = inferred_dim

        new_total_elements = self._calculate_total_elements(new_shape_with_minus_one)

        if total_elements != new_total_elements:
            raise ValueError("Reshape cannot change the total number of elements.")

        self.shape = tuple(new_shape_with_minus_one)
        self.stride = self._compute_stride(self.shape)

    def permute(self, *dims):
        if len(dims) != len(self.shape):
            raise ValueError("Invalid permutation, dims should match the number of dimensions.")

        self.shape = tuple(self.shape[dim] for dim in dims)
        self.stride = tuple(self.stride[dim] for dim in dims)

    def __repr__(self):
        return f"View(shape={self.shape}, stride={self.stride}, offset={self.offset})"


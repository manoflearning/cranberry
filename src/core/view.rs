use std::sync::Arc;

use super::storage::StorageInner;

#[derive(Clone, Debug)]
pub struct View {
    pub(crate) inner: Arc<StorageInner>,
    pub(crate) offset: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
}

impl View {
    pub fn from_inner_1d(inner: Arc<StorageInner>) -> Self {
        let len = inner.len();
        Self {
            inner,
            offset: 0,
            shape: vec![len],
            strides: vec![1],
        }
    }

    pub fn new(
        inner: Arc<StorageInner>,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<isize>,
    ) -> Self {
        debug_assert_eq!(shape.len(), strides.len());
        Self {
            inner,
            offset,
            shape,
            strides,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected: isize = 1;
        for (&dim, &stride) in self.shape.iter().rev().zip(self.strides.iter().rev()) {
            if dim == 1 {
                continue;
            }
            if stride != expected {
                return false;
            }
            expected *= dim as isize;
        }
        true
    }

    pub fn slice_1d(&self, offset: usize, len: usize) -> Self {
        debug_assert_eq!(self.shape.len(), 1, "slice_1d expects a 1D view");
        debug_assert!(offset + len <= self.numel());
        let mut out = self.clone();
        out.offset = self.offset + offset;
        out.shape = vec![len];
        out.strides = vec![1];
        out
    }

    /// Reshape a contiguous view to `new_shape` (row-major). Number of elements must match.
    pub fn reshape_contiguous(&self, new_shape: &[usize]) -> Self {
        assert!(self.is_contiguous(), "reshape requires contiguous view");
        assert_eq!(
            self.numel(),
            new_shape.iter().copied().product::<usize>(),
            "reshape size mismatch"
        );
        let mut stride = 1isize;
        let mut strides = vec![0isize; new_shape.len()];
        for (i, dim) in new_shape.iter().rev().enumerate() {
            let idx = new_shape.len() - 1 - i;
            strides[idx] = stride;
            stride *= *dim as isize;
        }
        Self {
            inner: self.inner.clone(),
            offset: self.offset,
            shape: new_shape.to_vec(),
            strides,
        }
    }

    /// Permute axes by `dims`. Metadata-only; strides/shape are reordered.
    pub fn permute(&self, dims: &[usize]) -> Self {
        assert_eq!(dims.len(), self.shape.len());
        let mut seen = vec![false; dims.len()];
        for &d in dims {
            assert!(d < dims.len() && !seen[d]);
            seen[d] = true;
        }
        let shape = dims.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();
        let strides = dims.iter().map(|&i| self.strides[i]).collect::<Vec<_>>();
        Self {
            inner: self.inner.clone(),
            offset: self.offset,
            shape,
            strides,
        }
    }

    /// Broadcast this view to `new_shape` by setting stride 0 on expanded axes.
    pub fn expand(&self, new_shape: &[usize]) -> Self {
        let mut s_shape = self.shape.clone();
        while s_shape.len() < new_shape.len() {
            s_shape.insert(0, 1);
        }
        let mut s_strides = {
            let mut ss = self.strides.clone();
            while ss.len() < new_shape.len() {
                ss.insert(0, 0);
            }
            ss
        };
        for i in 0..new_shape.len() {
            let need = new_shape[i];
            let have = s_shape[i];
            assert!(have == need || have == 1, "expand incompatible at dim {i}");
            if have == 1 && need > 1 {
                s_strides[i] = 0;
            }
        }
        Self {
            inner: self.inner.clone(),
            offset: self.offset,
            shape: new_shape.to_vec(),
            strides: s_strides,
        }
    }

    // TODO: general strided iteration, overlap/alias checks.
}

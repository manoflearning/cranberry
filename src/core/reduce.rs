use std::sync::Arc;

use crate::core::{storage::StorageInner, view::View};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisError {
    ScalarHasNoDim,
    OutOfRange,
}

pub fn numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

pub fn ensure_contiguous(view: &View) -> View {
    if view.is_contiguous() {
        view.clone()
    } else {
        view.to_contiguous()
    }
}

pub fn normalize_axis(dim: Option<isize>, ndim: usize) -> Result<Option<usize>, AxisError> {
    match dim {
        None => Ok(None),
        Some(axis) => {
            if ndim == 0 {
                return Err(AxisError::ScalarHasNoDim);
            }
            let mut axis = axis;
            if axis < 0 {
                axis += ndim as isize;
            }
            if axis < 0 || axis >= ndim as isize {
                Err(AxisError::OutOfRange)
            } else {
                Ok(Some(axis as usize))
            }
        }
    }
}

pub fn reduce_sum(view: &View, axis: Option<usize>, keepdim: bool) -> View {
    let base = ensure_contiguous(view);
    let device = base.inner.device();
    let shape = base.shape.clone();
    let data = base.inner.as_slice(base.offset, base.numel());

    match axis {
        None => {
            let total: f32 = data.iter().copied().sum();
            let out_shape = if keepdim {
                vec![1; shape.len()]
            } else {
                Vec::new()
            };
            let inner = StorageInner::from_vec(vec![total], device);
            View::from_inner_contiguous(Arc::new(inner), &out_shape)
        }
        Some(axis) => {
            let ndim = shape.len();
            assert!(axis < ndim, "axis out of bounds");
            let axis_size = shape[axis];
            let outer = shape[..axis].iter().product::<usize>();
            let inner = shape[axis + 1..].iter().product::<usize>();
            let mut out_data = vec![0.0f32; outer * inner];
            for outer_idx in 0..outer {
                for inner_idx in 0..inner {
                    let mut acc = 0.0f32;
                    for axis_idx in 0..axis_size {
                        let idx = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                        acc += data[idx];
                    }
                    out_data[outer_idx * inner + inner_idx] = acc;
                }
            }
            let out_shape = if keepdim {
                let mut s = shape.clone();
                s[axis] = 1;
                s
            } else {
                let mut s = shape.clone();
                s.remove(axis);
                s
            };
            let inner = StorageInner::from_vec(out_data, device);
            View::from_inner_contiguous(Arc::new(inner), &out_shape)
        }
    }
}

pub fn reduce_max(view: &View, axis: Option<usize>, keepdim: bool) -> View {
    let base = ensure_contiguous(view);
    let device = base.inner.device();
    let shape = base.shape.clone();
    let data = base.inner.as_slice(base.offset, base.numel());

    match axis {
        None => {
            let mut current = f32::NEG_INFINITY;
            for &v in data.iter() {
                if v > current {
                    current = v;
                }
            }
            let out_shape = if keepdim {
                vec![1; shape.len()]
            } else {
                Vec::new()
            };
            let inner = StorageInner::from_vec(vec![current], device);
            View::from_inner_contiguous(Arc::new(inner), &out_shape)
        }
        Some(axis) => {
            let ndim = shape.len();
            assert!(axis < ndim, "axis out of bounds");
            let axis_size = shape[axis];
            let outer = shape[..axis].iter().product::<usize>();
            let inner = shape[axis + 1..].iter().product::<usize>();
            let mut out_data = vec![f32::NEG_INFINITY; outer * inner];
            for outer_idx in 0..outer {
                for inner_idx in 0..inner {
                    let mut best = f32::NEG_INFINITY;
                    for axis_idx in 0..axis_size {
                        let idx = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                        let val = data[idx];
                        if val > best {
                            best = val;
                        }
                    }
                    out_data[outer_idx * inner + inner_idx] = best;
                }
            }
            let out_shape = if keepdim {
                let mut s = shape.clone();
                s[axis] = 1;
                s
            } else {
                let mut s = shape.clone();
                s.remove(axis);
                s
            };
            let inner = StorageInner::from_vec(out_data, device);
            View::from_inner_contiguous(Arc::new(inner), &out_shape)
        }
    }
}

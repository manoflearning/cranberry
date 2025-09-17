use std::sync::Arc;

use pyo3::prelude::*;

use crate::backend::{registry, Backend, BackendError, BinaryOp, UnaryOp};
use crate::core::{
    reduce::{self, AxisError},
    storage::StorageInner,
    view::View,
};
use crate::device::Device;
use rand::{distributions::Distribution, rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;

fn axis_error_to_py(err: AxisError) -> PyErr {
    match err {
        AxisError::ScalarHasNoDim => {
            pyo3::exceptions::PyValueError::new_err("dim specified for scalar tensor")
        }
        AxisError::OutOfRange => pyo3::exceptions::PyValueError::new_err("dim out of range"),
    }
}

fn backend_err_to_py(err: BackendError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

fn backend_for_device(device: Device) -> Result<&'static dyn Backend, PyErr> {
    match registry::get(device) {
        Ok(backend) => Ok(backend),
        Err(BackendError::UnsupportedDevice(device)) => {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                BackendError::UnsupportedDevice(device).to_string(),
            ))
        }
        Err(err) => Err(backend_err_to_py(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::core::reduce::{ensure_contiguous, normalize_axis, numel, reduce_max, reduce_sum};

    fn reduce_sum_view(view: &View, axis: Option<usize>, keepdim: bool) -> View {
        reduce_sum(view, axis, keepdim)
    }

    fn reduce_max_view(view: &View, axis: Option<usize>, keepdim: bool) -> View {
        reduce_max(view, axis, keepdim)
    }

    fn make_view(shape: &[usize], data: Vec<f32>) -> View {
        let inner = Arc::new(StorageInner::from_vec(data, Device::Cpu));
        View::from_inner_contiguous(inner, shape)
    }

    #[test]
    fn numel_handles_scalar_and_shape() {
        assert_eq!(numel(&[]), 1);
        assert_eq!(numel(&[3, 4, 5]), 60);
    }

    #[test]
    fn ensure_contiguous_clones_or_copies() {
        let base = make_view(&[2, 3], vec![0.0; 6]);
        let cloned = ensure_contiguous(&base);
        assert!(Arc::ptr_eq(&cloned.inner, &base.inner));

        let permuted = base.permute(&[1, 0]);
        let contiguous = ensure_contiguous(&permuted);
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.shape, vec![3, 2]);
    }

    #[test]
    fn normalize_axis_accepts_negative() {
        assert_eq!(normalize_axis(Some(-1), 3).unwrap(), Some(2));
        assert!(normalize_axis(Some(0), 0).is_err());
        assert!(normalize_axis(Some(4), 3).is_err());
    }

    #[test]
    fn reduce_sum_none_matches_total() {
        let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let view = make_view(&[2, 3], data);
        let permuted = view.permute(&[1, 0]);
        let out = reduce_sum_view(&permuted, None, false);
        assert_eq!(out.shape, Vec::<usize>::new());
        let slice = out.inner.as_slice(out.offset, out.numel());
        assert_eq!(slice, &[21.0]);

        let out_keep = reduce_sum_view(&permuted, None, true);
        assert_eq!(out_keep.shape, vec![1, 1]);
        let slice = out_keep.inner.as_slice(out_keep.offset, out_keep.numel());
        assert_eq!(slice, &[21.0]);
    }

    #[test]
    fn reduce_sum_axis_matches_manual() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let view = make_view(&[2, 3, 2], data.clone());
        let out = reduce_sum_view(&view, Some(1), false);
        assert_eq!(out.shape, vec![2, 2]);
        let slice = out.inner.as_slice(out.offset, out.numel());
        let mut expected = Vec::new();
        for i in 0..2 {
            for k in 0..2 {
                let mut acc = 0.0f32;
                for j in 0..3 {
                    let idx = (i * 3 + j) * 2 + k;
                    acc += data[idx];
                }
                expected.push(acc);
            }
        }
        assert_eq!(slice, expected.as_slice());

        let out_keep = reduce_sum_view(&view, Some(1), true);
        assert_eq!(out_keep.shape, vec![2, 1, 2]);
    }

    #[test]
    fn reduce_max_axis_matches_manual() {
        let data: Vec<f32> = vec![1.0, 5.0, -2.0, 3.0, 4.0, 0.5, 7.0, -1.0];
        let view = make_view(&[2, 2, 2], data.clone());
        let out = reduce_max_view(&view, Some(0), false);
        assert_eq!(out.shape, vec![2, 2]);
        let slice = out.inner.as_slice(out.offset, out.numel());
        let mut expected = Vec::new();
        for j in 0..2 {
            for k in 0..2 {
                let mut best = f32::NEG_INFINITY;
                for i in 0..2 {
                    let idx = (i * 2 + j) * 2 + k;
                    best = best.max(data[idx]);
                }
                expected.push(best);
            }
        }
        assert_eq!(slice, expected.as_slice());

        let out_keep = reduce_max_view(&view, Some(0), true);
        assert_eq!(out_keep.shape, vec![1, 2, 2]);
    }
}

#[pyo3::pyclass]
#[derive(Clone)]
pub struct StorageView {
    view: View,
}

#[pyo3::pymethods]
impl StorageView {
    #[staticmethod]
    pub fn from_vec(vec: Vec<f32>, device: &str) -> PyResult<Self> {
        let inner = StorageInner::from_vec(vec, Device::from_str(device));
        Ok(Self {
            view: View::from_inner_1d(Arc::new(inner)),
        })
    }

    #[staticmethod]
    pub fn full(value: f32, size: usize, device: &str) -> PyResult<Self> {
        let inner = StorageInner::new_full(value, size, Device::from_str(device));
        Ok(Self {
            view: View::from_inner_1d(Arc::new(inner)),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, device="cpu"))]
    pub fn zeros(shape: Vec<usize>, device: &str) -> PyResult<Self> {
        let device = Device::from_str(device);
        let inner = StorageInner::new_full(0.0, reduce::numel(&shape), device);
        Ok(Self {
            view: View::from_inner_contiguous(Arc::new(inner), &shape),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, device="cpu"))]
    pub fn ones(shape: Vec<usize>, device: &str) -> PyResult<Self> {
        let device = Device::from_str(device);
        let inner = StorageInner::new_full(1.0, reduce::numel(&shape), device);
        Ok(Self {
            view: View::from_inner_contiguous(Arc::new(inner), &shape),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, device="cpu", seed=None))]
    pub fn randn(shape: Vec<usize>, device: &str, seed: Option<u64>) -> PyResult<Self> {
        let device = Device::from_str(device);
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let total = reduce::numel(&shape);
        let mut data = Vec::with_capacity(total);
        for _ in 0..total {
            let sample: f64 = StandardNormal.sample(&mut rng);
            data.push(sample as f32);
        }
        let inner = StorageInner::from_vec(data, device);
        Ok(Self {
            view: View::from_inner_contiguous(Arc::new(inner), &shape),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, low, high, device="cpu", seed=None))]
    pub fn uniform(
        shape: Vec<usize>,
        low: f32,
        high: f32,
        device: &str,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if low >= high {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "uniform requires low < high",
            ));
        }
        let device = Device::from_str(device);
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let total = reduce::numel(&shape);
        let mut data = Vec::with_capacity(total);
        for _ in 0..total {
            data.push(rng.gen_range(low..high));
        }
        let inner = StorageInner::from_vec(data, device);
        Ok(Self {
            view: View::from_inner_contiguous(Arc::new(inner), &shape),
        })
    }

    pub fn len(&self) -> usize {
        self.view.numel()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.view.shape.clone()
    }

    pub fn to_vec(&self) -> PyResult<Vec<f32>> {
        if !self.view.is_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "to_vec only supports contiguous views for now",
            ));
        }
        Ok(self.view.inner.to_vec(self.view.offset, self.view.numel()))
    }

    pub fn contiguous(&self) -> PyResult<StorageView> {
        Ok(StorageView {
            view: self.view.to_contiguous(),
        })
    }

    pub fn slice(&self, offset: usize, size: usize) -> PyResult<StorageView> {
        if self.view.shape.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "slice(offset, size) is only available for 1D views",
            ));
        }
        Ok(StorageView {
            view: self.view.slice_1d(offset, size),
        })
    }

    pub fn reshape(&self, shape: Vec<usize>) -> PyResult<StorageView> {
        Ok(StorageView {
            view: self.view.reshape_contiguous(&shape),
        })
    }

    pub fn expand(&self, shape: Vec<usize>) -> PyResult<StorageView> {
        Ok(StorageView {
            view: self.view.expand(&shape),
        })
    }

    pub fn permute(&self, dims: Vec<usize>) -> PyResult<StorageView> {
        Ok(StorageView {
            view: self.view.permute(&dims),
        })
    }

    pub fn neg(&self) -> PyResult<StorageView> {
        let backend = backend_for_device(self.view.inner.device())?;
        let out = backend
            .unary(UnaryOp::Neg, &self.view)
            .map_err(backend_err_to_py)?;
        Ok(StorageView { view: out })
    }
    pub fn sqrt(&self) -> PyResult<StorageView> {
        let backend = backend_for_device(self.view.inner.device())?;
        let out = backend
            .unary(UnaryOp::Sqrt, &self.view)
            .map_err(backend_err_to_py)?;
        Ok(StorageView { view: out })
    }
    pub fn relu(&self) -> PyResult<StorageView> {
        let backend = backend_for_device(self.view.inner.device())?;
        let out = backend
            .unary(UnaryOp::Relu, &self.view)
            .map_err(backend_err_to_py)?;
        Ok(StorageView { view: out })
    }
    pub fn exp(&self) -> PyResult<StorageView> {
        let backend = backend_for_device(self.view.inner.device())?;
        let out = backend
            .unary(UnaryOp::Exp, &self.view)
            .map_err(backend_err_to_py)?;
        Ok(StorageView { view: out })
    }
    pub fn log(&self) -> PyResult<StorageView> {
        let backend = backend_for_device(self.view.inner.device())?;
        let out = backend
            .unary(UnaryOp::Log, &self.view)
            .map_err(backend_err_to_py)?;
        Ok(StorageView { view: out })
    }

    pub fn add(&self, other: &StorageView) -> PyResult<StorageView> {
        if self.view.inner.device() != other.view.inner.device() {
            return Err(pyo3::exceptions::PyValueError::new_err("device mismatch"));
        }
        let backend = backend_for_device(self.view.inner.device())?;
        let out = backend
            .binary(BinaryOp::Add, &self.view, &other.view)
            .map_err(backend_err_to_py)?;
        Ok(StorageView { view: out })
    }
    pub fn sub(&self, other: &StorageView) -> PyResult<StorageView> {
        if self.view.inner.device() != other.view.inner.device() {
            return Err(pyo3::exceptions::PyValueError::new_err("device mismatch"));
        }
        let backend = backend_for_device(self.view.inner.device())?;
        let out = backend
            .binary(BinaryOp::Sub, &self.view, &other.view)
            .map_err(backend_err_to_py)?;
        Ok(StorageView { view: out })
    }
    pub fn mul(&self, other: &StorageView) -> PyResult<StorageView> {
        if self.view.inner.device() != other.view.inner.device() {
            return Err(pyo3::exceptions::PyValueError::new_err("device mismatch"));
        }
        let backend = backend_for_device(self.view.inner.device())?;
        let out = backend
            .binary(BinaryOp::Mul, &self.view, &other.view)
            .map_err(backend_err_to_py)?;
        Ok(StorageView { view: out })
    }
    pub fn div(&self, other: &StorageView) -> PyResult<StorageView> {
        if self.view.inner.device() != other.view.inner.device() {
            return Err(pyo3::exceptions::PyValueError::new_err("device mismatch"));
        }
        let backend = backend_for_device(self.view.inner.device())?;
        let out = backend
            .binary(BinaryOp::Div, &self.view, &other.view)
            .map_err(backend_err_to_py)?;
        Ok(StorageView { view: out })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn sum(&self, dim: Option<isize>, keepdim: bool) -> PyResult<StorageView> {
        match self.view.inner.device() {
            crate::device::Device::Cpu => {
                let axis =
                    reduce::normalize_axis(dim, self.view.shape.len()).map_err(axis_error_to_py)?;
                let reduced = reduce::reduce_sum(&self.view, axis, keepdim);
                Ok(StorageView { view: reduced })
            }
            other => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                BackendError::UnsupportedDevice(other).to_string(),
            )),
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn max(&self, dim: Option<isize>, keepdim: bool) -> PyResult<StorageView> {
        match self.view.inner.device() {
            crate::device::Device::Cpu => {
                let axis =
                    reduce::normalize_axis(dim, self.view.shape.len()).map_err(axis_error_to_py)?;
                let reduced = reduce::reduce_max(&self.view, axis, keepdim);
                Ok(StorageView { view: reduced })
            }
            other => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                BackendError::UnsupportedDevice(other).to_string(),
            )),
        }
    }
}

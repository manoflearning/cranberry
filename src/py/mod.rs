use std::sync::Arc;

use pyo3::prelude::*;

use crate::backend::{Backend, BinaryOp, CpuBackend, UnaryOp};
use crate::core::{storage::StorageInner, view::View};
use crate::device::Device;

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

    pub fn len(&self) -> usize {
        self.view.numel()
    }

    pub fn to_vec(&self) -> PyResult<Vec<f32>> {
        if !self.view.is_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "to_vec only supports contiguous views for now",
            ));
        }
        Ok(self.view.inner.to_vec(self.view.offset, self.view.numel()))
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
        match self.view.inner.device() {
            crate::device::Device::Cpu => {
                let backend = CpuBackend;
                let out = backend
                    .unary(UnaryOp::Neg, &self.view)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(StorageView { view: out })
            }
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "unary ops not implemented for this device",
            )),
        }
    }
    pub fn sqrt(&self) -> PyResult<StorageView> {
        match self.view.inner.device() {
            crate::device::Device::Cpu => {
                let backend = CpuBackend;
                let out = backend
                    .unary(UnaryOp::Sqrt, &self.view)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(StorageView { view: out })
            }
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "unary ops not implemented for this device",
            )),
        }
    }
    pub fn exp(&self) -> PyResult<StorageView> {
        match self.view.inner.device() {
            crate::device::Device::Cpu => {
                let backend = CpuBackend;
                let out = backend
                    .unary(UnaryOp::Exp, &self.view)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(StorageView { view: out })
            }
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "unary ops not implemented for this device",
            )),
        }
    }
    pub fn log(&self) -> PyResult<StorageView> {
        match self.view.inner.device() {
            crate::device::Device::Cpu => {
                let backend = CpuBackend;
                let out = backend
                    .unary(UnaryOp::Log, &self.view)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(StorageView { view: out })
            }
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "unary ops not implemented for this device",
            )),
        }
    }

    pub fn add(&self, other: &StorageView) -> PyResult<StorageView> {
        if self.view.inner.device() != other.view.inner.device() {
            return Err(pyo3::exceptions::PyValueError::new_err("device mismatch"));
        }
        match self.view.inner.device() {
            crate::device::Device::Cpu => {
                let backend = CpuBackend;
                let out = backend
                    .binary(BinaryOp::Add, &self.view, &other.view)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(StorageView { view: out })
            }
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "binary ops not implemented for this device",
            )),
        }
    }
    pub fn sub(&self, other: &StorageView) -> PyResult<StorageView> {
        if self.view.inner.device() != other.view.inner.device() {
            return Err(pyo3::exceptions::PyValueError::new_err("device mismatch"));
        }
        match self.view.inner.device() {
            crate::device::Device::Cpu => {
                let backend = CpuBackend;
                let out = backend
                    .binary(BinaryOp::Sub, &self.view, &other.view)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(StorageView { view: out })
            }
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "binary ops not implemented for this device",
            )),
        }
    }
    pub fn mul(&self, other: &StorageView) -> PyResult<StorageView> {
        if self.view.inner.device() != other.view.inner.device() {
            return Err(pyo3::exceptions::PyValueError::new_err("device mismatch"));
        }
        match self.view.inner.device() {
            crate::device::Device::Cpu => {
                let backend = CpuBackend;
                let out = backend
                    .binary(BinaryOp::Mul, &self.view, &other.view)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(StorageView { view: out })
            }
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "binary ops not implemented for this device",
            )),
        }
    }
    pub fn div(&self, other: &StorageView) -> PyResult<StorageView> {
        if self.view.inner.device() != other.view.inner.device() {
            return Err(pyo3::exceptions::PyValueError::new_err("device mismatch"));
        }
        match self.view.inner.device() {
            crate::device::Device::Cpu => {
                let backend = CpuBackend;
                let out = backend
                    .binary(BinaryOp::Div, &self.view, &other.view)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Ok(StorageView { view: out })
            }
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "binary ops not implemented for this device",
            )),
        }
    }
}

use thiserror::Error;

use std::sync::Arc;

use crate::core::storage::StorageInner;
use crate::core::view::{contiguous_strides, View};
use crate::device::Device;

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Sqrt,
    Exp,
    Log,
    Relu,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Error, Clone)]
pub enum BackendError {
    #[error("operation requires contiguous views")]
    NotContiguous,
    #[error("shape mismatch")]
    ShapeMismatch,
    #[error("backend for device {0:?} is not implemented")]
    UnsupportedDevice(Device),
    #[error("cuda runtime error: {0}")]
    Cuda(String),
    #[error("cuda device unavailable: {0}")]
    CudaUnavailable(String),
}

pub type BackendResult<T> = Result<T, BackendError>;

pub trait Backend {
    fn unary(&self, op: UnaryOp, a: &View) -> BackendResult<View>;
    fn binary(&self, op: BinaryOp, a: &View, b: &View) -> BackendResult<View>;
}

fn require_contiguous(view: &View) -> BackendResult<()> {
    if view.is_contiguous() {
        Ok(())
    } else {
        Err(BackendError::NotContiguous)
    }
}

fn require_same_numel(a: &View, b: &View) -> BackendResult<()> {
    if a.numel() == b.numel() {
        Ok(())
    } else {
        Err(BackendError::ShapeMismatch)
    }
}

fn view_from_storage(inner: StorageInner, shape: &[usize]) -> View {
    View {
        inner: Arc::new(inner),
        offset: 0,
        shape: shape.to_vec(),
        strides: contiguous_strides(shape),
    }
}

mod cpu;
mod cuda;

pub mod registry;

pub use cpu::CpuBackend;
pub use cuda::CudaBackend;

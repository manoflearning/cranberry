use thiserror::Error;

use crate::core::view::View;

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Sqrt,
    Exp,
    Log,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("operation requires contiguous views")]
    NotContiguous,
    #[error("shape mismatch")]
    ShapeMismatch,
}

pub type BackendResult<T> = Result<T, BackendError>;

pub trait Backend {
    fn unary(&self, op: UnaryOp, a: &View) -> BackendResult<View>;
    fn binary(&self, op: BinaryOp, a: &View, b: &View) -> BackendResult<View>;
}

// === CPU backend (contiguous-only for now) ===

use std::sync::Arc;

use crate::core::storage::StorageInner;
use crate::device::Device;

mod kernels_simd;
use kernels_simd as kernels;

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn unary(&self, op: UnaryOp, a: &View) -> BackendResult<View> {
        if !a.is_contiguous() {
            return Err(BackendError::NotContiguous);
        }
        let a_slice = a.inner.as_slice(a.offset, a.numel());
        let mut out_inner = StorageInner::new_full(0.0, a.numel(), Device::Cpu);
        {
            let out_slice = out_inner.as_mut_slice(0, a.numel());
            match op {
                UnaryOp::Neg => kernels::unary_ops::neg(a_slice, out_slice),
                UnaryOp::Sqrt => kernels::unary_ops::sqrt(a_slice, out_slice),
                UnaryOp::Exp => kernels::unary_ops::exp(a_slice, out_slice),
                UnaryOp::Log => kernels::unary_ops::log(a_slice, out_slice),
            }
        }
        let mut stride = 1isize;
        let mut strides = vec![0isize; a.shape.len()];
        for (i, dim) in a.shape.iter().rev().enumerate() {
            let idx = a.shape.len() - 1 - i;
            strides[idx] = stride;
            stride *= *dim as isize;
        }
        Ok(View {
            inner: Arc::new(out_inner),
            offset: 0,
            shape: a.shape.clone(),
            strides,
        })
    }

    fn binary(&self, op: BinaryOp, a: &View, b: &View) -> BackendResult<View> {
        if !a.is_contiguous() || !b.is_contiguous() {
            return Err(BackendError::NotContiguous);
        }
        if a.numel() != b.numel() {
            return Err(BackendError::ShapeMismatch);
        }
        let a_slice = a.inner.as_slice(a.offset, a.numel());
        let b_slice = b.inner.as_slice(b.offset, b.numel());
        let mut out_inner = StorageInner::new_full(0.0, a.numel(), Device::Cpu);
        {
            let out_slice = out_inner.as_mut_slice(0, a.numel());
            match op {
                BinaryOp::Add => kernels::binary_ops::add(a_slice, b_slice, out_slice),
                BinaryOp::Sub => kernels::binary_ops::sub(a_slice, b_slice, out_slice),
                BinaryOp::Mul => kernels::binary_ops::mul(a_slice, b_slice, out_slice),
                BinaryOp::Div => kernels::binary_ops::div(a_slice, b_slice, out_slice),
            }
        }
        let mut stride = 1isize;
        let mut strides = vec![0isize; a.shape.len()];
        for (i, dim) in a.shape.iter().rev().enumerate() {
            let idx = a.shape.len() - 1 - i;
            strides[idx] = stride;
            stride *= *dim as isize;
        }
        Ok(View {
            inner: Arc::new(out_inner),
            offset: 0,
            shape: a.shape.clone(),
            strides,
        })
    }
}

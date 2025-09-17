use std::simd::num::SimdFloat;
use std::simd::{f32x64, StdFloat};

use super::{
    require_contiguous, require_same_numel, view_from_storage, Backend, BackendResult, BinaryOp,
    UnaryOp,
};
use crate::core::{storage::StorageInner, view::View};
use crate::device::Device;

const CHUNK_SIZE: usize = 64;

pub struct CpuBackend;

impl CpuBackend {
    fn apply_unary(op: UnaryOp, input: &[f32], output: &mut [f32]) {
        unary::apply(op, input, output);
    }

    fn apply_binary(op: BinaryOp, lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
        binary::apply(op, lhs, rhs, output);
    }
}

impl Backend for CpuBackend {
    fn unary(&self, op: UnaryOp, a: &View) -> BackendResult<View> {
        require_contiguous(a)?;
        let input = a.inner.as_slice(a.offset, a.numel());
        let mut storage = StorageInner::new_full(0.0, a.numel(), Device::Cpu);
        {
            let output = storage.as_mut_slice(0, a.numel());
            Self::apply_unary(op, input, output);
        }
        Ok(view_from_storage(storage, &a.shape))
    }

    fn binary(&self, op: BinaryOp, a: &View, b: &View) -> BackendResult<View> {
        require_contiguous(a)?;
        require_contiguous(b)?;
        require_same_numel(a, b)?;
        let lhs = a.inner.as_slice(a.offset, a.numel());
        let rhs = b.inner.as_slice(b.offset, b.numel());
        let mut storage = StorageInner::new_full(0.0, a.numel(), Device::Cpu);
        {
            let output = storage.as_mut_slice(0, a.numel());
            Self::apply_binary(op, lhs, rhs, output);
        }
        Ok(view_from_storage(storage, &a.shape))
    }
}

pub(crate) mod unary {
    use super::*;
    use std::ops::Neg;

    pub fn apply(op: UnaryOp, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        match op {
            UnaryOp::Neg => neg(input, output),
            UnaryOp::Sqrt => sqrt(input, output),
            UnaryOp::Exp => exp(input, output),
            UnaryOp::Log => log(input, output),
            UnaryOp::Relu => relu(input, output),
        }
    }

    fn neg(a: &[f32], b: &mut [f32]) {
        let (a_main, a_rem) = a.split_at(a.len() - a.len() % CHUNK_SIZE);
        let (b_main, b_rem) = b.split_at_mut(b.len() - b.len() % CHUNK_SIZE);
        a_main
            .chunks_exact(CHUNK_SIZE)
            .zip(b_main.chunks_exact_mut(CHUNK_SIZE))
            .for_each(|(a, b)| f32x64::from_slice(a).neg().copy_to_slice(b));
        a_rem
            .iter()
            .zip(b_rem.iter_mut())
            .for_each(|(a, b)| *b = -a);
    }

    fn sqrt(a: &[f32], b: &mut [f32]) {
        let (a_main, a_rem) = a.split_at(a.len() - a.len() % CHUNK_SIZE);
        let (b_main, b_rem) = b.split_at_mut(b.len() - b.len() % CHUNK_SIZE);
        a_main
            .chunks_exact(CHUNK_SIZE)
            .zip(b_main.chunks_exact_mut(CHUNK_SIZE))
            .for_each(|(a, b)| f32x64::from_slice(a).sqrt().copy_to_slice(b));
        a_rem
            .iter()
            .zip(b_rem.iter_mut())
            .for_each(|(a, b)| *b = a.sqrt());
    }

    fn relu(a: &[f32], b: &mut [f32]) {
        let (a_main, a_rem) = a.split_at(a.len() - a.len() % CHUNK_SIZE);
        let (b_main, b_rem) = b.split_at_mut(b.len() - b.len() % CHUNK_SIZE);
        let zero = f32x64::splat(0.0);
        a_main
            .chunks_exact(CHUNK_SIZE)
            .zip(b_main.chunks_exact_mut(CHUNK_SIZE))
            .for_each(|(a, b)| f32x64::from_slice(a).simd_max(zero).copy_to_slice(b));
        a_rem
            .iter()
            .zip(b_rem.iter_mut())
            .for_each(|(a, b)| *b = a.max(0.0));
    }

    fn exp(a: &[f32], b: &mut [f32]) {
        let (a_main, a_rem) = a.split_at(a.len() - a.len() % CHUNK_SIZE);
        let (b_main, b_rem) = b.split_at_mut(b.len() - b.len() % CHUNK_SIZE);
        a_main
            .chunks_exact(CHUNK_SIZE)
            .zip(b_main.chunks_exact_mut(CHUNK_SIZE))
            .for_each(|(a, b)| f32x64::from_slice(a).exp().copy_to_slice(b));
        a_rem
            .iter()
            .zip(b_rem.iter_mut())
            .for_each(|(a, b)| *b = a.exp());
    }

    fn log(a: &[f32], b: &mut [f32]) {
        let (a_main, a_rem) = a.split_at(a.len() - a.len() % CHUNK_SIZE);
        let (b_main, b_rem) = b.split_at_mut(b.len() - b.len() % CHUNK_SIZE);
        a_main
            .chunks_exact(CHUNK_SIZE)
            .zip(b_main.chunks_exact_mut(CHUNK_SIZE))
            .for_each(|(a, b)| f32x64::from_slice(a).ln().copy_to_slice(b));
        a_rem
            .iter()
            .zip(b_rem.iter_mut())
            .for_each(|(a, b)| *b = a.ln());
    }
}

pub(crate) mod binary {
    use super::*;
    use std::ops::{Add, Div, Mul, Sub};

    pub fn apply(op: BinaryOp, lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(rhs.len(), output.len());
        match op {
            BinaryOp::Add => add(lhs, rhs, output),
            BinaryOp::Sub => sub(lhs, rhs, output),
            BinaryOp::Mul => mul(lhs, rhs, output),
            BinaryOp::Div => div(lhs, rhs, output),
        }
    }

    fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
        let main = a.len() - a.len() % CHUNK_SIZE;
        let (a_main, a_rem) = a.split_at(main);
        let (b_main, b_rem) = b.split_at(main);
        let (c_main, c_rem) = c.split_at_mut(main);
        a_main
            .chunks_exact(CHUNK_SIZE)
            .zip(b_main.chunks_exact(CHUNK_SIZE))
            .zip(c_main.chunks_exact_mut(CHUNK_SIZE))
            .for_each(|((a, b), c)| {
                f32x64::from_slice(a)
                    .add(f32x64::from_slice(b))
                    .copy_to_slice(c)
            });
        a_rem
            .iter()
            .zip(b_rem.iter())
            .zip(c_rem.iter_mut())
            .for_each(|((a, b), c)| *c = a + b);
    }

    fn sub(a: &[f32], b: &[f32], c: &mut [f32]) {
        let main = a.len() - a.len() % CHUNK_SIZE;
        let (a_main, a_rem) = a.split_at(main);
        let (b_main, b_rem) = b.split_at(main);
        let (c_main, c_rem) = c.split_at_mut(main);
        a_main
            .chunks_exact(CHUNK_SIZE)
            .zip(b_main.chunks_exact(CHUNK_SIZE))
            .zip(c_main.chunks_exact_mut(CHUNK_SIZE))
            .for_each(|((a, b), c)| {
                f32x64::from_slice(a)
                    .sub(f32x64::from_slice(b))
                    .copy_to_slice(c)
            });
        a_rem
            .iter()
            .zip(b_rem.iter())
            .zip(c_rem.iter_mut())
            .for_each(|((a, b), c)| *c = a - b);
    }

    fn mul(a: &[f32], b: &[f32], c: &mut [f32]) {
        let main = a.len() - a.len() % CHUNK_SIZE;
        let (a_main, a_rem) = a.split_at(main);
        let (b_main, b_rem) = b.split_at(main);
        let (c_main, c_rem) = c.split_at_mut(main);
        a_main
            .chunks_exact(CHUNK_SIZE)
            .zip(b_main.chunks_exact(CHUNK_SIZE))
            .zip(c_main.chunks_exact_mut(CHUNK_SIZE))
            .for_each(|((a, b), c)| {
                f32x64::from_slice(a)
                    .mul(f32x64::from_slice(b))
                    .copy_to_slice(c)
            });
        a_rem
            .iter()
            .zip(b_rem.iter())
            .zip(c_rem.iter_mut())
            .for_each(|((a, b), c)| *c = a * b);
    }

    fn div(a: &[f32], b: &[f32], c: &mut [f32]) {
        let main = a.len() - a.len() % CHUNK_SIZE;
        let (a_main, a_rem) = a.split_at(main);
        let (b_main, b_rem) = b.split_at(main);
        let (c_main, c_rem) = c.split_at_mut(main);
        a_main
            .chunks_exact(CHUNK_SIZE)
            .zip(b_main.chunks_exact(CHUNK_SIZE))
            .zip(c_main.chunks_exact_mut(CHUNK_SIZE))
            .for_each(|((a, b), c)| {
                f32x64::from_slice(a)
                    .div(f32x64::from_slice(b))
                    .copy_to_slice(c)
            });
        a_rem
            .iter()
            .zip(b_rem.iter())
            .zip(c_rem.iter_mut())
            .for_each(|((a, b), c)| *c = a / b);
    }
}

use std::simd::{f32x64, StdFloat};

const CHUNK_SIZE: usize = 64;

pub mod unary_ops {
    use super::*;
    use std::ops::Neg;

    #[inline(always)]
    pub fn neg(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());
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

    #[inline(always)]
    pub fn sqrt(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());
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

    #[inline(always)]
    pub fn exp(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());
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

    #[inline(always)]
    pub fn log(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());
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

pub mod binary_ops {
    use super::*;
    use std::ops::{Add, Div, Mul, Sub};

    #[inline(always)]
    pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());
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

    #[inline(always)]
    pub fn sub(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());
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

    #[inline(always)]
    pub fn mul(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());
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

    #[inline(always)]
    pub fn div(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());
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

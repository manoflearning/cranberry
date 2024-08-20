use std::simd::{f32x64, StdFloat};

const CHUNK_SIZE: usize = 64;

pub mod unary_ops {
    use std::ops::Neg;

    use super::*;
    #[inline(always)]
    pub fn neg(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(b.array_chunks_mut::<CHUNK_SIZE>())
            .for_each(|(a, b)| {
                a.neg().copy_to_slice(b);
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..].iter().zip(&mut b[remain..]).for_each(|(a, b)| {
            *b = -a;
        });
    }
    #[inline(always)]
    pub fn sqrt(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(b.array_chunks_mut::<CHUNK_SIZE>())
            .for_each(|(a, b)| {
                a.sqrt().copy_to_slice(b);
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..].iter().zip(&mut b[remain..]).for_each(|(a, b)| {
            *b = a.sqrt();
        });
    }
    #[inline(always)]
    pub fn exp(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(b.array_chunks_mut::<CHUNK_SIZE>())
            .for_each(|(a, b)| {
                a.exp().copy_to_slice(b);
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..].iter().zip(&mut b[remain..]).for_each(|(a, b)| {
            *b = a.exp();
        });
    }
    #[inline(always)]
    pub fn log(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(b.array_chunks_mut::<CHUNK_SIZE>())
            .for_each(|(a, b)| {
                a.ln().copy_to_slice(b);
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..].iter().zip(&mut b[remain..]).for_each(|(a, b)| {
            *b = a.ln();
        });
    }
}

pub mod binary_ops {
    use std::ops::{Add, Div, Mul, Sub};

    use super::*;
    #[inline(always)]
    pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_slice(a))
            .zip(
                b.array_chunks::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_slice(b)),
            )
            .zip(c.array_chunks_mut::<CHUNK_SIZE>())
            .for_each(|((a, b), c)| {
                a.add(b).copy_to_slice(c);
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..]
            .iter()
            .zip(&b[remain..])
            .zip(&mut c[remain..])
            .for_each(|((a, b), c)| {
                *c = a + b;
            });
    }
    #[inline(always)]
    pub fn sub(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_slice(a))
            .zip(
                b.array_chunks::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_slice(b)),
            )
            .zip(c.array_chunks_mut::<CHUNK_SIZE>())
            .for_each(|((a, b), c)| {
                a.sub(b).copy_to_slice(c);
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..]
            .iter()
            .zip(&b[remain..])
            .zip(&mut c[remain..])
            .for_each(|((a, b), c)| {
                *c = a - b;
            });
    }
    #[inline(always)]
    pub fn mul(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_slice(a))
            .zip(
                b.array_chunks::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_slice(b)),
            )
            .zip(c.array_chunks_mut::<CHUNK_SIZE>())
            .for_each(|((a, b), c)| {
                a.mul(b).copy_to_slice(c);
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..]
            .iter()
            .zip(&b[remain..])
            .zip(&mut c[remain..])
            .for_each(|((a, b), c)| {
                *c = a * b;
            });
    }
    #[inline(always)]
    pub fn div(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_slice(a))
            .zip(
                b.array_chunks::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_slice(b)),
            )
            .zip(c.array_chunks_mut::<CHUNK_SIZE>())
            .for_each(|((a, b), c)| {
                a.div(b).copy_to_slice(c);
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..]
            .iter()
            .zip(&b[remain..])
            .zip(&mut c[remain..])
            .for_each(|((a, b), c)| {
                *c = a / b;
            });
    }
}

pub mod reduce_ops {
    use super::*;
    use std::{ops::AddAssign, simd::num::SimdFloat};
    #[inline(always)]
    pub fn sum(a: &[f32], b: &mut [f32]) {
        assert!(b.len() == 1);

        let mut acc = f32x64::splat(0.0);
        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .for_each(|a| {
                acc.add_assign(a);
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        b[0] = acc.reduce_sum() + a[remain..].iter().sum::<f32>();
    }
    #[inline(always)]
    pub fn max(a: &[f32], b: &mut [f32]) {
        assert!(b.len() == 1);

        let mut acc = f32x64::splat(f32::NEG_INFINITY);
        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .for_each(|a| {
                acc = acc.simd_max(a);
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        b[0] = a[remain..].iter().copied().fold(acc.reduce_max(), f32::max);
    }
}

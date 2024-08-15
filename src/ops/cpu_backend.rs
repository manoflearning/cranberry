use std::simd::{f32x64, StdFloat};

const CHUNK_SIZE: usize = 64;

pub mod unary_ops {
    use super::*;
    pub fn neg(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(
                b.array_chunks_mut::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_array(*b)),
            )
            .for_each(|(a, mut _b)| {
                _b = -a;
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..].iter().zip(&mut b[remain..]).for_each(|(a, b)| {
            *b = -a;
        });
    }
    pub fn sqrt(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(
                b.array_chunks_mut::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_array(*b)),
            )
            .for_each(|(a, mut _b)| {
                _b = a.sqrt();
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..].iter().zip(&mut b[remain..]).for_each(|(a, b)| {
            *b = a.sqrt();
        });
    }
    pub fn relu(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());

        let zero = f32x64::splat(0.0);
        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(
                b.array_chunks_mut::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_array(*b)),
            )
            .for_each(|(a, mut _b)| {
                _b = if a > zero { a } else { zero };
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..].iter().zip(&mut b[remain..]).for_each(|(a, b)| {
            *b = if *a > 0.0 { *a } else { 0.0 };
        });
    }
    pub fn exp(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(
                b.array_chunks_mut::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_array(*b)),
            )
            .for_each(|(a, mut _b)| {
                _b = a.exp();
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..].iter().zip(&mut b[remain..]).for_each(|(a, b)| {
            *b = a.exp();
        });
    }
    pub fn log(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(
                b.array_chunks_mut::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_array(*b)),
            )
            .for_each(|(a, mut _b)| {
                _b = a.ln();
            });

        let remain = a.len() - a.len() % CHUNK_SIZE;
        a[remain..].iter().zip(&mut b[remain..]).for_each(|(a, b)| {
            *b = a.ln();
        });
    }
}

pub mod binary_ops {
    use super::*;
    pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(
                b.array_chunks::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_array(*b)),
            )
            .zip(
                c.array_chunks_mut::<CHUNK_SIZE>()
                    .map(|c| f32x64::from_array(*c)),
            )
            .for_each(|((a, b), mut _c)| {
                _c = a + b;
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
    pub fn sub(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(
                b.array_chunks::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_array(*b)),
            )
            .zip(
                c.array_chunks_mut::<CHUNK_SIZE>()
                    .map(|c| f32x64::from_array(*c)),
            )
            .for_each(|((a, b), mut _c)| {
                _c = a - b;
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
    pub fn mul(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(
                b.array_chunks::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_array(*b)),
            )
            .zip(
                c.array_chunks_mut::<CHUNK_SIZE>()
                    .map(|c| f32x64::from_array(*c)),
            )
            .for_each(|((a, b), mut _c)| {
                _c = a * b;
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
    pub fn div(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());

        a.array_chunks::<CHUNK_SIZE>()
            .map(|a| f32x64::from_array(*a))
            .zip(
                b.array_chunks::<CHUNK_SIZE>()
                    .map(|b| f32x64::from_array(*b)),
            )
            .zip(
                c.array_chunks_mut::<CHUNK_SIZE>()
                    .map(|c| f32x64::from_array(*c)),
            )
            .for_each(|((a, b), mut _c)| {
                _c = a / b;
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

pub mod unary_ops {
    pub fn neg(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());
        for i in 0..a.len() {
            b[i] = -a[i];
        }
    }
    pub fn sqrt(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());
        for i in 0..a.len() {
            b[i] = a[i].sqrt();
        }
    }
    pub fn relu(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());
        for i in 0..a.len() {
            b[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
        }
    }
    pub fn exp(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());
        for i in 0..a.len() {
            b[i] = a[i].exp();
        }
    }
    pub fn log(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());
        for i in 0..a.len() {
            b[i] = a[i].ln();
        }
    }
}

pub mod binary_ops {
    use std::simd::f32x4;
    pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());

        a.array_chunks::<4>()
            .map(|a| f32x4::from_array(*a))
            .zip(b.array_chunks::<4>().map(|b| f32x4::from_array(*b)))
            .zip(c.array_chunks_mut::<4>().map(|c| f32x4::from_array(*c)))
            .for_each(|((a, b), mut _c)| {
                _c = a + b;
            });

        let remain = a.len() - a.len() % 4;
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

        a.array_chunks::<4>()
            .map(|a| f32x4::from_array(*a))
            .zip(b.array_chunks::<4>().map(|b| f32x4::from_array(*b)))
            .zip(c.array_chunks_mut::<4>().map(|c| f32x4::from_array(*c)))
            .for_each(|((a, b), mut _c)| {
                _c = a - b;
            });

        let remain = a.len() - a.len() % 4;
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

        a.array_chunks::<4>()
            .map(|a| f32x4::from_array(*a))
            .zip(b.array_chunks::<4>().map(|b| f32x4::from_array(*b)))
            .zip(c.array_chunks_mut::<4>().map(|c| f32x4::from_array(*c)))
            .for_each(|((a, b), mut _c)| {
                _c = a * b;
            });

        let remain = a.len() - a.len() % 4;
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

        a.array_chunks::<4>()
            .map(|a| f32x4::from_array(*a))
            .zip(b.array_chunks::<4>().map(|b| f32x4::from_array(*b)))
            .zip(c.array_chunks_mut::<4>().map(|c| f32x4::from_array(*c)))
            .for_each(|((a, b), mut _c)| {
                _c = a / b;
            });

        let remain = a.len() - a.len() % 4;
        a[remain..]
            .iter()
            .zip(&b[remain..])
            .zip(&mut c[remain..])
            .for_each(|((a, b), c)| {
                *c = a / b;
            });
    }
}

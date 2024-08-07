pub mod unary_ops {
    pub fn relu(a: &[f32], b: &mut [f32]) {
        assert!(a.len() == b.len());
        for i in 0..a.len() {
            b[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
        }
    }
}

pub mod binary_ops {
    pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert!(a.len() == b.len() && b.len() == c.len());
        for i in 0..a.len() {
            c[i] = a[i] + b[i];
        }
    }
}

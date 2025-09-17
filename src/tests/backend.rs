use std::sync::Arc;

use crate::backend::{Backend, BinaryOp, CpuBackend, UnaryOp};
use crate::core::{storage::StorageInner, view::View};
use crate::device::Device;

fn vec_view(v: Vec<f32>) -> View {
    let inner = Arc::new(StorageInner::from_vec(v, Device::Cpu));
    View::from_inner_1d(inner)
}

#[test]
fn unary_neg_matches_scalar() {
    let a = vec_view(vec![1.0, -2.0, 3.0, -4.5, 0.25]);
    let be = CpuBackend;
    let out = be.unary(UnaryOp::Neg, &a).unwrap();
    let actual = out.inner.as_slice(out.offset, out.numel());
    let expect: Vec<f32> = a
        .inner
        .as_slice(a.offset, a.numel())
        .iter()
        .map(|x| -*x)
        .collect();
    assert_eq!(actual, expect.as_slice());
}

#[test]
fn unary_sqrt_matches_scalar() {
    let a = vec_view(vec![0.0, 0.25, 1.0, 4.0, 9.0, 16.0, 2.25]);
    let be = CpuBackend;
    let out = be.unary(UnaryOp::Sqrt, &a).unwrap();
    let actual = out.inner.as_slice(out.offset, out.numel());
    let expect: Vec<f32> = a
        .inner
        .as_slice(a.offset, a.numel())
        .iter()
        .map(|x| x.sqrt())
        .collect();
    for (got, exp) in actual.iter().zip(expect.iter()) {
        assert!((got - exp).abs() < 1e-6);
    }
}

#[test]
fn unary_relu_matches_scalar() {
    let a = vec_view(vec![-3.0, -0.5, 0.0, 0.5, 2.0, 5.0]);
    let be = CpuBackend;
    let out = be.unary(UnaryOp::Relu, &a).unwrap();
    let actual = out.inner.as_slice(out.offset, out.numel());
    let expect: Vec<f32> = a
        .inner
        .as_slice(a.offset, a.numel())
        .iter()
        .map(|x| x.max(0.0))
        .collect();
    for (got, exp) in actual.iter().zip(expect.iter()) {
        assert!((got - exp).abs() < 1e-6);
    }
}

#[test]
fn binary_add_matches_scalar() {
    let a = vec_view((0..130).map(|i| i as f32 * 0.5).collect()); // span SIMD + remainder
    let b = vec_view((0..130).map(|i| i as f32 * -0.25).collect());
    let be = CpuBackend;
    let out = be.binary(BinaryOp::Add, &a, &b).unwrap();
    let actual = out.inner.as_slice(out.offset, out.numel());
    let a_s = a.inner.as_slice(a.offset, a.numel());
    let b_s = b.inner.as_slice(b.offset, b.numel());
    for i in 0..a.numel() {
        assert!((actual[i] - (a_s[i] + b_s[i])).abs() < 1e-6);
    }
}

#[test]
fn binary_shape_mismatch_error() {
    let a = vec_view(vec![1.0, 2.0, 3.0]);
    let b = vec_view(vec![4.0, 5.0]);
    let be = CpuBackend;
    let err = be.binary(BinaryOp::Add, &a, &b).unwrap_err();
    assert!(matches!(err, crate::backend::BackendError::ShapeMismatch));
}

#[test]
fn unary_not_contiguous_error() {
    let a = vec_view((0..12).map(|i| i as f32).collect()).reshape_contiguous(&[3, 4]);
    let a_nc = a.permute(&[1, 0]);
    assert!(!a_nc.is_contiguous());
    let be = CpuBackend;
    let err = be.unary(UnaryOp::Neg, &a_nc).unwrap_err();
    assert!(matches!(err, crate::backend::BackendError::NotContiguous));
}

#[test]
fn binary_not_contiguous_error() {
    let a = vec_view((0..12).map(|i| i as f32).collect()).reshape_contiguous(&[3, 4]);
    let b = vec_view((0..12).rev().map(|i| i as f32).collect()).reshape_contiguous(&[3, 4]);
    let a_nc = a.permute(&[1, 0]);
    let be = CpuBackend;
    let err = be.binary(BinaryOp::Add, &a_nc, &b).unwrap_err();
    assert!(matches!(err, crate::backend::BackendError::NotContiguous));
}

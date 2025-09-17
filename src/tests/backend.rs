use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::backend::CudaBackend;
use crate::backend::{registry, Backend, BackendError, BinaryOp, UnaryOp};
use crate::core::{storage::StorageInner, view::View};
use crate::device::Device;

fn vec_view(v: Vec<f32>) -> View {
    let inner = Arc::new(StorageInner::from_vec(v, Device::Cpu));
    View::from_inner_1d(inner)
}

#[cfg(feature = "cuda")]
fn cuda_view(v: Vec<f32>) -> View {
    let inner = Arc::new(StorageInner::from_vec(v, Device::Cuda));
    View::from_inner_1d(inner)
}

#[cfg(feature = "cuda")]
fn cuda_backend_or_skip(test_name: &str) -> Option<&'static CudaBackend> {
    match registry::cuda() {
        Ok(be) => Some(be),
        Err(err) => {
            eprintln!("skipping {test_name}: {err}");
            None
        }
    }
}

#[cfg(feature = "cuda")]
fn assert_close(got: f32, expected: f32, tol: f32, idx: usize) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol,
        "value mismatch at index {idx}: got {got}, expected {expected}, diff {diff} > tol {tol}"
    );
}

#[cfg(feature = "cuda")]
fn assert_vec_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (idx, (got, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(*got, *exp, tol, idx);
    }
}

#[cfg(feature = "cuda")]
fn run_cuda_unary_case<F>(name: &str, op: UnaryOp, input: &[f32], reference: F)
where
    F: Fn(f32) -> f32,
{
    let backend = match cuda_backend_or_skip(name) {
        Some(be) => be,
        None => return,
    };

    let view = cuda_view(input.to_vec());
    let out = backend
        .unary(op, &view)
        .unwrap_or_else(|err| panic!("{} failed: {}", name, err));
    let actual = out.inner.as_slice(out.offset, out.numel());
    let expected: Vec<f32> = input.iter().copied().map(reference).collect();
    assert_vec_close(actual, expected.as_slice(), 1e-5);
}

#[cfg(feature = "cuda")]
fn run_cuda_binary_case<F>(name: &str, op: BinaryOp, lhs: &[f32], rhs: &[f32], reference: F)
where
    F: Fn(f32, f32) -> f32,
{
    let backend = match cuda_backend_or_skip(name) {
        Some(be) => be,
        None => return,
    };

    let a = cuda_view(lhs.to_vec());
    let b = cuda_view(rhs.to_vec());
    let out = backend
        .binary(op, &a, &b)
        .unwrap_or_else(|err| panic!("{} failed: {}", name, err));
    let actual = out.inner.as_slice(out.offset, out.numel());
    let expected: Vec<f32> = lhs
        .iter()
        .copied()
        .zip(rhs.iter().copied())
        .map(|(x, y)| reference(x, y))
        .collect();
    assert_vec_close(actual, expected.as_slice(), 1e-5);
}

#[test]
fn unary_neg_matches_scalar() {
    let a = vec_view(vec![1.0, -2.0, 3.0, -4.5, 0.25]);
    let be = registry::cpu();
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
    let be = registry::cpu();
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
    let be = registry::cpu();
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
    let be = registry::cpu();
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
    let be = registry::cpu();
    let err = be.binary(BinaryOp::Add, &a, &b).unwrap_err();
    assert!(matches!(err, crate::backend::BackendError::ShapeMismatch));
}

#[test]
fn unary_not_contiguous_error() {
    let a = vec_view((0..12).map(|i| i as f32).collect()).reshape_contiguous(&[3, 4]);
    let a_nc = a.permute(&[1, 0]);
    assert!(!a_nc.is_contiguous());
    let be = registry::cpu();
    let err = be.unary(UnaryOp::Neg, &a_nc).unwrap_err();
    assert!(matches!(err, crate::backend::BackendError::NotContiguous));
}

#[test]
fn binary_not_contiguous_error() {
    let a = vec_view((0..12).map(|i| i as f32).collect()).reshape_contiguous(&[3, 4]);
    let b = vec_view((0..12).rev().map(|i| i as f32).collect()).reshape_contiguous(&[3, 4]);
    let a_nc = a.permute(&[1, 0]);
    let be = registry::cpu();
    let err = be.binary(BinaryOp::Add, &a_nc, &b).unwrap_err();
    assert!(matches!(err, crate::backend::BackendError::NotContiguous));
}

#[test]
fn registry_get_cpu_backend() {
    let backend = registry::get(Device::Cpu).unwrap();
    let view = vec_view(vec![1.0, -2.0, 3.0]);
    let out = backend.unary(UnaryOp::Neg, &view).unwrap();
    assert_eq!(
        out.inner.as_slice(out.offset, out.numel()),
        &[-1.0, 2.0, -3.0]
    );
}

#[test]
fn registry_reports_unsupported_device() {
    let err = match registry::get(Device::Metal) {
        Ok(_) => panic!("expected registry lookup to fail for Metal"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        BackendError::UnsupportedDevice(Device::Metal)
    ));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn registry_reports_cuda_unavailable_when_feature_disabled() {
    let err = match registry::get(Device::Cuda) {
        Ok(_) => panic!("expected CUDA backend lookup to fail without feature"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        BackendError::CudaUnavailable(message) if message.contains("`cuda` feature")
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unary_neg_matches_cpu_when_device_available() {
    run_cuda_unary_case(
        "cuda_unary_neg",
        UnaryOp::Neg,
        &[1.0, -3.0, 0.5, 7.25],
        |x| -x,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unary_sqrt_matches_cpu_when_device_available() {
    run_cuda_unary_case(
        "cuda_unary_sqrt",
        UnaryOp::Sqrt,
        &[0.0, 0.25, 1.0, 4.0, 9.0],
        |x| x.sqrt(),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unary_exp_matches_cpu_when_device_available() {
    run_cuda_unary_case(
        "cuda_unary_exp",
        UnaryOp::Exp,
        &[1.0, -2.5, 3.25, -4.75, 0.0],
        |x| x.exp(),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unary_log_matches_cpu_when_device_available() {
    run_cuda_unary_case(
        "cuda_unary_log",
        UnaryOp::Log,
        &[0.25, 1.0, 2.5, 10.0],
        |x| x.ln(),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unary_relu_matches_cpu_when_device_available() {
    run_cuda_unary_case(
        "cuda_unary_relu",
        UnaryOp::Relu,
        &[-3.0, -0.5, 0.0, 0.5, 2.0, 5.0],
        |x| x.max(0.0),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_binary_add_matches_cpu_when_device_available() {
    let lhs: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
    let rhs: Vec<f32> = (0..64).map(|i| i as f32 * -0.25).collect();
    run_cuda_binary_case("cuda_binary_add", BinaryOp::Add, &lhs, &rhs, |x, y| x + y);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_binary_sub_matches_cpu_when_device_available() {
    let lhs: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
    let rhs: Vec<f32> = (0..64).map(|i| i as f32 * -0.25).collect();
    run_cuda_binary_case("cuda_binary_sub", BinaryOp::Sub, &lhs, &rhs, |x, y| x - y);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_binary_mul_matches_cpu_when_device_available() {
    let lhs: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
    let rhs: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.1).collect();
    run_cuda_binary_case("cuda_binary_mul", BinaryOp::Mul, &lhs, &rhs, |x, y| x * y);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_binary_div_matches_cpu_when_device_available() {
    let lhs: Vec<f32> = (1..65).map(|i| i as f32).collect();
    let rhs: Vec<f32> = (1..65).map(|i| (i as f32) * 0.5 + 1.0).collect();
    run_cuda_binary_case("cuda_binary_div", BinaryOp::Div, &lhs, &rhs, |x, y| x / y);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unary_not_contiguous_error_when_device_available() {
    let backend = match cuda_backend_or_skip("cuda_unary_not_contiguous_error") {
        Some(be) => be,
        None => return,
    };
    let view = cuda_view((0..12).map(|i| i as f32).collect()).reshape_contiguous(&[3, 4]);
    let non_contig = view.permute(&[1, 0]);
    let err = backend.unary(UnaryOp::Neg, &non_contig).unwrap_err();
    assert!(matches!(err, crate::backend::BackendError::NotContiguous));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_binary_not_contiguous_error_when_device_available() {
    let backend = match cuda_backend_or_skip("cuda_binary_not_contiguous_error") {
        Some(be) => be,
        None => return,
    };
    let view_a = cuda_view((0..12).map(|i| i as f32).collect()).reshape_contiguous(&[3, 4]);
    let view_b = cuda_view((0..12).rev().map(|i| i as f32).collect()).reshape_contiguous(&[3, 4]);
    let non_contig = view_a.permute(&[1, 0]);
    let err = backend
        .binary(BinaryOp::Add, &non_contig, &view_b)
        .unwrap_err();
    assert!(matches!(err, crate::backend::BackendError::NotContiguous));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_binary_shape_mismatch_error_when_device_available() {
    let backend = match cuda_backend_or_skip("cuda_binary_shape_mismatch_error") {
        Some(be) => be,
        None => return,
    };
    let a = cuda_view(vec![1.0, 2.0, 3.0]);
    let b = cuda_view(vec![4.0, 5.0]);
    let err = backend.binary(BinaryOp::Add, &a, &b).unwrap_err();
    assert!(matches!(err, crate::backend::BackendError::ShapeMismatch));
}

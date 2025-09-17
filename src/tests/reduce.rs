use std::sync::Arc;

use crate::core::{
    reduce::{self, AxisError},
    storage::StorageInner,
    view::View,
};

fn make_view(shape: &[usize], data: Vec<f32>) -> View {
    let inner = Arc::new(StorageInner::from_vec(data, crate::device::Device::Cpu));
    View::from_inner_contiguous(inner, shape)
}

#[test]
fn numel_handles_scalar_and_shape() {
    assert_eq!(reduce::numel(&[]), 1);
    assert_eq!(reduce::numel(&[3, 4, 5]), 60);
}

#[test]
fn ensure_contiguous_preserves_or_clones() {
    let base = make_view(&[2, 3], vec![0.0; 6]);
    let cloned = reduce::ensure_contiguous(&base);
    assert!(Arc::ptr_eq(&cloned.inner, &base.inner));

    let permuted = base.permute(&[1, 0]);
    let contiguous = reduce::ensure_contiguous(&permuted);
    assert!(contiguous.is_contiguous());
    assert_eq!(contiguous.shape, vec![3, 2]);
}

#[test]
fn normalize_axis_reports_errors() {
    assert_eq!(reduce::normalize_axis(Some(-1), 3).unwrap(), Some(2));
    assert!(matches!(
        reduce::normalize_axis(Some(0), 0),
        Err(AxisError::ScalarHasNoDim)
    ));
    assert!(matches!(
        reduce::normalize_axis(Some(5), 3),
        Err(AxisError::OutOfRange)
    ));
}

#[test]
fn reduce_sum_none_matches_total() {
    let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let view = make_view(&[2, 3], data);
    let permuted = view.permute(&[1, 0]);
    let out = reduce::reduce_sum(&permuted, None, false);
    assert_eq!(out.shape, Vec::<usize>::new());
    assert_eq!(out.inner.as_slice(out.offset, out.numel()), &[21.0]);

    let out_keep = reduce::reduce_sum(&permuted, None, true);
    assert_eq!(out_keep.shape, vec![1, 1]);
    assert_eq!(
        out_keep.inner.as_slice(out_keep.offset, out_keep.numel()),
        &[21.0]
    );
}

#[test]
fn reduce_sum_axis_matches_manual() {
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let view = make_view(&[2, 3, 2], data.clone());
    let out = reduce::reduce_sum(&view, Some(1), false);
    assert_eq!(out.shape, vec![2, 2]);
    let slice = out.inner.as_slice(out.offset, out.numel());
    let mut expected = Vec::new();
    for i in 0..2 {
        for k in 0..2 {
            let mut acc = 0.0f32;
            for j in 0..3 {
                let idx = (i * 3 + j) * 2 + k;
                acc += data[idx];
            }
            expected.push(acc);
        }
    }
    assert_eq!(slice, expected.as_slice());

    let out_keep = reduce::reduce_sum(&view, Some(1), true);
    assert_eq!(out_keep.shape, vec![2, 1, 2]);
}

#[test]
fn reduce_max_axis_matches_manual() {
    let data: Vec<f32> = vec![1.0, 5.0, -2.0, 3.0, 4.0, 0.5, 7.0, -1.0];
    let view = make_view(&[2, 2, 2], data.clone());
    let out = reduce::reduce_max(&view, Some(0), false);
    assert_eq!(out.shape, vec![2, 2]);
    let slice = out.inner.as_slice(out.offset, out.numel());
    let mut expected = Vec::new();
    for j in 0..2 {
        for k in 0..2 {
            let mut best = f32::NEG_INFINITY;
            for i in 0..2 {
                let idx = (i * 2 + j) * 2 + k;
                best = best.max(data[idx]);
            }
            expected.push(best);
        }
    }
    assert_eq!(slice, expected.as_slice());

    let out_keep = reduce::reduce_max(&view, Some(0), true);
    assert_eq!(out_keep.shape, vec![1, 2, 2]);
}

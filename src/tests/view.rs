use std::sync::Arc;

use crate::core::{storage::StorageInner, view::View};
use crate::device::Device;

#[test]
fn view_1d_contiguous_and_numel() {
    let inner = Arc::new(StorageInner::new_full(0.0, 10, Device::Cpu));
    let v = View::from_inner_1d(inner);
    assert!(v.is_contiguous());
    assert_eq!(v.numel(), 10);
    assert_eq!(v.shape, vec![10]);
    assert_eq!(v.strides, vec![1]);
}

#[test]
fn view_slice_1d() {
    let inner = Arc::new(StorageInner::new_full(0.0, 10, Device::Cpu));
    let v = View::from_inner_1d(inner);
    let s = v.slice_1d(3, 4);
    assert!(s.is_contiguous());
    assert_eq!(s.offset, 3);
    assert_eq!(s.shape, vec![4]);
    assert_eq!(s.strides, vec![1]);
}

#[test]
fn view_reshape_contiguous() {
    let inner = Arc::new(StorageInner::new_full(0.0, 12, Device::Cpu));
    let v = View::from_inner_1d(inner);
    let r = v.reshape_contiguous(&[3, 4]);
    assert!(r.is_contiguous());
    assert_eq!(r.shape, vec![3, 4]);
    assert_eq!(r.strides, vec![4, 1]);
    assert_eq!(r.offset, 0);
}

#[test]
fn view_from_inner_contiguous_matches_shape() {
    let inner = Arc::new(StorageInner::new_full(1.0, 12, Device::Cpu));
    let v = View::from_inner_contiguous(inner.clone(), &[3, 4]);
    assert!(v.is_contiguous());
    assert_eq!(v.shape, vec![3, 4]);
    assert_eq!(v.strides, vec![4, 1]);
    assert!(Arc::ptr_eq(&v.inner, &inner));
}

#[test]
fn view_permute_non_contiguous() {
    let inner = Arc::new(StorageInner::new_full(0.0, 12, Device::Cpu));
    let v = View::from_inner_1d(inner).reshape_contiguous(&[3, 4]);
    let p = v.permute(&[1, 0]);
    assert_eq!(p.shape, vec![4, 3]);
    assert_eq!(p.strides, vec![1, 4]);
    assert!(!p.is_contiguous());
}

#[test]
fn view_expand_broadcast() {
    let inner = Arc::new(StorageInner::new_full(0.0, 3, Device::Cpu));
    let v = View::from_inner_1d(inner).reshape_contiguous(&[1, 3]);
    let e = v.expand(&[2, 3]);
    assert_eq!(e.shape, vec![2, 3]);
    assert_eq!(e.strides, vec![0, 1]); // broadcasted axis gets stride 0
    assert!(!e.is_contiguous());
}

#[test]
fn view_expand_incompatible_panics() {
    let inner = Arc::new(StorageInner::new_full(0.0, 3, Device::Cpu));
    let v = View::from_inner_1d(inner);
    let res = std::panic::catch_unwind(|| {
        let _ = v.expand(&[4]);
    });
    assert!(res.is_err());
}

#[test]
fn view_to_contiguous_copies_data_in_row_major_order() {
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let inner = Arc::new(StorageInner::from_vec(data.clone(), Device::Cpu));
    let base = View::from_inner_contiguous(inner, &[3, 4]);
    let permuted = base.permute(&[1, 0]);
    assert!(!permuted.is_contiguous());
    let contiguous = permuted.to_contiguous();
    assert!(contiguous.is_contiguous());
    assert_eq!(contiguous.shape, vec![4, 3]);
    let slice = contiguous
        .inner
        .as_slice(contiguous.offset, contiguous.numel());
    let mut expected = vec![0.0f32; 12];
    for i in 0..3 {
        for j in 0..4 {
            expected[j * 3 + i] = data[i * 4 + j];
        }
    }
    assert_eq!(slice, expected.as_slice());
}

#[test]
fn view_reshape_requires_contiguous_panics() {
    let inner = Arc::new(StorageInner::new_full(0.0, 12, Device::Cpu));
    let v = View::from_inner_1d(inner).reshape_contiguous(&[3, 4]);
    let p = v.permute(&[1, 0]);
    let res = std::panic::catch_unwind(|| {
        let _ = p.reshape_contiguous(&[2, 6]);
    });
    assert!(res.is_err());
}

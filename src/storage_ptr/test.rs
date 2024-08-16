use std::ops::Neg;

use crate::storage_ptr::{
    storage_add, storage_clone, storage_div, storage_drop, storage_exp, storage_full,
    storage_full_vec, storage_log, storage_max, storage_mul, storage_neg, storage_relu,
    storage_sqrt, storage_sub, storage_sum,
};
use rand::random;

const DEVICE: &str = "cpu";
const VEC_SIZE: usize = 1000;
const DECIMAL_PLACES: u8 = 6;
const EPSILON: f32 = 1e-6; // poor man's float comparison

#[test]
fn test_storage_full() {
    todo!();
}
#[test]
fn test_storage_full_vec() {
    todo!()
}
#[test]
fn test_storage_clone() {
    todo!()
}
#[test]
fn test_storage_drop() {
    todo!()
}
#[test]
fn test_storage_neg() {
    let fill_vec: Vec<f32> = (0..VEC_SIZE).map(|_| random::<f32>()).collect();

    let a = storage_full_vec(fill_vec.clone(), DEVICE);
    let mut b = storage_full(0.0, VEC_SIZE, DEVICE);
    storage_neg(&a, &mut b, 0, 0, VEC_SIZE);

    let b_data = b.get_storage().get_items(0, VEC_SIZE).to_vec();
    for i in 0..VEC_SIZE {
        assert!((fill_vec[i].neg() - b_data[i]).abs() < EPSILON);
    }
}
#[test]
fn test_storage_sqrt() {
    let fill_vec: Vec<f32> = (0..VEC_SIZE).map(|_| random::<f32>()).collect();

    let a = storage_full_vec(fill_vec.clone(), DEVICE);
    let mut b = storage_full(0.0, VEC_SIZE, DEVICE);
    storage_sqrt(&a, &mut b, 0, 0, VEC_SIZE);

    let b_data = b.get_storage().get_items(0, VEC_SIZE);
    for i in 0..VEC_SIZE {
        assert!((fill_vec[i].sqrt() - b_data[i]).abs() < EPSILON);
    }
}
#[test]
fn test_storage_relu() {
    todo!()
}
#[test]
fn test_storage_exp() {
    todo!()
}
#[test]
fn test_storage_log() {
    todo!()
}
#[test]
fn test_storage_add() {
    todo!()
}
#[test]
fn test_storage_sub() {
    todo!()
}
#[test]
fn test_storage_mul() {
    todo!()
}
#[test]
fn test_storage_div() {
    todo!()
}
#[test]
fn test_storage_sum() {
    todo!()
}
#[test]
fn test_storage_max() {
    todo!()
}

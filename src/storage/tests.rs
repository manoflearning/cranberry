use crate::storage::Storage;
use rand::random;

const DEVICE: &str = "cpu";
const VEC_SIZE: usize = 1000;

#[test]
fn test_storage_neg() {
    let x = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let y = x.iter().map(|a| -a).collect::<Vec<f32>>();

    let a = Storage::from_vec(x.clone(), DEVICE);
    let mut b = Storage::new(0.0, VEC_SIZE, DEVICE);
    Storage::neg(&a, &mut b, 0, 0, VEC_SIZE);

    assert!(y.as_slice() == b.get_items(0, VEC_SIZE));
}
#[test]
fn test_storage_sqrt() {
    let x = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let y = x.iter().map(|a| a.sqrt()).collect::<Vec<f32>>();

    let a = Storage::from_vec(x.clone(), DEVICE);
    let mut b = Storage::new(0.0, VEC_SIZE, DEVICE);
    Storage::sqrt(&a, &mut b, 0, 0, VEC_SIZE);

    assert!(y.as_slice() == b.get_items(0, VEC_SIZE));
}
#[test]
fn test_storage_exp() {
    let x = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let y = x.iter().map(|a| a.exp()).collect::<Vec<f32>>();

    let a = Storage::from_vec(x.clone(), DEVICE);
    let mut b = Storage::new(0.0, VEC_SIZE, DEVICE);
    Storage::exp(&a, &mut b, 0, 0, VEC_SIZE);

    assert!(y.as_slice() == b.get_items(0, VEC_SIZE));
}
#[test]
fn test_storage_log() {
    let x = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let y = x.iter().map(|a| a.ln()).collect::<Vec<f32>>();

    let a = Storage::from_vec(x.clone(), DEVICE);
    let mut b = Storage::new(0.0, VEC_SIZE, DEVICE);
    Storage::log(&a, &mut b, 0, 0, VEC_SIZE);

    assert!(y.as_slice() == b.get_items(0, VEC_SIZE));
}
#[test]
fn test_storage_add() {
    let x = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let y = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let z = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect::<Vec<f32>>();

    let a = Storage::from_vec(x.clone(), DEVICE);
    let b = Storage::from_vec(y.clone(), DEVICE);
    let mut c = Storage::new(0.0, VEC_SIZE, DEVICE);
    Storage::add(&a, &b, &mut c, 0, 0, 0, VEC_SIZE);

    assert!(z.as_slice() == c.get_items(0, VEC_SIZE));
}
#[test]
fn test_storage_sub() {
    let x = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let y = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let z = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect::<Vec<f32>>();

    let a = Storage::from_vec(x.clone(), DEVICE);
    let b = Storage::from_vec(y.clone(), DEVICE);
    let mut c = Storage::new(0.0, VEC_SIZE, DEVICE);
    Storage::sub(&a, &b, &mut c, 0, 0, 0, VEC_SIZE);

    assert!(z.as_slice() == c.get_items(0, VEC_SIZE));
}
#[test]
fn test_storage_mul() {
    let x = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let y = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let z = x.iter().zip(y.iter()).map(|(a, b)| a * b).collect::<Vec<f32>>();

    let a = Storage::from_vec(x.clone(), DEVICE);
    let b = Storage::from_vec(y.clone(), DEVICE);
    let mut c = Storage::new(0.0, VEC_SIZE, DEVICE);
    Storage::mul(&a, &b, &mut c, 0, 0, 0, VEC_SIZE);

    assert!(z.as_slice() == c.get_items(0, VEC_SIZE));
}
#[test]
fn test_storage_div() {
    let x = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let y = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let z = x.iter().zip(y.iter()).map(|(a, b)| a / b).collect::<Vec<f32>>();

    let a = Storage::from_vec(x.clone(), DEVICE);
    let b = Storage::from_vec(y.clone(), DEVICE);
    let mut c = Storage::new(0.0, VEC_SIZE, DEVICE);
    Storage::div(&a, &b, &mut c, 0, 0, 0, VEC_SIZE);

    assert!(z.as_slice() == c.get_items(0, VEC_SIZE));
}
#[test]
fn test_storage_sum() {
    let x = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let y = x.iter().sum::<f32>();

    let a = Storage::from_vec(x.clone(), DEVICE);
    let mut b = Storage::new(0.0, 1, DEVICE);
    Storage::sum(&a, &mut b, 0, 0, VEC_SIZE);

    assert!((y - b.get_items(0, 1)[0]).abs() < 1e-6 * y); // not sure this is right
}
#[test]
fn test_storage_max() {
    let x = (0..VEC_SIZE).map(|_| random::<f32>()).collect::<Vec<f32>>();
    let y = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let a = Storage::from_vec(x.clone(), DEVICE);
    let mut b = Storage::new(0.0, 1, DEVICE);
    Storage::max(&a, &mut b, 0, 0, VEC_SIZE);

    assert!(y == b.get_items(0, 1)[0]);
}

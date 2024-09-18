use crate::storage_ptr::StoragePtr;
use rand::random;

const DEVICE: &str = "cpu";
const MAX_VEC_SIZE: usize = 4000;
const DEFAULT_TEST_COUNT: usize = 10;

#[test]
fn test_storage_ptr_neg() {
    for _ in 0..DEFAULT_TEST_COUNT {
        let vec_size = random::<usize>() % MAX_VEC_SIZE + 1;

        let idx_1 = random::<usize>() % vec_size;
        let idx_2 = random::<usize>() % vec_size;
        let size = random::<usize>() % (vec_size - idx_1.max(idx_2)) + 1;

        let x = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let mut y = vec![0.0; vec_size];
        for i in 0..size {
            y[idx_2 + i] = -x[idx_1 + i];
        }

        let a = StoragePtr::from_vec(x.clone(), DEVICE);
        let b = StoragePtr::full(0.0, vec_size, DEVICE);
        StoragePtr::neg(&a, &b, idx_1, idx_2, size);

        assert_eq!(b.to_vec(), y);
    }
}
#[test]
fn test_storage_ptr_sqrt() {
    for _ in 0..DEFAULT_TEST_COUNT {
        let vec_size = random::<usize>() % MAX_VEC_SIZE + 1;

        let idx_1 = random::<usize>() % vec_size;
        let idx_2 = random::<usize>() % vec_size;
        let size = random::<usize>() % (vec_size - idx_1.max(idx_2)) + 1;

        let x = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let mut y = vec![0.0; vec_size];
        for i in 0..size {
            y[idx_2 + i] = x[idx_1 + i].sqrt();
        }

        let a = StoragePtr::from_vec(x.clone(), DEVICE);
        let b = StoragePtr::full(0.0, vec_size, DEVICE);
        StoragePtr::sqrt(&a, &b, idx_1, idx_2, size);

        assert_eq!(b.to_vec(), y);
    }
}
#[test]
fn test_storage_ptr_exp() {
    for _ in 0..DEFAULT_TEST_COUNT {
        let vec_size = random::<usize>() % MAX_VEC_SIZE + 1;

        let idx_1 = random::<usize>() % vec_size;
        let idx_2 = random::<usize>() % vec_size;
        let size = random::<usize>() % (vec_size - idx_1.max(idx_2)) + 1;

        let x = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let mut y = vec![0.0; vec_size];
        for i in 0..size {
            y[idx_2 + i] = x[idx_1 + i].exp();
        }

        let a = StoragePtr::from_vec(x.clone(), DEVICE);
        let b = StoragePtr::full(0.0, vec_size, DEVICE);
        StoragePtr::exp(&a, &b, idx_1, idx_2, size);

        assert_eq!(b.to_vec(), y);
    }
}
#[test]
fn test_storage_ptr_log() {
    for _ in 0..DEFAULT_TEST_COUNT {
        let vec_size = random::<usize>() % MAX_VEC_SIZE + 1;

        let idx_1 = random::<usize>() % vec_size;
        let idx_2 = random::<usize>() % vec_size;
        let size = random::<usize>() % (vec_size - idx_1.max(idx_2)) + 1;

        let x = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let mut y = vec![0.0; vec_size];
        for i in 0..size {
            y[idx_2 + i] = x[idx_1 + i].ln();
        }

        let a = StoragePtr::from_vec(x.clone(), DEVICE);
        let b = StoragePtr::full(0.0, vec_size, DEVICE);
        StoragePtr::log(&a, &b, idx_1, idx_2, size);

        assert_eq!(b.to_vec(), y);
    }
}
#[test]
fn test_storage_ptr_add() {
    for _ in 0..DEFAULT_TEST_COUNT {
        let vec_size = random::<usize>() % MAX_VEC_SIZE + 1;

        let idx_1 = random::<usize>() % vec_size;
        let idx_2 = random::<usize>() % vec_size;
        let idx_3 = random::<usize>() % vec_size;
        let size = random::<usize>() % (vec_size - idx_1.max(idx_2).max(idx_3)) + 1;

        let x = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let y = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let mut z = vec![0.0; vec_size];
        for i in 0..size {
            z[idx_3 + i] = x[idx_1 + i] + y[idx_2 + i];
        }

        let a = StoragePtr::from_vec(x.clone(), DEVICE);
        let b = StoragePtr::from_vec(y.clone(), DEVICE);
        let c = StoragePtr::full(0.0, vec_size, DEVICE);
        StoragePtr::add(&a, &b, &c, idx_1, idx_2, idx_3, size);

        assert_eq!(c.to_vec(), z);
    }
}
#[test]
fn test_storage_ptr_sub() {
    for _ in 0..DEFAULT_TEST_COUNT {
        let vec_size = random::<usize>() % MAX_VEC_SIZE + 1;

        let idx_1 = random::<usize>() % vec_size;
        let idx_2 = random::<usize>() % vec_size;
        let idx_3 = random::<usize>() % vec_size;
        let size = random::<usize>() % (vec_size - idx_1.max(idx_2).max(idx_3)) + 1;

        let x = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let y = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let mut z = vec![0.0; vec_size];
        for i in 0..size {
            z[idx_3 + i] = x[idx_1 + i] - y[idx_2 + i];
        }

        let a = StoragePtr::from_vec(x.clone(), DEVICE);
        let b = StoragePtr::from_vec(y.clone(), DEVICE);
        let c = StoragePtr::full(0.0, vec_size, DEVICE);
        StoragePtr::sub(&a, &b, &c, idx_1, idx_2, idx_3, size);

        assert_eq!(c.to_vec(), z);
    }
}
#[test]
fn test_storage_ptr_mul() {
    for _ in 0..DEFAULT_TEST_COUNT {
        let vec_size = random::<usize>() % MAX_VEC_SIZE + 1;

        let idx_1 = random::<usize>() % vec_size;
        let idx_2 = random::<usize>() % vec_size;
        let idx_3 = random::<usize>() % vec_size;
        let size = random::<usize>() % (vec_size - idx_1.max(idx_2).max(idx_3)) + 1;

        let x = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let y = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let mut z = vec![0.0; vec_size];
        for i in 0..size {
            z[idx_3 + i] = x[idx_1 + i] * y[idx_2 + i];
        }

        let a = StoragePtr::from_vec(x.clone(), DEVICE);
        let b = StoragePtr::from_vec(y.clone(), DEVICE);
        let c = StoragePtr::full(0.0, vec_size, DEVICE);
        StoragePtr::mul(&a, &b, &c, idx_1, idx_2, idx_3, size);

        assert_eq!(c.to_vec(), z);
    }
}
#[test]
fn test_storage_ptr_div() {
    for _ in 0..DEFAULT_TEST_COUNT {
        let vec_size = random::<usize>() % MAX_VEC_SIZE + 1;

        let idx_1 = random::<usize>() % vec_size;
        let idx_2 = random::<usize>() % vec_size;
        let idx_3 = random::<usize>() % vec_size;
        let size = random::<usize>() % (vec_size - idx_1.max(idx_2).max(idx_3)) + 1;

        let x = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let y = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let mut z = vec![0.0; vec_size];
        for i in 0..size {
            z[idx_3 + i] = x[idx_1 + i] / y[idx_2 + i];
        }

        let a = StoragePtr::from_vec(x.clone(), DEVICE);
        let b = StoragePtr::from_vec(y.clone(), DEVICE);
        let c = StoragePtr::full(0.0, vec_size, DEVICE);
        StoragePtr::div(&a, &b, &c, idx_1, idx_2, idx_3, size);

        assert_eq!(c.to_vec(), z);
    }
}
#[test]
fn test_storage_ptr_sum() {
    for _ in 0..DEFAULT_TEST_COUNT {
        let vec_size = random::<usize>() % MAX_VEC_SIZE + 1;

        let idx_1 = random::<usize>() % vec_size;
        let idx_2 = random::<usize>() % vec_size;
        let size = random::<usize>() % (vec_size - idx_1) + 1;

        let x = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let mut y = vec![0.0; vec_size];
        for i in 0..size {
            y[idx_2] += x[idx_1 + i];
        }

        let a = StoragePtr::from_vec(x.clone(), DEVICE);
        let b = StoragePtr::full(0.0, vec_size, DEVICE);
        StoragePtr::sum(&a, &b, idx_1, idx_2, size);

        assert!((y[idx_2] - b.to_vec()[idx_2]).abs() < 1e-5 * y[idx_2].abs());
    }
}
#[test]
fn test_storage_ptr_max() {
    for _ in 0..DEFAULT_TEST_COUNT {
        let vec_size = random::<usize>() % MAX_VEC_SIZE + 1;

        let idx_1 = random::<usize>() % vec_size;
        let idx_2 = random::<usize>() % vec_size;
        let size = random::<usize>() % (vec_size - idx_1) + 1;

        let x = (0..vec_size).map(|_| random::<f32>()).collect::<Vec<f32>>();
        let mut y = vec![0.0; vec_size];
        y[idx_2] = f32::NEG_INFINITY;
        for i in 0..size {
            if y[idx_2] < x[idx_1 + i] {
                y[idx_2] = x[idx_1 + i];
            }
        }

        let a = StoragePtr::from_vec(x.clone(), DEVICE);
        let b = StoragePtr::full(0.0, vec_size, DEVICE);
        StoragePtr::max(&a, &b, idx_1, idx_2, size);

        assert_eq!(b.to_vec(), y);
    }
}

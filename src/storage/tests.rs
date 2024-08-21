use crate::storage::Storage;
use rand::random;

const DEVICE: &str = "cpu";
const MAX_VEC_SIZE: usize = 4000;
const DEFAULT_TEST_COUNT: usize = 10;

#[test]
fn test_storage_neg() {
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

        let a = Storage::from_vec(x.clone(), DEVICE);
        let mut b = Storage::new(0.0, vec_size, DEVICE);
        Storage::neg(&a, &mut b, idx_1, idx_2, size);

        assert!(y.as_slice() == b.get_items(0, vec_size));
    }
}
#[test]
fn test_storage_sqrt() {
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

        let a = Storage::from_vec(x.clone(), DEVICE);
        let mut b = Storage::new(0.0, vec_size, DEVICE);
        Storage::sqrt(&a, &mut b, idx_1, idx_2, size);

        assert!(y.as_slice() == b.get_items(0, vec_size));
    }
}
#[test]
fn test_storage_exp() {
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

        let a = Storage::from_vec(x.clone(), DEVICE);
        let mut b = Storage::new(0.0, vec_size, DEVICE);
        Storage::exp(&a, &mut b, idx_1, idx_2, size);

        assert!(y.as_slice() == b.get_items(0, vec_size));
    }
}
#[test]
fn test_storage_log() {
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

        let a = Storage::from_vec(x.clone(), DEVICE);
        let mut b = Storage::new(0.0, vec_size, DEVICE);
        Storage::log(&a, &mut b, idx_1, idx_2, size);

        assert!(y.as_slice() == b.get_items(0, vec_size));
    }
}
#[test]
fn test_storage_add() {
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

        let a = Storage::from_vec(x.clone(), DEVICE);
        let b = Storage::from_vec(y.clone(), DEVICE);
        let mut c = Storage::new(0.0, vec_size, DEVICE);
        Storage::add(&a, &b, &mut c, idx_1, idx_2, idx_3, size);

        assert!(y.as_slice() == b.get_items(0, vec_size));
    }
}
#[test]
fn test_storage_sub() {
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

        let a = Storage::from_vec(x.clone(), DEVICE);
        let b = Storage::from_vec(y.clone(), DEVICE);
        let mut c = Storage::new(0.0, vec_size, DEVICE);
        Storage::sub(&a, &b, &mut c, idx_1, idx_2, idx_3, size);

        assert!(y.as_slice() == b.get_items(0, vec_size));
    }
}
#[test]
fn test_storage_mul() {
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

        let a = Storage::from_vec(x.clone(), DEVICE);
        let b = Storage::from_vec(y.clone(), DEVICE);
        let mut c = Storage::new(0.0, vec_size, DEVICE);
        Storage::mul(&a, &b, &mut c, idx_1, idx_2, idx_3, size);

        assert!(y.as_slice() == b.get_items(0, vec_size));
    }
}
#[test]
fn test_storage_div() {
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

        let a = Storage::from_vec(x.clone(), DEVICE);
        let b = Storage::from_vec(y.clone(), DEVICE);
        let mut c = Storage::new(0.0, vec_size, DEVICE);
        Storage::div(&a, &b, &mut c, idx_1, idx_2, idx_3, size);

        assert!(y.as_slice() == b.get_items(0, vec_size));
    }
}
#[test]
fn test_storage_sum() {
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

        let a = Storage::from_vec(x.clone(), DEVICE);
        let mut b = Storage::new(0.0, vec_size, DEVICE);
        Storage::sum(&a, &mut b, idx_1, idx_2, size);

        assert!((y[idx_2] - b.get_items(idx_2, 1)[0]).abs() < 1e-5 * y[idx_2]); // not sure this is right
    }
}
#[test]
fn test_storage_max() {
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

        let a = Storage::from_vec(x.clone(), DEVICE);
        let mut b = Storage::new(0.0, vec_size, DEVICE);
        Storage::max(&a, &mut b, idx_1, idx_2, size);

        assert!(y[idx_2] == b.get_items(idx_2, 1)[0]);
    }
}

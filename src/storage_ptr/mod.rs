use std::sync::{Arc, Mutex};

use crate::storage::Storage;

#[cfg(test)]
mod tests;

#[pyo3::pyclass]
#[derive(Clone)]
pub struct StoragePtr {
    storage: Arc<Mutex<Storage>>, // Arc allows shared ownership, Mutex ensures safe mutable access
}

impl StoragePtr {
    fn new(storage: Storage) -> StoragePtr {
        StoragePtr {
            storage: Arc::new(Mutex::new(storage)),
        }
    }
    fn from_storage(storage: Storage) -> StoragePtr {
        StoragePtr::new(storage)
    }
    fn get_storage(&self) -> Arc<Mutex<Storage>> {
        Arc::clone(&self.storage)
    }
}

#[pyo3::pymethods]
impl StoragePtr {
    #[staticmethod]
    fn full(value: f32, size: usize, device: &str) -> StoragePtr {
        StoragePtr::from_storage(Storage::new(value, size, device))
    }
    #[staticmethod]
    fn from_vec(vec: Vec<f32>, device: &str) -> StoragePtr {
        StoragePtr::from_storage(Storage::from_vec(vec, device))
    }
    #[staticmethod]
    fn neg(a: &StoragePtr, b: &StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        let binding = a.get_storage();
        let a_storage = binding.lock().unwrap();
        let binding = b.get_storage();
        let mut b_storage = binding.lock().unwrap();
        Storage::neg(&a_storage, &mut b_storage, idx_a, idx_b, size);
    }
    #[staticmethod]
    fn sqrt(a: &StoragePtr, b: &StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        let binding = a.get_storage();
        let a_storage = binding.lock().unwrap();
        let binding = b.get_storage();
        let mut b_storage = binding.lock().unwrap();
        Storage::sqrt(&a_storage, &mut b_storage, idx_a, idx_b, size);
    }
    #[staticmethod]
    fn exp(a: &StoragePtr, b: &StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        let binding = a.get_storage();
        let a_storage = binding.lock().unwrap();
        let binding = b.get_storage();
        let mut b_storage = binding.lock().unwrap();
        Storage::exp(&a_storage, &mut b_storage, idx_a, idx_b, size);
    }
    #[staticmethod]
    fn log(a: &StoragePtr, b: &StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        let binding = a.get_storage();
        let a_storage = binding.lock().unwrap();
        let binding = b.get_storage();
        let mut b_storage = binding.lock().unwrap();
        Storage::log(&a_storage, &mut b_storage, idx_a, idx_b, size);
    }
    #[staticmethod]
    fn add(
        a: &StoragePtr,
        b: &StoragePtr,
        c: &StoragePtr,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        let binding = a.get_storage();
        let a_storage = binding.lock().unwrap();
        let binding = b.get_storage();
        let b_storage = binding.lock().unwrap();
        let binding = c.get_storage();
        let mut c_storage = binding.lock().unwrap();
        Storage::add(
            &a_storage,
            &b_storage,
            &mut c_storage,
            idx_a,
            idx_b,
            idx_c,
            size,
        );
    }
    #[staticmethod]
    fn sub(
        a: &StoragePtr,
        b: &StoragePtr,
        c: &StoragePtr,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        let binding = a.get_storage();
        let a_storage = binding.lock().unwrap();
        let binding = b.get_storage();
        let b_storage = binding.lock().unwrap();
        let binding = c.get_storage();
        let mut c_storage = binding.lock().unwrap();
        Storage::sub(
            &a_storage,
            &b_storage,
            &mut c_storage,
            idx_a,
            idx_b,
            idx_c,
            size,
        );
    }
    #[staticmethod]
    fn mul(
        a: &StoragePtr,
        b: &StoragePtr,
        c: &StoragePtr,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        let binding = a.get_storage();
        let a_storage = binding.lock().unwrap();
        let binding = b.get_storage();
        let b_storage = binding.lock().unwrap();
        let binding = c.get_storage();
        let mut c_storage = binding.lock().unwrap();
        Storage::mul(
            &a_storage,
            &b_storage,
            &mut c_storage,
            idx_a,
            idx_b,
            idx_c,
            size,
        );
    }
    #[staticmethod]
    fn div(
        a: &StoragePtr,
        b: &StoragePtr,
        c: &StoragePtr,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        let binding = a.get_storage();
        let a_storage = binding.lock().unwrap();
        let binding = b.get_storage();
        let b_storage = binding.lock().unwrap();
        let binding = c.get_storage();
        let mut c_storage = binding.lock().unwrap();
        Storage::div(
            &a_storage,
            &b_storage,
            &mut c_storage,
            idx_a,
            idx_b,
            idx_c,
            size,
        );
    }
    #[staticmethod]
    fn sum(a: &StoragePtr, b: &StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        let binding = a.get_storage();
        let a_storage = binding.lock().unwrap();
        let binding = b.get_storage();
        let mut b_storage = binding.lock().unwrap();
        Storage::sum(&a_storage, &mut b_storage, idx_a, idx_b, size);
    }
    #[staticmethod]
    fn max(a: &StoragePtr, b: &StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        let binding = a.get_storage();
        let a_storage = binding.lock().unwrap();
        let binding = b.get_storage();
        let mut b_storage = binding.lock().unwrap();
        Storage::max(&a_storage, &mut b_storage, idx_a, idx_b, size);
    }

    fn to_vec(&self) -> Vec<f32> {
        let binding = self.get_storage();
        let self_storage = binding.lock().unwrap();
        self_storage.to_vec()
    }
}

use uuid::Uuid;

use crate::{device::Device, storage::Storage};

#[pyo3::pyclass]
#[derive(PartialEq)]
pub struct StoragePtr {
    ptr: String,
    uuid: String,
}

impl StoragePtr {
    fn new(ptr: String) -> StoragePtr {
        StoragePtr {
            ptr,
            uuid: Uuid::new_v4().to_string(),
        }
    }
    fn from_storage(storage: &Storage) -> StoragePtr {
        StoragePtr::new(format!("{:p}", storage as *const _))
    }
    fn get_storage(&self) -> &Storage {
        unsafe { &*(self.ptr.parse::<usize>().unwrap() as *const Storage) }
    }
    fn get_storage_mut(&mut self) -> &mut Storage {
        unsafe { &mut *(self.ptr.parse::<usize>().unwrap() as *mut Storage) }
    }
}

#[pyo3::pyfunction]
pub fn storage_full(fill_value: f32, size: usize, device: &str) -> StoragePtr {
    StoragePtr::from_storage(&Storage::new(fill_value, size, Device::from_str(device)))
}
#[pyo3::pyfunction]
pub fn storage_clone(storage_ptr: &mut StoragePtr) -> StoragePtr {
    storage_ptr.get_storage_mut().incref();
    StoragePtr::new(storage_ptr.ptr.clone())
}
#[pyo3::pyfunction]
pub fn storage_drop(storage_ptr: &mut StoragePtr) {
    storage_ptr.get_storage_mut().decref();
    storage_ptr.ptr.clear();
}
#[pyo3::pyfunction]
pub fn storage_neg(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
    Storage::neg(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
}
#[pyo3::pyfunction]
pub fn storage_sqrt(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
    Storage::sqrt(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size)
}
#[pyo3::pyfunction]
pub fn storage_relu(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
    Storage::relu(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
}
#[pyo3::pyfunction]
pub fn storage_exp(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
    Storage::exp(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
}
#[pyo3::pyfunction]
pub fn storage_log(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
    Storage::log(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
}
#[pyo3::pyfunction]
pub fn storage_add(
    a: &StoragePtr,
    b: &StoragePtr,
    c: &mut StoragePtr,
    idx_a: usize,
    idx_b: usize,
    idx_c: usize,
    size: usize,
) {
    Storage::add(
        a.get_storage(),
        b.get_storage(),
        c.get_storage_mut(),
        idx_a,
        idx_b,
        idx_c,
        size,
    );
}
#[pyo3::pyfunction]
pub fn storage_sub(
    a: &StoragePtr,
    b: &StoragePtr,
    c: &mut StoragePtr,
    idx_a: usize,
    idx_b: usize,
    idx_c: usize,
    size: usize,
) {
    Storage::sub(
        a.get_storage(),
        b.get_storage(),
        c.get_storage_mut(),
        idx_a,
        idx_b,
        idx_c,
        size,
    );
}
#[pyo3::pyfunction]
pub fn storage_mul(
    a: &StoragePtr,
    b: &StoragePtr,
    c: &mut StoragePtr,
    idx_a: usize,
    idx_b: usize,
    idx_c: usize,
    size: usize,
) {
    Storage::mul(
        a.get_storage(),
        b.get_storage(),
        c.get_storage_mut(),
        idx_a,
        idx_b,
        idx_c,
        size,
    );
}
#[pyo3::pyfunction]
pub fn storage_div(
    a: &StoragePtr,
    b: &StoragePtr,
    c: &mut StoragePtr,
    idx_a: usize,
    idx_b: usize,
    idx_c: usize,
    size: usize,
) {
    Storage::div(
        a.get_storage(),
        b.get_storage(),
        c.get_storage_mut(),
        idx_a,
        idx_b,
        idx_c,
        size,
    );
}
#[pyo3::pyfunction]
pub fn storage_sum(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
    Storage::sum(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
}
#[pyo3::pyfunction]
pub fn storage_max(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
    Storage::max(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
}

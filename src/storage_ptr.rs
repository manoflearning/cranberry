use uuid::Uuid;

use crate::storage::Storage;

#[pyo3::pyclass]
#[derive(PartialEq)]
pub struct StoragePtr {
    ptr: usize,
    uuid: String,
}

impl StoragePtr {
    fn new(ptr: usize) -> StoragePtr {
        StoragePtr {
            ptr,
            uuid: Uuid::new_v4().to_string(),
        }
    }
    fn from_storage(storage: &Storage) -> StoragePtr {
        StoragePtr::new(storage as *const Storage as usize)
    }
    fn get_storage(&self) -> &Storage {
        let ptr = self.ptr as *const Storage;
        assert!(!ptr.is_null());
        assert!(ptr.align_offset(std::mem::align_of::<Storage>()) == 0);
        unsafe { &*ptr }
    }
    fn get_storage_mut(&mut self) -> &mut Storage {
        let ptr = self.ptr as *mut Storage;
        assert!(!ptr.is_null());
        assert!(ptr.align_offset(std::mem::align_of::<Storage>()) == 0);
        unsafe { &mut *ptr }
    }
}

#[pyo3::pymethods]
impl StoragePtr {
    #[staticmethod]
    fn storage_full(value: f32, size: usize, device: &str) -> StoragePtr {
        StoragePtr::from_storage(&Storage::new(value, size, device))
    }
    #[staticmethod]
    fn storage_from_vec(vec: Vec<f32>, device: &str) -> StoragePtr {
        StoragePtr::from_storage(&Storage::from_vec(vec, device))
    }
    #[staticmethod]
    fn storage_clone(storage_ptr: &mut StoragePtr) -> StoragePtr {
        storage_ptr.get_storage_mut().incref();
        StoragePtr::new(storage_ptr.ptr)
    }
    #[staticmethod]
    fn storage_drop(storage_ptr: &mut StoragePtr) {
        storage_ptr.get_storage_mut().decref();
        storage_ptr.ptr = 0;
    }
    #[staticmethod]
    fn storage_neg(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        Storage::neg(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }
    #[staticmethod]
    fn storage_sqrt(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        Storage::sqrt(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size)
    }
    #[staticmethod]
    fn storage_exp(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        Storage::exp(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }
    #[staticmethod]
    fn storage_log(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        Storage::log(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }
    #[staticmethod]
    fn storage_add(
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
    #[staticmethod]
    fn storage_sub(
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
    #[staticmethod]
    fn storage_mul(
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
    #[staticmethod]
    fn storage_div(
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
    #[staticmethod]
    fn storage_sum(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        Storage::sum(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }
    #[staticmethod]
    fn storage_max(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        Storage::max(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }
}

#![feature(portable_simd)]
#![feature(array_chunks)]

mod device;
mod ops;
use device::Device;

pub struct Storage {
    data: Vec<f32>,
    data_size: usize,
    device: Device,
    ref_count: i32,
}

impl Storage {
    pub fn new(value: f32, size: usize, device: Device) -> Self {
        Storage {
            data: vec![value; size],
            data_size: size,
            device,
            ref_count: 1,
        }
    }
    pub fn incref(&mut self) {
        assert!(0 < self.ref_count);
        self.ref_count += 1;
    }
    pub fn decref(&mut self) {
        assert!(0 < self.ref_count);
        self.ref_count -= 1;
        if self.ref_count == 0 {
            // You might wonder why we manually drop the memory here,
            // instead of letting the Rust compiler handle it.
            // The reason is that this is the library code binding to the Python interpreter.
            // The Rust compiler does not know when the Python interpreter will release the memory.
            // Therefore, we need to manually drop the memory when the reference count is zero.
            self.data.clear();
            self.data.shrink_to_fit();
        }
    }
}

fn storage_getitems(a: &Storage, idx: usize, size: usize) -> &[f32] {
    assert!(0 < size);
    assert!(idx + size <= a.data_size);
    a.data[idx..idx + size].as_ref()
}

fn storage_getitems_mut(a: &mut Storage, idx: usize, size: usize) -> &mut [f32] {
    assert!(0 < size);
    assert!(idx + size <= a.data_size);
    a.data[idx..idx + size].as_mut()
}

#[pyo3::pymodule]
mod storage {
    use crate::device::Device;
    use crate::Storage;
    use pyo3::prelude::*;
    use uuid::Uuid;

    #[pyclass]
    #[derive(PartialEq)]
    struct StoragePtr {
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

    #[pyfunction]
    fn storage_full(fill_value: f32, size: usize, device: &str) -> StoragePtr {
        StoragePtr::from_storage(&Storage::new(fill_value, size, Device::from_str(device)))
    }

    #[pyfunction]
    fn storage_clone(storage_ptr: &mut StoragePtr) -> StoragePtr {
        storage_ptr.get_storage_mut().incref();
        StoragePtr::new(storage_ptr.ptr.clone())
    }
    #[pyfunction]
    fn storage_drop(storage_ptr: &mut StoragePtr) {
        storage_ptr.get_storage_mut().decref();
        storage_ptr.ptr.clear();
    }

    #[pyfunction]
    fn storage_neg(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        crate::ops::storage_neg(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }
    #[pyfunction]
    fn storage_sqrt(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        crate::ops::storage_sqrt(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size)
    }
    #[pyfunction]
    fn storage_relu(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        crate::ops::storage_relu(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }
    #[pyfunction]
    fn storage_exp(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        crate::ops::storage_exp(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }
    #[pyfunction]
    fn storage_log(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        crate::ops::storage_log(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }

    #[pyfunction]
    fn storage_add(
        a: &StoragePtr,
        b: &StoragePtr,
        c: &mut StoragePtr,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        crate::ops::storage_add(
            a.get_storage(),
            b.get_storage(),
            c.get_storage_mut(),
            idx_a,
            idx_b,
            idx_c,
            size,
        );
    }
    #[pyfunction]
    fn storage_sub(
        a: &StoragePtr,
        b: &StoragePtr,
        c: &mut StoragePtr,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        crate::ops::storage_sub(
            a.get_storage(),
            b.get_storage(),
            c.get_storage_mut(),
            idx_a,
            idx_b,
            idx_c,
            size,
        );
    }
    #[pyfunction]
    fn storage_mul(
        a: &StoragePtr,
        b: &StoragePtr,
        c: &mut StoragePtr,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        crate::ops::storage_mul(
            a.get_storage(),
            b.get_storage(),
            c.get_storage_mut(),
            idx_a,
            idx_b,
            idx_c,
            size,
        );
    }
    #[pyfunction]
    fn storage_div(
        a: &StoragePtr,
        b: &StoragePtr,
        c: &mut StoragePtr,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        crate::ops::storage_div(
            a.get_storage(),
            b.get_storage(),
            c.get_storage_mut(),
            idx_a,
            idx_b,
            idx_c,
            size,
        );
    }

    #[pyfunction]
    fn storage_sum(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        crate::ops::storage_sum(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }
    #[pyfunction]
    fn storage_max(a: &StoragePtr, b: &mut StoragePtr, idx_a: usize, idx_b: usize, size: usize) {
        crate::ops::storage_max(a.get_storage(), b.get_storage_mut(), idx_a, idx_b, size);
    }
}

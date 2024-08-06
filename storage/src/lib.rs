mod cpu_backend;

pub struct Storage {
    data: Vec<f32>,
    data_size: usize,
    ref_count: i16,
}

impl Storage {
    fn new(size: usize) -> Storage {
        Storage {
            data: vec![0.0; size as usize],
            data_size: size,
            ref_count: 1,
        }
    }
    fn storage_getitem(&self, idx: usize, size: usize) -> &[f32] {
        assert!(0 < size);
        assert!(idx + size <= self.data_size);
        self.data[idx..idx + size].as_ref()
    }
    fn storage_getitem_mut(&mut self, idx: usize, size: usize) -> &mut [f32] {
        assert!(0 < size);
        assert!(idx + size <= self.data_size);
        self.data[idx..idx + size].as_mut()
    }
    fn storage_incref(&mut self) {
        assert!(0 < self.ref_count);
        self.ref_count += 1;
    }
    fn storage_decref(&mut self) {
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

fn storage_relu(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    cpu_backend::unary_ops::relu(
        a.storage_getitem(idx_a, size), 
        b.storage_getitem_mut(idx_b, size),
    );
}

fn storage_add(a: &Storage, b: &Storage, c: &mut Storage, idx_a: usize, idx_b: usize, idx_c: usize, size: usize) {
    cpu_backend::binary_ops::add(
        a.storage_getitem(idx_a, size), 
        b.storage_getitem(idx_b, size), 
        c.storage_getitem_mut(idx_c, size),
    );
}
pub struct Storage {
    data: Vec<f32>,
    data_size: i64,
    ref_count: i8,
}

impl Storage {
    fn new(size: i64) -> Storage {
        assert!(size >= 0);
        Storage {
            data: vec![0.0; size as usize],
            data_size: size,
            ref_count: 1,
        }
    }
    fn storage_getitem(&self, idx: i64) -> f32 {
        assert!(idx >= 0 && idx < self.data_size);
        self.data[idx as usize]
    }
    fn storage_setitem(&mut self, idx: i64, value: f32) {
        assert!(idx >= 0 && idx < self.data_size);
        self.data[idx as usize] = value;
    }
    fn storage_incref(&mut self) {
        assert!(self.ref_count > 0);
        self.ref_count += 1;
    }
    fn storage_decref(&mut self) {
        assert!(self.ref_count > 0);
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

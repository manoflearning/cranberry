use crate::device::Device;
mod cpu_backend;

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
    pub fn device(&self) -> Device {
        self.device.clone()
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
    pub fn get_items(&self, idx: usize, size: usize) -> &[f32] {
        assert!(0 < size);
        assert!(idx + size <= self.data_size);
        self.data[idx..idx + size].as_ref()
    }
    pub fn get_items_mut(&mut self, idx: usize, size: usize) -> &mut [f32] {
        assert!(0 < size);
        assert!(idx + size <= self.data_size);
        self.data[idx..idx + size].as_mut()
    }
}

impl Storage {
    pub fn neg(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device() == b.device());
        match a.device() {
            Device::Cpu => {
                cpu_backend::unary_ops::neg(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
            }
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }
    pub fn sqrt(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device() == b.device());
        match a.device() {
            Device::Cpu => {
                cpu_backend::unary_ops::sqrt(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
            }
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }
    pub fn relu(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device() == b.device());
        match a.device() {
            Device::Cpu => {
                cpu_backend::unary_ops::relu(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
            }
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }
    pub fn exp(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device() == b.device());
        match a.device() {
            Device::Cpu => {
                cpu_backend::unary_ops::exp(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
            }
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }
    pub fn log(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device() == b.device());
        match a.device() {
            Device::Cpu => {
                cpu_backend::unary_ops::log(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
            }
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }

    pub fn add(
        a: &Storage,
        b: &Storage,
        c: &mut Storage,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        assert!(a.device() == b.device() && b.device() == c.device());
        match a.device() {
            Device::Cpu => cpu_backend::binary_ops::add(
                a.get_items(idx_a, size),
                b.get_items(idx_b, size),
                c.get_items_mut(idx_c, size),
            ),
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }

    pub fn sub(
        a: &Storage,
        b: &Storage,
        c: &mut Storage,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        assert!(a.device() == b.device() && b.device() == c.device());
        match a.device() {
            Device::Cpu => cpu_backend::binary_ops::sub(
                a.get_items(idx_a, size),
                b.get_items(idx_b, size),
                c.get_items_mut(idx_c, size),
            ),
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }

    pub fn mul(
        a: &Storage,
        b: &Storage,
        c: &mut Storage,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        assert!(a.device() == b.device() && b.device() == c.device());
        match a.device() {
            Device::Cpu => cpu_backend::binary_ops::mul(
                a.get_items(idx_a, size),
                b.get_items(idx_b, size),
                c.get_items_mut(idx_c, size),
            ),
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }

    pub fn div(
        a: &Storage,
        b: &Storage,
        c: &mut Storage,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        assert!(a.device() == b.device() && b.device() == c.device());
        match a.device() {
            Device::Cpu => cpu_backend::binary_ops::div(
                a.get_items(idx_a, size),
                b.get_items(idx_b, size),
                c.get_items_mut(idx_c, size),
            ),
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }

    pub fn sum(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device() == b.device());
        match a.device() {
            Device::Cpu => {
                cpu_backend::reduce_ops::sum(a.get_items(idx_a, size), b.get_items_mut(idx_b, 1))
            }
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }

    pub fn max(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device() == b.device());
        match a.device() {
            Device::Cpu => {
                cpu_backend::reduce_ops::max(a.get_items(idx_a, size), b.get_items_mut(idx_b, 1))
            }
            Device::Metal => unimplemented!(),
            Device::Cuda => unimplemented!(),
        }
    }
}
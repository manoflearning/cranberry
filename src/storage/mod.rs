use crate::device::Device;
mod cpu_backend;

#[cfg(test)]
mod tests;

#[derive(Clone, PartialEq)]
pub struct Storage {
    data: Vec<f32>,
    data_size: usize,
    device: Device,
    // ref_count: i32,
}

impl Storage {
    pub fn new(value: f32, size: usize, device: &str) -> Self {
        Storage {
            data: vec![value; size],
            data_size: size,
            device: Device::from_str(device),
            // ref_count: 1,
        }
    }
    pub fn from_vec(vec: Vec<f32>, device: &str) -> Self {
        Storage {
            data_size: vec.len(),
            data: vec,
            device: Device::from_str(device),
            // ref_count: 1,
        }
    }
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }
    pub fn device(&self) -> String {
        self.device.to_string()
    }
    pub fn size(&self) -> usize {
        self.data.len()
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
    #[inline(always)]
    pub fn neg(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device == b.device);
        match a.device {
            Device::Cpu => {
                cpu_backend::unary_ops::neg(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
            }
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
    #[inline(always)]
    pub fn sqrt(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device == b.device);
        match a.device {
            Device::Cpu => {
                cpu_backend::unary_ops::sqrt(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
            }
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
    #[inline(always)]
    pub fn exp(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device == b.device);
        match a.device {
            Device::Cpu => {
                cpu_backend::unary_ops::exp(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
            }
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
    #[inline(always)]
    pub fn log(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device == b.device);
        match a.device {
            Device::Cpu => {
                cpu_backend::unary_ops::log(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
            }
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
    #[inline(always)]
    pub fn add(
        a: &Storage,
        b: &Storage,
        c: &mut Storage,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        assert!(a.device == b.device && b.device == c.device);
        match a.device {
            Device::Cpu => cpu_backend::binary_ops::add(
                a.get_items(idx_a, size),
                b.get_items(idx_b, size),
                c.get_items_mut(idx_c, size),
            ),
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
    #[inline(always)]
    pub fn sub(
        a: &Storage,
        b: &Storage,
        c: &mut Storage,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        assert!(a.device == b.device && b.device == c.device);
        match a.device {
            Device::Cpu => cpu_backend::binary_ops::sub(
                a.get_items(idx_a, size),
                b.get_items(idx_b, size),
                c.get_items_mut(idx_c, size),
            ),
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
    #[inline(always)]
    pub fn mul(
        a: &Storage,
        b: &Storage,
        c: &mut Storage,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        assert!(a.device == b.device && b.device == c.device);
        match a.device {
            Device::Cpu => cpu_backend::binary_ops::mul(
                a.get_items(idx_a, size),
                b.get_items(idx_b, size),
                c.get_items_mut(idx_c, size),
            ),
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
    #[inline(always)]
    pub fn div(
        a: &Storage,
        b: &Storage,
        c: &mut Storage,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        assert!(a.device == b.device && b.device == c.device);
        match a.device {
            Device::Cpu => cpu_backend::binary_ops::div(
                a.get_items(idx_a, size),
                b.get_items(idx_b, size),
                c.get_items_mut(idx_c, size),
            ),
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
    pub fn cmplt(
        a: &Storage,
        b: &Storage,
        c: &mut Storage,
        idx_a: usize,
        idx_b: usize,
        idx_c: usize,
        size: usize,
    ) {
        assert!(a.device == b.device && b.device == c.device);
        match a.device {
            Device::Cpu => cpu_backend::binary_ops::cmplt(
                a.get_items(idx_a, size),
                b.get_items(idx_b, size),
                c.get_items_mut(idx_c, size),
            ),
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
    #[inline(always)]
    pub fn sum(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device == b.device);
        match a.device {
            Device::Cpu => {
                cpu_backend::reduce_ops::sum(a.get_items(idx_a, size), b.get_items_mut(idx_b, 1))
            }
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
    #[inline(always)]
    pub fn max(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
        assert!(a.device == b.device);
        match a.device {
            Device::Cpu => {
                cpu_backend::reduce_ops::max(a.get_items(idx_a, size), b.get_items_mut(idx_b, 1))
            }
            Device::Metal => todo!(),
            Device::Cuda => todo!(),
        }
    }
}

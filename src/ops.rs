use crate::{device::Device, storage_getitems, storage_getitems_mut, Storage};

mod cpu_backend;

pub fn storage_neg(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => cpu_backend::unary_ops::neg(
            storage_getitems(a, idx_a, size),
            storage_getitems_mut(b, idx_b, size),
        ),
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}
pub fn storage_sqrt(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => cpu_backend::unary_ops::sqrt(
            storage_getitems(a, idx_a, size),
            storage_getitems_mut(b, idx_b, size),
        ),
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}
pub fn storage_relu(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => cpu_backend::unary_ops::relu(
            storage_getitems(a, idx_a, size),
            storage_getitems_mut(b, idx_b, size),
        ),
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}
pub fn storage_exp(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => cpu_backend::unary_ops::exp(
            storage_getitems(a, idx_a, size),
            storage_getitems_mut(b, idx_b, size),
        ),
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}
pub fn storage_log(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => cpu_backend::unary_ops::log(
            storage_getitems(a, idx_a, size),
            storage_getitems_mut(b, idx_b, size),
        ),
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}

pub fn storage_add(
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
            storage_getitems(a, idx_a, size),
            storage_getitems(b, idx_b, size),
            storage_getitems_mut(c, idx_c, size),
        ),
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}

pub fn storage_sub(
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
            storage_getitems(a, idx_a, size),
            storage_getitems(b, idx_b, size),
            storage_getitems_mut(c, idx_c, size),
        ),
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}

pub fn storage_mul(
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
            storage_getitems(a, idx_a, size),
            storage_getitems(b, idx_b, size),
            storage_getitems_mut(c, idx_c, size),
        ),
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}

pub fn storage_div(
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
            storage_getitems(a, idx_a, size),
            storage_getitems(b, idx_b, size),
            storage_getitems_mut(c, idx_c, size),
        ),
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}

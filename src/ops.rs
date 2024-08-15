use crate::{device::Device, Storage};

mod cpu_backend;

pub fn neg(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => {
            cpu_backend::unary_ops::neg(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
        }
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}
pub fn sqrt(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => {
            cpu_backend::unary_ops::sqrt(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
        }
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}
pub fn relu(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => {
            cpu_backend::unary_ops::relu(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
        }
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}
pub fn exp(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => {
            cpu_backend::unary_ops::exp(a.get_items(idx_a, size), b.get_items_mut(idx_b, size))
        }
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}
pub fn log(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
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
    assert!(a.device == b.device && b.device == c.device);
    match a.device {
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
    assert!(a.device == b.device && b.device == c.device);
    match a.device {
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
    assert!(a.device == b.device && b.device == c.device);
    match a.device {
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
    assert!(a.device == b.device && b.device == c.device);
    match a.device {
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
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => {
            cpu_backend::reduce_ops::sum(a.get_items(idx_a, size), b.get_items_mut(idx_b, 1))
        }
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}

pub fn max(a: &Storage, b: &mut Storage, idx_a: usize, idx_b: usize, size: usize) {
    assert!(a.device == b.device);
    match a.device {
        Device::Cpu => {
            cpu_backend::reduce_ops::max(a.get_items(idx_a, size), b.get_items_mut(idx_b, 1))
        }
        Device::Metal => unimplemented!(),
        Device::Cuda => unimplemented!(),
    }
}

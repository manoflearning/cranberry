use crate::device::Device;

/// Owns the raw allocation backing one or more `View`s.
#[derive(Debug)]
pub struct StorageInner {
    pub(crate) data: Vec<f32>,
    pub(crate) device: Device,
}

impl StorageInner {
    pub fn new_full(value: f32, size: usize, device: Device) -> Self {
        Self {
            data: vec![value; size],
            device,
        }
    }

    pub fn from_vec(vec: Vec<f32>, device: Device) -> Self {
        Self { data: vec, device }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    #[inline]
    pub fn as_slice(&self, offset: usize, size: usize) -> &[f32] {
        &self.data[offset..offset + size]
    }

    #[inline]
    pub fn as_mut_slice(&mut self, offset: usize, size: usize) -> &mut [f32] {
        &mut self.data[offset..offset + size]
    }

    pub fn to_vec(&self, offset: usize, size: usize) -> Vec<f32> {
        self.as_slice(offset, size).to_vec()
    }
}

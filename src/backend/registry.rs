use once_cell::sync::OnceCell;

use super::{Backend, BackendError, BackendResult, CpuBackend, CudaBackend};
use crate::device::Device;

static CPU_BACKEND: CpuBackend = CpuBackend;
static CUDA_BACKEND: OnceCell<Result<CudaBackend, BackendError>> = OnceCell::new();

fn cuda_backend() -> BackendResult<&'static CudaBackend> {
    CUDA_BACKEND
        .get_or_init(|| CudaBackend::new())
        .as_ref()
        .map_err(Clone::clone)
}

pub fn cpu() -> &'static CpuBackend {
    &CPU_BACKEND
}

pub fn cuda() -> BackendResult<&'static CudaBackend> {
    cuda_backend()
}

pub fn get(device: Device) -> BackendResult<&'static dyn Backend> {
    match device {
        Device::Cpu => Ok(cpu() as &dyn Backend),
        Device::Cuda => cuda().map(|backend| backend as &dyn Backend),
        other => Err(BackendError::UnsupportedDevice(other)),
    }
}

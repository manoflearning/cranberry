#[cfg(feature = "cuda")]
use super::CudaBackend;
use super::{Backend, BackendError, BackendResult, CpuBackend};
use crate::device::Device;
#[cfg(feature = "cuda")]
use once_cell::sync::OnceCell;

static CPU_BACKEND: CpuBackend = CpuBackend;
#[cfg(feature = "cuda")]
static CUDA_BACKEND: OnceCell<Result<CudaBackend, BackendError>> = OnceCell::new();

#[cfg(feature = "cuda")]
fn cuda_backend() -> BackendResult<&'static CudaBackend> {
    CUDA_BACKEND
        .get_or_init(|| CudaBackend::new())
        .as_ref()
        .map_err(Clone::clone)
}

pub fn cpu() -> &'static CpuBackend {
    &CPU_BACKEND
}

#[cfg(feature = "cuda")]
pub fn cuda() -> BackendResult<&'static CudaBackend> {
    cuda_backend()
}

pub fn get(device: Device) -> BackendResult<&'static dyn Backend> {
    match device {
        Device::Cpu => Ok(cpu() as &dyn Backend),
        Device::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda().map(|backend| backend as &dyn Backend)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(BackendError::CudaUnavailable(
                    "crate compiled without the `cuda` feature".to_string(),
                ))
            }
        }
        other => Err(BackendError::UnsupportedDevice(other)),
    }
}

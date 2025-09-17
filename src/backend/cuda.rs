use std::{fmt, sync::Arc};

use cudarc::driver::{self, CudaContext, CudaSlice, CudaStream, LaunchConfig};

use super::{
    require_contiguous, require_same_numel, view_from_storage, Backend, BackendError,
    BackendResult, BinaryOp, UnaryOp,
};
use crate::core::{storage::StorageInner, view::View};
use crate::device::Device;

fn map_driver_err(err: driver::result::DriverError) -> BackendError {
    BackendError::Cuda(err.to_string())
}

mod kernels {
    use super::{map_driver_err, BackendError, BackendResult, BinaryOp, UnaryOp};
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use cudarc::nvrtc;
    use std::sync::Arc;

    pub(crate) struct Kernels {
        unary: unary::Module,
        binary: binary::Module,
    }

    impl Kernels {
        pub fn compile(ctx: &Arc<CudaContext>) -> Result<Self, BackendError> {
            // Produce PTX from the inline CUDA source.  NVRTC jit-compiles at runtime,
            // which keeps the crate self-contained and avoids a separate `nvcc` step.
            let ptx =
                nvrtc::compile_ptx(SOURCE).map_err(|err| BackendError::Cuda(err.to_string()))?;
            let module = ctx.load_module(ptx).map_err(map_driver_err)?;
            Ok(Self {
                unary: unary::Module::load(&module)?,
                binary: binary::Module::load(&module)?,
            })
        }

        pub fn launch_unary(
            &self,
            stream: &Arc<CudaStream>,
            op: UnaryOp,
            input: &CudaSlice<f32>,
            output: &mut CudaSlice<f32>,
            len: usize,
            cfg: LaunchConfig,
        ) -> BackendResult<()> {
            self.unary.launch(stream, op, input, output, len, cfg)
        }

        pub fn launch_binary(
            &self,
            stream: &Arc<CudaStream>,
            op: BinaryOp,
            lhs: &CudaSlice<f32>,
            rhs: &CudaSlice<f32>,
            output: &mut CudaSlice<f32>,
            len: usize,
            cfg: LaunchConfig,
        ) -> BackendResult<()> {
            self.binary.launch(stream, op, lhs, rhs, output, len, cfg)
        }
    }

    mod unary {
        use super::*;

        pub struct Module {
            neg: CudaFunction,
            sqrt: CudaFunction,
            exp: CudaFunction,
            log: CudaFunction,
            relu: CudaFunction,
        }

        impl Module {
            pub fn load(module: &Arc<CudaModule>) -> Result<Self, BackendError> {
                Ok(Self {
                    neg: module.load_function("unary_neg").map_err(map_driver_err)?,
                    sqrt: module.load_function("unary_sqrt").map_err(map_driver_err)?,
                    exp: module.load_function("unary_exp").map_err(map_driver_err)?,
                    log: module.load_function("unary_log").map_err(map_driver_err)?,
                    relu: module.load_function("unary_relu").map_err(map_driver_err)?,
                })
            }

            pub fn launch(
                &self,
                stream: &Arc<CudaStream>,
                op: UnaryOp,
                input: &CudaSlice<f32>,
                output: &mut CudaSlice<f32>,
                len: usize,
                cfg: LaunchConfig,
            ) -> BackendResult<()> {
                let func = match op {
                    UnaryOp::Neg => &self.neg,
                    UnaryOp::Sqrt => &self.sqrt,
                    UnaryOp::Exp => &self.exp,
                    UnaryOp::Log => &self.log,
                    UnaryOp::Relu => &self.relu,
                };
                unsafe {
                    stream
                        .launch_builder(func)
                        .arg(input)
                        .arg(output)
                        .arg(&len)
                        .launch(cfg)
                }
                .map(|_| ())
                .map_err(map_driver_err)
            }
        }
    }

    mod binary {
        use super::*;

        pub struct Module {
            add: CudaFunction,
            sub: CudaFunction,
            mul: CudaFunction,
            div: CudaFunction,
        }

        impl Module {
            pub fn load(module: &Arc<CudaModule>) -> Result<Self, BackendError> {
                Ok(Self {
                    add: module.load_function("binary_add").map_err(map_driver_err)?,
                    sub: module.load_function("binary_sub").map_err(map_driver_err)?,
                    mul: module.load_function("binary_mul").map_err(map_driver_err)?,
                    div: module.load_function("binary_div").map_err(map_driver_err)?,
                })
            }

            pub fn launch(
                &self,
                stream: &Arc<CudaStream>,
                op: BinaryOp,
                lhs: &CudaSlice<f32>,
                rhs: &CudaSlice<f32>,
                output: &mut CudaSlice<f32>,
                len: usize,
                cfg: LaunchConfig,
            ) -> BackendResult<()> {
                let func = match op {
                    BinaryOp::Add => &self.add,
                    BinaryOp::Sub => &self.sub,
                    BinaryOp::Mul => &self.mul,
                    BinaryOp::Div => &self.div,
                };
                unsafe {
                    stream
                        .launch_builder(func)
                        .arg(lhs)
                        .arg(rhs)
                        .arg(output)
                        .arg(&len)
                        .launch(cfg)
                }
                .map(|_| ())
                .map_err(map_driver_err)
            }
        }
    }

    /// CUDA kernels expressed as C strings compiled at runtime with NVRTC.
    const SOURCE: &str = r#"
// NVRTC treats this source as C++, so we pull in the headers that define the
// math intrinsics (`sqrtf`, `expf`, `logf`) and the `size_t` type that we use for
// element counts.
#include <stddef.h>
#include <math.h>

// Every kernel follows the exact same thread-indexing pattern: each CUDA thread
// owns a unique flat index into the logical tensor.  The math is identical for all
// kernels, so we declare a helper macro to make the generated PTX a little smaller
// and, just as importantly for new CUDA developers, to keep the thread math readable.
#define GLOBAL_INDEX() (blockIdx.x * blockDim.x + threadIdx.x)

extern "C" __global__ void unary_neg(const float* __restrict__ inp,
                                      float* __restrict__ out,
                                      size_t n) {
    const size_t idx = GLOBAL_INDEX();
    if (idx < n) {
        out[idx] = -inp[idx];
    }
}

extern "C" __global__ void unary_sqrt(const float* __restrict__ inp,
                                       float* __restrict__ out,
                                       size_t n) {
    const size_t idx = GLOBAL_INDEX();
    if (idx < n) {
        out[idx] = sqrtf(inp[idx]);
    }
}

extern "C" __global__ void unary_exp(const float* __restrict__ inp,
                                      float* __restrict__ out,
                                      size_t n) {
    const size_t idx = GLOBAL_INDEX();
    if (idx < n) {
        out[idx] = expf(inp[idx]);
    }
}

extern "C" __global__ void unary_log(const float* __restrict__ inp,
                                      float* __restrict__ out,
                                      size_t n) {
    const size_t idx = GLOBAL_INDEX();
    if (idx < n) {
        out[idx] = logf(inp[idx]);
    }
}

extern "C" __global__ void unary_relu(const float* __restrict__ inp,
                                       float* __restrict__ out,
                                       size_t n) {
    const size_t idx = GLOBAL_INDEX();
    if (idx < n) {
        const float v = inp[idx];
        out[idx] = v > 0.0f ? v : 0.0f;
    }
}

extern "C" __global__ void binary_add(const float* __restrict__ lhs,
                                       const float* __restrict__ rhs,
                                       float* __restrict__ out,
                                       size_t n) {
    const size_t idx = GLOBAL_INDEX();
    if (idx < n) {
        out[idx] = lhs[idx] + rhs[idx];
    }
}

extern "C" __global__ void binary_sub(const float* __restrict__ lhs,
                                       const float* __restrict__ rhs,
                                       float* __restrict__ out,
                                       size_t n) {
    const size_t idx = GLOBAL_INDEX();
    if (idx < n) {
        out[idx] = lhs[idx] - rhs[idx];
    }
}

extern "C" __global__ void binary_mul(const float* __restrict__ lhs,
                                       const float* __restrict__ rhs,
                                       float* __restrict__ out,
                                       size_t n) {
    const size_t idx = GLOBAL_INDEX();
    if (idx < n) {
        out[idx] = lhs[idx] * rhs[idx];
    }
}

extern "C" __global__ void binary_div(const float* __restrict__ lhs,
                                       const float* __restrict__ rhs,
                                       float* __restrict__ out,
                                       size_t n) {
    const size_t idx = GLOBAL_INDEX();
    if (idx < n) {
        out[idx] = lhs[idx] / rhs[idx];
    }
}
"#;
}

/// Holds the CUDA context, primary stream, and all kernel handles shared by GPU ops.
pub struct CudaBackend {
    stream: Arc<CudaStream>,
    kernels: kernels::Kernels,
}

impl fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device", &Device::Cuda)
            .finish()
    }
}

impl CudaBackend {
    /// Creates a new CUDA backend by initializing the context, default stream, and kernels.
    pub fn new() -> BackendResult<Self> {
        Self::init()
    }

    fn init() -> Result<Self, BackendError> {
        let device_count = CudaContext::device_count().map_err(map_driver_err)?;
        if device_count <= 0 {
            return Err(BackendError::CudaUnavailable(
                "no CUDA devices detected".to_string(),
            ));
        }

        // Acquire the primary context for device 0 and keep its default stream alive.
        // We intentionally pick device 0 for now; extending this to support multiple
        // GPUs just means plumbing an ordinal through the Python API and reusing the
        // exact same initialization procedure.
        let ctx = CudaContext::new(0).map_err(map_driver_err)?;
        let stream = ctx.default_stream();

        // Load the compiled kernels into the new context.
        let kernels = kernels::Kernels::compile(&ctx)?;

        Ok(CudaBackend { stream, kernels })
    }
}

impl CudaBackend {
    fn copy_to_device(&self, view: &View) -> BackendResult<CudaSlice<f32>> {
        // The storage currently lives in host memory.  Before we can launch a
        // kernel we need to stage that data on the device.  `memcpy_stod`
        // allocates the device buffer for us and schedules the host-to-device
        // transfer onto the stream we cached during initialization.
        let slice = view.inner.as_slice(view.offset, view.numel());
        self.stream.memcpy_stod(slice).map_err(map_driver_err)
    }

    fn copy_from_device(&self, slice: CudaSlice<f32>, shape: &[usize]) -> BackendResult<View> {
        // The kernel launch is asynchronous with respect to the host.  We force
        // completion here so that when we copy the results back we are guaranteed
        // to see the writes that the GPU just produced.
        self.stream.synchronize().map_err(map_driver_err)?;
        // Bring the freshly computed buffer back to the host.  The helper allocates
        // a `Vec<f32>` of the right size and performs a device-to-host copy.
        let host = self.stream.memcpy_dtov(&slice).map_err(map_driver_err)?;
        // Wrap the vector in our existing `StorageInner` abstraction so the rest
        // of the codebase can keep treating the buffer like any other contiguous
        // allocation.  We still tag it as `Device::Cuda` to preserve the logical
        // device placement that the caller requested.
        let inner = StorageInner::from_vec(host, Device::Cuda);
        Ok(view_from_storage(inner, shape))
    }
}

impl Backend for CudaBackend {
    fn unary(&self, op: UnaryOp, a: &View) -> BackendResult<View> {
        require_contiguous(a)?;

        // Upload the source tensor to device memory so the kernel can consume it.
        // We do this lazily on every invocation; a production implementation would
        // want to keep allocations around, but this keeps the control flow easy to
        // follow while still exercising the GPU cores.
        let device_in = self.copy_to_device(a)?;
        let mut device_out = self
            .stream
            .alloc_zeros::<f32>(a.numel())
            .map_err(map_driver_err)?;
        // `alloc_zeros` both reserves device memory and memset's it to 0.  The
        // zero-fill is not strictly required for our kernels, but it gives us a
        // well-defined value in case the GPU ever reads an out-of-bounds index
        // (which would indicate a bug elsewhere).

        // Precompute the launch geometry once per kernel dispatch.  The helper
        // chooses a 1024-thread block size and enough blocks to touch every
        // element, which works well for simple pointwise kernels.
        let cfg = LaunchConfig::for_num_elems(a.numel() as u32);
        let len = a.numel();

        self.kernels
            .launch_unary(&self.stream, op, &device_in, &mut device_out, len, cfg)?;

        self.copy_from_device(device_out, &a.shape)
    }

    fn binary(&self, op: BinaryOp, a: &View, b: &View) -> BackendResult<View> {
        require_contiguous(a)?;
        require_contiguous(b)?;
        require_same_numel(a, b)?;

        // Mirror the upload path we used for the unary kernels, but stage both
        // operands on the device.
        let device_a = self.copy_to_device(a)?;
        let device_b = self.copy_to_device(b)?;
        let mut device_out = self
            .stream
            .alloc_zeros::<f32>(a.numel())
            .map_err(map_driver_err)?;
        // The output buffer is shared across all binary launches to avoid
        // allocating multiple temporaries when we chain different kernels for the
        // same call.

        // Again, a single helper computes the dispatch geometry.
        let cfg = LaunchConfig::for_num_elems(a.numel() as u32);
        let len = a.numel();

        self.kernels.launch_binary(
            &self.stream,
            op,
            &device_a,
            &device_b,
            &mut device_out,
            len,
            cfg,
        )?;

        self.copy_from_device(device_out, &a.shape)
    }
}

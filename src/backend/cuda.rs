use std::{fmt, sync::Arc};

use cudarc::{
    driver::{self, CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc,
};
use once_cell::sync::OnceCell;

use super::{Backend, BackendError, BackendResult, BinaryOp, UnaryOp};
use crate::core::{
    storage::StorageInner,
    view::{contiguous_strides, View},
};
use crate::device::Device;

/// CUDA kernels expressed as C strings compiled at runtime with NVRTC.
const KERNEL_SOURCE: &str = r#"
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
    // Convert the 3-D CUDA launch geometry into a single linear index.
    const size_t idx = GLOBAL_INDEX();
    // Guard against the final, partially-filled block writing past the end.
    if (idx < n) {
        // Apply the unary operation in-place for this linear element.
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

/// Lazily constructed global CUDA backend.
///
/// We initialize everything on first use so that simply importing the Python
/// extension does not require the user to have a CUDA device installed.
static CUDA_BACKEND: OnceCell<Result<CudaBackend, BackendError>> = OnceCell::new();

fn map_driver_err(err: driver::result::DriverError) -> BackendError {
    BackendError::Cuda(err.to_string())
}

struct UnaryKernels {
    neg: CudaFunction,
    sqrt: CudaFunction,
    exp: CudaFunction,
    log: CudaFunction,
    relu: CudaFunction,
}

struct BinaryKernels {
    add: CudaFunction,
    sub: CudaFunction,
    mul: CudaFunction,
    div: CudaFunction,
}

/// Holds the CUDA context, primary stream, and all kernel handles shared by GPU ops.
pub struct CudaBackend {
    stream: Arc<CudaStream>,
    unary: UnaryKernels,
    binary: BinaryKernels,
}

impl fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device", &Device::Cuda)
            .finish()
    }
}

impl CudaBackend {
    /// Returns the process-wide CUDA backend instance, creating it on first use.
    pub fn global() -> BackendResult<&'static CudaBackend> {
        CUDA_BACKEND
            .get_or_init(init_backend)
            .as_ref()
            .map_err(Clone::clone)
    }
}

fn init_backend() -> Result<CudaBackend, BackendError> {
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

    // Compile the small collection of pointwise kernels.  NVRTC gives us a way to
    // embed plain CUDA C++ as a string and turn it into PTX at runtime, which keeps
    // the crate self-contained and removes the need for a separate `nvcc` build
    // step in the user environment.
    let ptx =
        nvrtc::compile_ptx(KERNEL_SOURCE).map_err(|err| BackendError::Cuda(err.to_string()))?;
    let module = ctx.load_module(ptx).map_err(map_driver_err)?;

    let unary = UnaryKernels {
        neg: module.load_function("unary_neg").map_err(map_driver_err)?,
        sqrt: module.load_function("unary_sqrt").map_err(map_driver_err)?,
        exp: module.load_function("unary_exp").map_err(map_driver_err)?,
        log: module.load_function("unary_log").map_err(map_driver_err)?,
        relu: module.load_function("unary_relu").map_err(map_driver_err)?,
    };

    let binary = BinaryKernels {
        add: module.load_function("binary_add").map_err(map_driver_err)?,
        sub: module.load_function("binary_sub").map_err(map_driver_err)?,
        mul: module.load_function("binary_mul").map_err(map_driver_err)?,
        div: module.load_function("binary_div").map_err(map_driver_err)?,
    };

    Ok(CudaBackend {
        stream,
        unary,
        binary,
    })
}

impl CudaBackend {
    fn ensure_contiguous(view: &View) -> BackendResult<()> {
        if view.is_contiguous() {
            Ok(())
        } else {
            // For now we only expose element-wise kernels, so we require buffers
            // to be densely packed in memory.  The Python binding already exposes
            // `.contiguous()` to users, so we surface the same error that the CPU
            // backend reports in the mismatch case.
            Err(BackendError::NotContiguous)
        }
    }

    fn to_device_slice(&self, view: &View) -> BackendResult<CudaSlice<f32>> {
        // The storage currently lives in host memory.  Before we can launch a
        // kernel we need to stage that data on the device.  `memcpy_stod`
        // allocates the device buffer for us and schedules the host-to-device
        // transfer onto the stream we cached during initialization.
        let slice = view.inner.as_slice(view.offset, view.numel());
        self.stream.memcpy_stod(slice).map_err(map_driver_err)
    }

    fn view_from_device_slice(
        &self,
        slice: CudaSlice<f32>,
        shape: &[usize],
    ) -> BackendResult<View> {
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
        Ok(View {
            inner: Arc::new(inner),
            offset: 0,
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
        })
    }
}

impl Backend for CudaBackend {
    fn unary(&self, op: UnaryOp, a: &View) -> BackendResult<View> {
        Self::ensure_contiguous(a)?;

        // Upload the source tensor to device memory so the kernel can consume it.
        // We do this lazily on every invocation; a production implementation would
        // want to keep allocations around, but this keeps the control flow easy to
        // follow while still exercising the GPU cores.
        let device_in = self.to_device_slice(a)?;
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

        // The launch builder consumes immutable or mutable references to CUDA
        // slices.  We build the argument list in the exact same way for every
        // unary op and only vary the function handle.
        let mut launch = |func: &CudaFunction| -> BackendResult<()> {
            // Launching work on a CUDA stream is inherently unsafe because the
            // compiler cannot prove that the raw pointers we pass stay valid for
            // the duration of the kernel.  Every argument we push here is owned by
            // a `CudaSlice`, so the buffers remain alive until the launch has
            // completed.
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&device_in)
                    .arg(&mut device_out)
                    .arg(&len)
                    .launch(cfg)
            }
            .map(|_| ())
            .map_err(map_driver_err)
        };

        match op {
            UnaryOp::Neg => launch(&self.unary.neg)?,
            UnaryOp::Sqrt => launch(&self.unary.sqrt)?,
            UnaryOp::Exp => launch(&self.unary.exp)?,
            UnaryOp::Log => launch(&self.unary.log)?,
            UnaryOp::Relu => launch(&self.unary.relu)?,
        }

        self.view_from_device_slice(device_out, &a.shape)
    }

    fn binary(&self, op: BinaryOp, a: &View, b: &View) -> BackendResult<View> {
        Self::ensure_contiguous(a)?;
        Self::ensure_contiguous(b)?;

        if a.numel() != b.numel() {
            return Err(BackendError::ShapeMismatch);
        }

        // Mirror the upload path we used for the unary kernels, but stage both
        // operands on the device.
        let device_a = self.to_device_slice(a)?;
        let device_b = self.to_device_slice(b)?;
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

        // Build-and-launch pipeline that mirrors the unary helper, this time
        // attaching both input buffers.
        let mut launch = |func: &CudaFunction| -> BackendResult<()> {
            // Same safety story as the unary path: we guarantee that the
            // `CudaSlice` lifetimes outlive the kernel execution so the raw
            // pointers remain valid for the GPU.
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&device_a)
                    .arg(&device_b)
                    .arg(&mut device_out)
                    .arg(&len)
                    .launch(cfg)
            }
            .map(|_| ())
            .map_err(map_driver_err)
        };

        match op {
            BinaryOp::Add => launch(&self.binary.add)?,
            BinaryOp::Sub => launch(&self.binary.sub)?,
            BinaryOp::Mul => launch(&self.binary.mul)?,
            BinaryOp::Div => launch(&self.binary.div)?,
        }

        self.view_from_device_slice(device_out, &a.shape)
    }
}

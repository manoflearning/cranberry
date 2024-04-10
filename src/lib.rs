#![feature(portable_simd)]
use pyo3::prelude::*;
use thread_priority::*;

mod tensor;
use tensor::Tensor;
mod data;
mod optim;
use optim::SGD;

#[pymodule]
fn cranberry(_py: Python, m: &PyModule) -> PyResult<()> {
    assert!(set_current_thread_priority(ThreadPriority::Min).is_ok());
    m.add_class::<Tensor>()?;
    m.add_class::<SGD>()?;
    Ok(())
}
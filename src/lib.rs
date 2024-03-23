#![feature(portable_simd)]
use pyo3::prelude::*;

mod tensor;
use tensor::Tensor;

#[pymodule]
fn cranberry(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    Ok(())
}
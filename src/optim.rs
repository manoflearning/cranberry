use crate::tensor::Tensor;
use pyo3::prelude::*;

#[pyclass]
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
}

#[pymethods]
impl SGD {
    #[new]
    fn new(params: Vec<Tensor>, lr: f32) -> PyResult<Self> { Ok(Self { params, lr }) }

    fn zero_grad(&self) { for param in &self.params { param.zero_grad_(); } }
    fn step(&self) { for param in &self.params { param.step_(self.lr); } }
}
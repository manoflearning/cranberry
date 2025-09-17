#![feature(portable_simd)]
#![cfg_attr(not(feature = "python"), allow(dead_code))]

mod backend;
mod core;
mod device;
#[cfg(feature = "python")]
mod py;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
#[pyo3::pymodule]
fn cranberry(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py::StorageView>()?;
    Ok(())
}

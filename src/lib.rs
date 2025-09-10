#![feature(portable_simd)]

mod backend;
mod core;
mod device;
mod py;

use pyo3::prelude::*;

#[cfg(test)]
mod tests;

#[pyo3::pymodule]
fn cranberry(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py::StorageView>()?;
    Ok(())
}

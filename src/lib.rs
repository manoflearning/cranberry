#![feature(portable_simd)]
#![feature(array_chunks)]

mod device;
mod storage;
mod storage_ptr;

use pyo3::prelude::*;

#[pyo3::pymodule]
fn cranberry(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<storage_ptr::StoragePtr>()?;
    Ok(())
}

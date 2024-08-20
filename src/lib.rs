#![feature(portable_simd)]
#![feature(array_chunks)]

mod device;
mod storage;
mod storage_ptr;

use pyo3::prelude::*;

#[pyo3::pymodule]
fn cranberry(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<storage_ptr::StoragePtr>()?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_full, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_from_vec, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_clone, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_drop, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_neg, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_exp, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_log, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_add, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_sub, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_mul, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_div, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_sum, m)?)?;
    m.add_function(wrap_pyfunction!(storage_ptr::storage_max, m)?)?;
    Ok(())
}

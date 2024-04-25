#![feature(portable_simd)]
use pyo3::prelude::*;
use thread_priority::*;

mod core;
use core::Core;

#[pymodule]
fn cranberry(_py: Python, m: &PyModule) -> PyResult<()> {
    assert!(set_current_thread_priority(ThreadPriority::Min).is_ok()); // is this really working?
    m.add_class::<Core>()?;
    Ok(())
}
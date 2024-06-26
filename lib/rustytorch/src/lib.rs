mod cpu;
mod opencl;
mod tensor;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn rustytorch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tensor::Tensor>()?;
    m.add_wrapped(wrap_pyfunction!(tensor::add_tensor))?;
    m.add_wrapped(wrap_pyfunction!(tensor::reshape_tensor))?;
    Ok(())
}

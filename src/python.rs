use crate::{convert_pt_to_safetensors, inspect_pt, ConvertOptions, ConvertResult, InspectionReport};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::Path;

fn into_py_error(err: impl std::fmt::Display) -> PyErr {
  PyRuntimeError::new_err(err.to_string())
}

fn to_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
  serde_json::to_string(value).map_err(into_py_error)
}

#[pyfunction]
fn inspect(input_pt: &str) -> PyResult<String> {
  let report: InspectionReport = inspect_pt(Path::new(input_pt)).map_err(into_py_error)?;
  to_json(&report)
}

#[pyfunction(
  signature = (
    input_pt,
    out_dir = "out",
    *,
    max_archive_bytes = None,
    max_tensor_count = None,
    max_tensor_bytes = None,
    max_pickle_bytes = None,
    strict_contiguous = None
  )
)]
fn convert(
  input_pt: &str,
  out_dir: &str,
  max_archive_bytes: Option<u64>,
  max_tensor_count: Option<usize>,
  max_tensor_bytes: Option<usize>,
  max_pickle_bytes: Option<usize>,
  strict_contiguous: Option<bool>,
) -> PyResult<String> {
  if input_pt.is_empty() {
    return Err(PyValueError::new_err("input_pt must not be empty"));
  }

  let mut opts = ConvertOptions::default();
  if let Some(value) = max_archive_bytes {
    opts.max_archive_bytes = value;
  }
  if let Some(value) = max_tensor_count {
    opts.max_tensor_count = value;
  }
  if let Some(value) = max_tensor_bytes {
    opts.max_tensor_bytes = value;
  }
  if let Some(value) = max_pickle_bytes {
    opts.max_pickle_bytes = value;
  }
  if let Some(value) = strict_contiguous {
    opts.strict_contiguous = value;
  }

  let result: ConvertResult =
    convert_pt_to_safetensors(Path::new(input_pt), Path::new(out_dir), opts).map_err(into_py_error)?;
  to_json(&result)
}

#[pymodule]
fn model_converter(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
  module.add_function(wrap_pyfunction!(inspect, module)?)?;
  module.add_function(wrap_pyfunction!(convert, module)?)?;
  module.add("__version__", env!("CARGO_PKG_VERSION"))?;
  Ok(())
}

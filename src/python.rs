use crate::{
  CheckpointMetadata, DType, ExportFormat, ExportOptions, LoadOptions, PtCheckpoint, ReconstructSource, TensorData,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyType};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn into_py_error(err: impl std::fmt::Display) -> PyErr {
  PyRuntimeError::new_err(err.to_string())
}

fn to_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
  serde_json::to_string(value).map_err(into_py_error)
}

fn build_load_options(
  input_pt: &str,
  max_archive_bytes: Option<u64>,
  max_tensor_count: Option<usize>,
  max_tensor_bytes: Option<usize>,
  max_pickle_bytes: Option<usize>,
  strict_contiguous: Option<bool>,
) -> PyResult<LoadOptions> {
  if input_pt.is_empty() {
    return Err(PyValueError::new_err("input_pt must not be empty"));
  }

  let mut opts = LoadOptions::default();
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

  Ok(opts)
}

fn tensors_to_py_dict<'py>(py: Python<'py>, tensors: &BTreeMap<String, TensorData>) -> PyResult<Bound<'py, PyDict>> {
  let output = PyDict::new(py);
  for (name, tensor) in tensors {
    let item = PyDict::new(py);
    item.set_item("dtype", tensor.dtype.as_safetensors())?;
    item.set_item("shape", tensor.shape.clone())?;
    item.set_item("data", PyBytes::new(py, &tensor.bytes))?;
    output.set_item(name, item)?;
  }
  Ok(output)
}

fn py_dict_to_tensors(state_dict: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, TensorData>> {
  let mut tensors = BTreeMap::new();
  for (key, value) in state_dict.iter() {
    let name = key.extract::<String>()?;
    let payload = value
      .cast::<PyDict>()
      .map_err(|_| PyValueError::new_err(format!("state_dict[{name}] must be a dict")))?;

    let dtype_str = payload
      .get_item("dtype")?
      .ok_or_else(|| PyValueError::new_err(format!("state_dict[{name}] missing dtype")))?
      .extract::<String>()?;
    let dtype = DType::from_safetensors(&dtype_str)
      .ok_or_else(|| PyValueError::new_err(format!("state_dict[{name}] has unsupported dtype {dtype_str}")))?;

    let shape = payload
      .get_item("shape")?
      .ok_or_else(|| PyValueError::new_err(format!("state_dict[{name}] missing shape")))?
      .extract::<Vec<usize>>()?;

    let data = payload
      .get_item("data")?
      .ok_or_else(|| PyValueError::new_err(format!("state_dict[{name}] missing data")))?
      .extract::<Vec<u8>>()?;

    tensors.insert(
      name,
      TensorData {
        dtype,
        shape,
        bytes: data,
      },
    );
  }
  Ok(tensors)
}

#[pyclass(name = "PtCheckpointCore")]
pub struct PyPtCheckpoint {
  inner: PtCheckpoint,
}

#[pymethods]
impl PyPtCheckpoint {
  #[classmethod]
  #[pyo3(
    signature = (
      input_pt,
      *,
      max_archive_bytes = None,
      max_tensor_count = None,
      max_tensor_bytes = None,
      max_pickle_bytes = None,
      strict_contiguous = None
    )
  )]
  fn load(
    _cls: &Bound<'_, PyType>,
    input_pt: &str,
    max_archive_bytes: Option<u64>,
    max_tensor_count: Option<usize>,
    max_tensor_bytes: Option<usize>,
    max_pickle_bytes: Option<usize>,
    strict_contiguous: Option<bool>,
  ) -> PyResult<Self> {
    let opts = build_load_options(
      input_pt,
      max_archive_bytes,
      max_tensor_count,
      max_tensor_bytes,
      max_pickle_bytes,
      strict_contiguous,
    )?;
    let inner = PtCheckpoint::load(input_pt, opts).map_err(into_py_error)?;
    Ok(Self { inner })
  }

  #[classmethod]
  #[pyo3(signature = (metadata_yaml, *, weights_path = None, state_dict = None))]
  fn from_metadata_yaml(
    _cls: &Bound<'_, PyType>,
    metadata_yaml: &str,
    weights_path: Option<&str>,
    state_dict: Option<&Bound<'_, PyDict>>,
  ) -> PyResult<Self> {
    let metadata: CheckpointMetadata = serde_yaml::from_str(metadata_yaml).map_err(into_py_error)?;

    let source = match (weights_path, state_dict) {
      (Some(path), None) => ReconstructSource::WeightsFile(path.into()),
      (None, Some(values)) => ReconstructSource::StateDict(py_dict_to_tensors(values)?),
      (Some(_), Some(_)) => {
        return Err(PyValueError::new_err(
          "provide either weights_path or state_dict, not both",
        ));
      }
      (None, None) => {
        return Err(PyValueError::new_err("one of weights_path or state_dict is required"));
      }
    };

    let inner = PtCheckpoint::from_metadata(metadata, source).map_err(into_py_error)?;
    Ok(Self { inner })
  }

  #[pyo3(
    signature = (
      out_dir,
      *,
      format = "safetensors",
      weights_filename = "model.safetensors",
      metadata_filename = "model.yaml",
      include_metadata = true,
      overwrite = false
    )
  )]
  fn export_json(
    &self,
    out_dir: &str,
    format: &str,
    weights_filename: &str,
    metadata_filename: &str,
    include_metadata: bool,
    overwrite: bool,
  ) -> PyResult<String> {
    let export_format = match format {
      "safetensors" => ExportFormat::Safetensors,
      other => {
        return Err(PyValueError::new_err(format!(
          "unsupported format {other}; expected 'safetensors'"
        )));
      }
    };

    let mut opts = ExportOptions::new(export_format, None);
    opts.weights_filename = PathBuf::from(weights_filename);
    opts.metadata_filename = PathBuf::from(metadata_filename);
    opts.include_metadata = include_metadata;
    opts.overwrite = overwrite;

    let result = self.inner.export(out_dir, opts).map_err(into_py_error)?;
    to_json(&result)
  }

  fn metadata_json(&self) -> PyResult<String> {
    to_json(self.inner.metadata())
  }

  fn state_dict_raw<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    tensors_to_py_dict(py, self.inner.raw_tensors())
  }

  fn source_sha256(&self) -> String {
    self.inner.source_sha256().to_string()
  }

  fn tensor_count(&self) -> usize {
    self.inner.tensor_count()
  }

  fn __repr__(&self) -> String {
    format!(
      "PtCheckpointCore(tensor_count={}, source_sha256='{}')",
      self.inner.tensor_count(),
      self.inner.source_sha256()
    )
  }
}

#[pymodule]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
  module.add_class::<PyPtCheckpoint>()?;
  module.add("__version__", env!("CARGO_PKG_VERSION"))?;
  Ok(())
}

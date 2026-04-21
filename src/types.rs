use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct LoadOptions {
  pub max_archive_bytes: u64,
  pub max_tensor_count: usize,
  pub max_tensor_bytes: usize,
  pub max_pickle_bytes: usize,
  pub strict_contiguous: bool,
  pub state_dict_root_keys: Vec<String>,
  pub state_dict_root_strict: bool,
}

impl Default for LoadOptions {
  fn default() -> Self {
    Self {
      max_archive_bytes: 4 * 1024 * 1024 * 1024,
      max_tensor_count: 4096,
      max_tensor_bytes: 1024 * 1024 * 1024,
      max_pickle_bytes: 64 * 1024 * 1024,
      strict_contiguous: true,
      state_dict_root_keys: Vec::new(),
      state_dict_root_strict: true,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
  Safetensors,
}

impl ExportFormat {
  pub fn extension(self) -> &'static str {
    match self {
      ExportFormat::Safetensors => "safetensors",
    }
  }
}

#[derive(Debug, Clone)]
pub struct ExportOptions {
  pub format: ExportFormat,
  pub weights_filename: PathBuf,
  pub metadata_filename: PathBuf,
  pub include_metadata: bool,
  pub overwrite: bool,
}

impl ExportOptions {
  pub fn new(format: ExportFormat, input_path: Option<&Path>) -> Self {
    let weights_filename = default_weights_filename(format, input_path);
    let metadata_filename = weights_filename.with_extension("yaml");

    Self {
      format,
      weights_filename,
      metadata_filename,
      include_metadata: true,
      overwrite: false,
    }
  }
}

fn default_weights_filename(format: ExportFormat, input_path: Option<&Path>) -> PathBuf {
  let ext = format.extension();
  let Some(path) = input_path else {
    return PathBuf::from(format!("model.{ext}"));
  };

  let Some(name) = path.file_name() else {
    return PathBuf::from(format!("model.{ext}"));
  };

  Path::new(name).with_extension(ext)
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportResult {
  pub weights_path: PathBuf,
  #[serde(default)]
  pub weights_paths: BTreeMap<String, PathBuf>,
  pub metadata_path: Option<PathBuf>,
  pub source_sha256: String,
  pub tensor_count: usize,
  pub total_tensor_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointTensorMetadata {
  pub name: String,
  pub dtype: String,
  pub shape: Vec<usize>,
  pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSecurity {
  #[serde(default)]
  pub objects: Vec<String>,
  #[serde(default)]
  pub calls: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
  pub format_version: usize,
  pub source_file: String,
  pub source_sha256: String,
  pub safetensors_file: String,
  #[serde(default)]
  pub safetensors_files: BTreeMap<String, String>,
  pub created_at_unix: u64,
  pub tensor_count: usize,
  pub total_tensor_bytes: usize,
  #[serde(default)]
  pub metadata: serde_yaml::Value,
  pub security: CheckpointSecurity,
  pub tensors: TensorManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TensorManifest {
  List(Vec<CheckpointTensorMetadata>),
  ByRoot(BTreeMap<String, Vec<CheckpointTensorMetadata>>),
}

#[derive(Debug, Clone)]
pub enum ReconstructSource {
  WeightsFile(PathBuf),
  StateDict(BTreeMap<String, TensorData>),
}

#[derive(Debug, Clone)]
pub enum TensorArray {
  F32(ArrayD<f32>),
  F64(ArrayD<f64>),
  I8(ArrayD<i8>),
  I16(ArrayD<i16>),
  I32(ArrayD<i32>),
  I64(ArrayD<i64>),
  U8(ArrayD<u8>),
  Bool(ArrayD<bool>),
}

#[derive(Debug)]
pub enum ConvertError {
  Io(std::io::Error),
  Zip(zip::result::ZipError),
  Json(serde_json::Error),
  Yaml(serde_yaml::Error),
  Ndarray(ndarray::ShapeError),
  UnsupportedFormat(String),
  UnsafeOpcode { opcode: u8, offset: usize },
  InvalidStructure(String),
  ResourceLimitExceeded(String),
}

impl fmt::Display for ConvertError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      ConvertError::Io(err) => write!(f, "io error: {}", err),
      ConvertError::Zip(err) => write!(f, "zip error: {}", err),
      ConvertError::Json(err) => write!(f, "json error: {}", err),
      ConvertError::Yaml(err) => write!(f, "yaml error: {}", err),
      ConvertError::Ndarray(err) => write!(f, "ndarray error: {}", err),
      ConvertError::UnsupportedFormat(msg) => write!(f, "unsupported format: {}", msg),
      ConvertError::UnsafeOpcode { opcode, offset } => {
        write!(f, "unsafe/unsupported pickle opcode 0x{opcode:02x} at offset {offset}")
      }
      ConvertError::InvalidStructure(msg) => {
        write!(f, "invalid checkpoint structure: {}", msg)
      }
      ConvertError::ResourceLimitExceeded(msg) => {
        write!(f, "resource limit exceeded: {}", msg)
      }
    }
  }
}

impl std::error::Error for ConvertError {}

impl From<std::io::Error> for ConvertError {
  fn from(value: std::io::Error) -> Self {
    Self::Io(value)
  }
}

impl From<zip::result::ZipError> for ConvertError {
  fn from(value: zip::result::ZipError) -> Self {
    Self::Zip(value)
  }
}

impl From<serde_json::Error> for ConvertError {
  fn from(value: serde_json::Error) -> Self {
    Self::Json(value)
  }
}

impl From<serde_yaml::Error> for ConvertError {
  fn from(value: serde_yaml::Error) -> Self {
    Self::Yaml(value)
  }
}

impl From<ndarray::ShapeError> for ConvertError {
  fn from(value: ndarray::ShapeError) -> Self {
    Self::Ndarray(value)
  }
}

pub type Result<T> = std::result::Result<T, ConvertError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
  F16,
  BF16,
  F32,
  F64,
  I8,
  I16,
  I32,
  I64,
  U8,
  Bool,
}

impl DType {
  pub fn elem_size(self) -> usize {
    match self {
      DType::F16 | DType::BF16 | DType::I16 => 2,
      DType::F32 | DType::I32 => 4,
      DType::F64 | DType::I64 => 8,
      DType::I8 | DType::U8 | DType::Bool => 1,
    }
  }

  pub fn as_safetensors(self) -> &'static str {
    match self {
      DType::F16 => "F16",
      DType::BF16 => "BF16",
      DType::F32 => "F32",
      DType::F64 => "F64",
      DType::I8 => "I8",
      DType::I16 => "I16",
      DType::I32 => "I32",
      DType::I64 => "I64",
      DType::U8 => "U8",
      DType::Bool => "BOOL",
    }
  }

  pub fn from_safetensors(value: &str) -> Option<Self> {
    match value {
      "F16" => Some(DType::F16),
      "BF16" => Some(DType::BF16),
      "F32" => Some(DType::F32),
      "F64" => Some(DType::F64),
      "I8" => Some(DType::I8),
      "I16" => Some(DType::I16),
      "I32" => Some(DType::I32),
      "I64" => Some(DType::I64),
      "U8" => Some(DType::U8),
      "BOOL" => Some(DType::Bool),
      _ => None,
    }
  }
}

#[derive(Debug, Clone)]
pub struct StorageRef {
  pub key: String,
  pub dtype: DType,
  pub size_elems: usize,
}

#[derive(Debug, Clone)]
pub struct TensorRef {
  pub storage: StorageRef,
  pub offset_elems: usize,
  pub shape: Vec<usize>,
  pub stride: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct TensorData {
  pub dtype: DType,
  pub shape: Vec<usize>,
  pub bytes: Vec<u8>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Value {
  Marker,
  None,
  Bool(bool),
  Int(i64),
  Float(f64),
  String(String),
  Bytes(Vec<u8>),
  List(Vec<Value>),
  Set(Vec<Value>),
  Tuple(Vec<Value>),
  Dict(Vec<(Value, Value)>),
  Global {
    module: String,
    name: String,
  },
  StorageRef(StorageRef),
  TensorRef(TensorRef),
  OrderedDict(Vec<(String, Value)>),
  Call {
    func: String,
    args: Vec<Value>,
    state: Option<Box<Value>>,
  },
  Object {
    module: String,
    name: String,
    args: Vec<Value>,
    state: Option<Box<Value>>,
  },
}

impl Value {
  pub(crate) fn as_usize(&self) -> Result<usize> {
    match self {
      Value::Int(v) if *v >= 0 => Ok(*v as usize),
      _ => Err(ConvertError::InvalidStructure(
        "expected non-negative integer".to_string(),
      )),
    }
  }

  pub(crate) fn as_string(&self) -> Result<String> {
    match self {
      Value::String(v) => Ok(v.clone()),
      Value::Int(v) => Ok(v.to_string()),
      _ => Err(ConvertError::InvalidStructure("expected string".to_string())),
    }
  }

  pub(crate) fn as_usize_vec(&self) -> Result<Vec<usize>> {
    match self {
      Value::Tuple(items) | Value::List(items) => items.iter().map(Value::as_usize).collect(),
      _ => Err(ConvertError::InvalidStructure(
        "expected tuple/list of integers".to_string(),
      )),
    }
  }
}

pub struct ParsedCheckpoint {
  pub source_sha256: String,
  pub warnings: Vec<String>,
  pub tensors: BTreeMap<String, TensorData>,
  pub tensor_groups: BTreeMap<String, BTreeMap<String, TensorData>>,
  pub metadata: serde_yaml::Value,
  pub security: CheckpointSecurity,
}

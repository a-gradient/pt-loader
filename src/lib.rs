mod extract;
mod iohash;
mod metadata;
mod parser;
#[cfg(feature = "pyo3")]
mod python;
mod types;
pub mod writer;

pub use types::{
  CheckpointMetadata, CheckpointTensorMetadata, ConvertError, DType, ExportFormat, ExportOptions, ExportResult,
  LoadOptions, ReconstructSource, Result, StorageRef, TensorArray, TensorData, TensorRef, Value,
};

use ndarray::{ArrayD, IxDyn};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use zip::read::ZipArchive;

use extract::{contiguous_stride, extract_state_dict_tensors, numel};
use iohash::{find_data_pkl_name, read_storage_blob, read_zip_entry, sha256_file, sha256_hex};
use metadata::{collect_constructor_types, project_root_metadata};
use parser::parse_pickle;
use types::ParsedCheckpoint;
use writer::{write_metadata_yaml, write_safetensors};

#[derive(Debug, Clone)]
pub struct PtCheckpoint {
  source_sha256: String,
  warnings: Vec<String>,
  metadata: CheckpointMetadata,
  tensors: BTreeMap<String, TensorData>,
}

impl PtCheckpoint {
  pub fn from_pt(path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
    let path = path.as_ref();
    let parsed = parse_checkpoint(path, &opts)?;
    let metadata = build_checkpoint_metadata(
      path.display().to_string(),
      parsed.source_sha256.clone(),
      &parsed.metadata,
      &parsed.objects,
      &parsed.tensors,
      "model.safetensors".to_string(),
    );

    Ok(Self {
      source_sha256: parsed.source_sha256,
      warnings: parsed.warnings,
      metadata,
      tensors: parsed.tensors,
    })
  }

  pub fn from_metadata(metadata: CheckpointMetadata, source: ReconstructSource) -> Result<Self> {
    let tensors = match source {
      ReconstructSource::WeightsFile(path) => read_safetensors_tensors(&path)?,
      ReconstructSource::StateDict(values) => values,
    };

    validate_metadata_against_tensors(&metadata, &tensors)?;

    Ok(Self {
      source_sha256: metadata.source_sha256.clone(),
      warnings: Vec::new(),
      metadata,
      tensors,
    })
  }

  pub fn metadata(&self) -> &CheckpointMetadata {
    &self.metadata
  }

  pub fn source_sha256(&self) -> &str {
    &self.source_sha256
  }

  pub fn warnings(&self) -> &[String] {
    &self.warnings
  }

  pub fn tensor_count(&self) -> usize {
    self.tensors.len()
  }

  #[cfg(feature = "pyo3")]
  pub(crate) fn raw_tensors(&self) -> &BTreeMap<String, TensorData> {
    &self.tensors
  }

  pub fn state_dict(&self) -> Result<BTreeMap<String, TensorArray>> {
    let mut out = BTreeMap::new();
    for (name, tensor) in &self.tensors {
      out.insert(name.clone(), tensor_data_to_array(tensor)?);
    }
    Ok(out)
  }

  pub fn export(&self, out_dir: impl AsRef<Path>, opts: ExportOptions) -> Result<ExportResult> {
    match opts.format {
      ExportFormat::Safetensors => {}
    }

    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let weights_path = out_dir.join(&opts.weights_filename);
    if weights_path.exists() && !opts.overwrite {
      return Err(ConvertError::InvalidStructure(format!(
        "output already exists: {}",
        weights_path.display()
      )));
    }

    write_safetensors(&weights_path, &self.tensors, &self.source_sha256)?;

    let metadata_path = if opts.include_metadata {
      let metadata_path = out_dir.join(&opts.metadata_filename);
      if metadata_path.exists() && !opts.overwrite {
        return Err(ConvertError::InvalidStructure(format!(
          "output already exists: {}",
          metadata_path.display()
        )));
      }

      let mut metadata = self.metadata.clone();
      metadata.safetensors_file = opts.weights_filename.clone();
      metadata.created_at_unix = now_unix_secs();
      metadata.tensor_count = self.tensors.len();
      metadata.total_tensor_bytes = total_tensor_bytes(&self.tensors);
      metadata.tensors = tensor_summaries_for_metadata(&self.tensors);
      write_metadata_yaml(&metadata_path, &metadata)?;
      Some(metadata_path)
    } else {
      None
    };

    Ok(ExportResult {
      weights_path,
      metadata_path,
      source_sha256: self.source_sha256.clone(),
      tensor_count: self.tensors.len(),
      total_tensor_bytes: total_tensor_bytes(&self.tensors),
    })
  }
}

pub(crate) fn parse_checkpoint(path: &Path, opts: &LoadOptions) -> Result<ParsedCheckpoint> {
  let file = File::open(path)?;
  let metadata = file.metadata()?;
  if metadata.len() > opts.max_archive_bytes {
    return Err(ConvertError::ResourceLimitExceeded(format!(
      "archive is {} bytes, limit is {}",
      metadata.len(),
      opts.max_archive_bytes
    )));
  }

  let mut magic = [0u8; 4];
  let mut fh = File::open(path)?;
  fh.read_exact(&mut magic)?;
  if magic != [0x50, 0x4b, 0x03, 0x04] {
    return Err(ConvertError::UnsupportedFormat(
      "only torch zip checkpoints are supported (legacy raw-pickle .pt is rejected)".to_string(),
    ));
  }

  let source_sha256 = sha256_file(path)?;
  let mut archive = ZipArchive::new(file)?;
  let data_pkl_name = find_data_pkl_name(&mut archive)?;
  let prefix = data_pkl_name
    .strip_suffix("data.pkl")
    .ok_or_else(|| ConvertError::InvalidStructure("invalid data.pkl entry name".to_string()))?
    .to_string();
  let pickle_bytes = read_zip_entry(&mut archive, &data_pkl_name)?;
  if pickle_bytes.len() > opts.max_pickle_bytes {
    return Err(ConvertError::ResourceLimitExceeded(format!(
      "data.pkl is {} bytes, limit is {}",
      pickle_bytes.len(),
      opts.max_pickle_bytes
    )));
  }

  let root = parse_pickle(&pickle_bytes, opts)?;
  let metadata = project_root_metadata(&root);
  let objects = collect_constructor_types(&root);
  let tensor_refs = extract_state_dict_tensors(&root)?;
  if tensor_refs.is_empty() {
    return Err(ConvertError::InvalidStructure(
      "no tensors found in checkpoint state_dict".to_string(),
    ));
  }
  if tensor_refs.len() > opts.max_tensor_count {
    return Err(ConvertError::ResourceLimitExceeded(format!(
      "tensor count {} exceeds limit {}",
      tensor_refs.len(),
      opts.max_tensor_count
    )));
  }

  let mut storage_blobs: HashMap<String, Vec<u8>> = HashMap::new();
  for tensor in tensor_refs.values() {
    let key = &tensor.storage.key;
    if storage_blobs.contains_key(key) {
      continue;
    }
    let blob = read_storage_blob(&mut archive, &prefix, key)?;
    let required_bytes = tensor.storage.size_elems * tensor.storage.dtype.elem_size();
    if blob.len() < required_bytes {
      return Err(ConvertError::InvalidStructure(format!(
        "storage {} has {} bytes, expected at least {}",
        key,
        blob.len(),
        required_bytes
      )));
    }
    storage_blobs.insert(key.clone(), blob);
  }

  let mut tensors = BTreeMap::new();
  for (name, tensor_ref) in tensor_refs {
    if opts.strict_contiguous {
      let expected = contiguous_stride(&tensor_ref.shape);
      if expected != tensor_ref.stride {
        return Err(ConvertError::InvalidStructure(format!(
          "tensor {} has non-contiguous stride {:?}, expected {:?}",
          name, tensor_ref.stride, expected
        )));
      }
    }

    let elem_size = tensor_ref.storage.dtype.elem_size();
    let numel = numel(&tensor_ref.shape)?;
    let start = tensor_ref
      .offset_elems
      .checked_mul(elem_size)
      .ok_or_else(|| ConvertError::InvalidStructure("tensor byte offset overflow".to_string()))?;
    let byte_len = numel
      .checked_mul(elem_size)
      .ok_or_else(|| ConvertError::InvalidStructure("tensor byte length overflow".to_string()))?;
    if byte_len > opts.max_tensor_bytes {
      return Err(ConvertError::ResourceLimitExceeded(format!(
        "tensor {} is {} bytes, limit is {}",
        name, byte_len, opts.max_tensor_bytes
      )));
    }
    let end = start
      .checked_add(byte_len)
      .ok_or_else(|| ConvertError::InvalidStructure("tensor slice overflow".to_string()))?;

    let storage = storage_blobs
      .get(&tensor_ref.storage.key)
      .ok_or_else(|| ConvertError::InvalidStructure(format!("missing storage blob {}", tensor_ref.storage.key)))?;
    if end > storage.len() {
      return Err(ConvertError::InvalidStructure(format!(
        "tensor {} slice [{}, {}) is out of storage bounds {}",
        name,
        start,
        end,
        storage.len()
      )));
    }

    let raw = storage[start..end].to_vec();
    let normalized = normalize_tensor_dtype(tensor_ref.storage.dtype, tensor_ref.shape, raw)?;
    tensors.insert(name, normalized);
  }

  Ok(ParsedCheckpoint {
    source_sha256,
    warnings: Vec::new(),
    tensors,
    metadata,
    objects,
  })
}

fn build_checkpoint_metadata(
  source_file: String,
  source_sha256: String,
  metadata: &serde_yaml::Value,
  objects: &[String],
  tensors: &BTreeMap<String, TensorData>,
  safetensors_file: String,
) -> CheckpointMetadata {
  CheckpointMetadata {
    format_version: 1,
    source_file,
    source_sha256,
    safetensors_file,
    created_at_unix: now_unix_secs(),
    tensor_count: tensors.len(),
    total_tensor_bytes: total_tensor_bytes(tensors),
    metadata: metadata.clone(),
    objects: objects.to_vec(),
    tensors: tensor_summaries_for_metadata(tensors),
  }
}

fn tensor_summaries_for_metadata(tensors: &BTreeMap<String, TensorData>) -> Vec<CheckpointTensorMetadata> {
  tensors
    .iter()
    .map(|(name, tensor)| CheckpointTensorMetadata {
      name: name.clone(),
      dtype: tensor.dtype.as_safetensors().to_string(),
      shape: tensor.shape.clone(),
      sha256: sha256_hex(&tensor.bytes),
    })
    .collect()
}

fn total_tensor_bytes(tensors: &BTreeMap<String, TensorData>) -> usize {
  tensors.values().map(|tensor| tensor.bytes.len()).sum()
}

fn now_unix_secs() -> u64 {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|value| value.as_secs())
    .unwrap_or(0)
}

fn validate_metadata_against_tensors(
  metadata: &CheckpointMetadata,
  tensors: &BTreeMap<String, TensorData>,
) -> Result<()> {
  if metadata.tensor_count != tensors.len() {
    return Err(ConvertError::InvalidStructure(format!(
      "metadata tensor_count={} does not match loaded tensor count={}",
      metadata.tensor_count,
      tensors.len()
    )));
  }

  let tensor_bytes = total_tensor_bytes(tensors);
  if metadata.total_tensor_bytes != tensor_bytes {
    return Err(ConvertError::InvalidStructure(format!(
      "metadata total_tensor_bytes={} does not match loaded tensor bytes={}",
      metadata.total_tensor_bytes, tensor_bytes
    )));
  }

  for item in &metadata.tensors {
    let Some(tensor) = tensors.get(&item.name) else {
      return Err(ConvertError::InvalidStructure(format!(
        "metadata references missing tensor {}",
        item.name
      )));
    };
    if item.dtype != tensor.dtype.as_safetensors() {
      return Err(ConvertError::InvalidStructure(format!(
        "metadata dtype mismatch for {}: {} != {}",
        item.name,
        item.dtype,
        tensor.dtype.as_safetensors()
      )));
    }
    if item.shape != tensor.shape {
      return Err(ConvertError::InvalidStructure(format!(
        "metadata shape mismatch for {}",
        item.name
      )));
    }
    if item.sha256 != sha256_hex(&tensor.bytes) {
      return Err(ConvertError::InvalidStructure(format!(
        "metadata sha256 mismatch for {}",
        item.name
      )));
    }
  }

  Ok(())
}

#[derive(Debug, Deserialize)]
struct SafetensorHeaderEntry {
  dtype: String,
  shape: Vec<usize>,
  data_offsets: [usize; 2],
}

fn read_safetensors_tensors(path: &Path) -> Result<BTreeMap<String, TensorData>> {
  let file_bytes = fs::read(path)?;
  if file_bytes.len() < 8 {
    return Err(ConvertError::InvalidStructure(
      "safetensors file is too short".to_string(),
    ));
  }

  let header_len = u64::from_le_bytes(file_bytes[0..8].try_into().expect("8-byte header"));
  let header_len = header_len as usize;
  if file_bytes.len() < 8 + header_len {
    return Err(ConvertError::InvalidStructure(
      "safetensors header is truncated".to_string(),
    ));
  }

  let header_bytes = &file_bytes[8..8 + header_len];
  let data = &file_bytes[8 + header_len..];
  let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header_bytes)?;

  let mut tensors = BTreeMap::new();
  for (name, value) in header {
    if name == "__metadata__" {
      continue;
    }
    let entry: SafetensorHeaderEntry = serde_json::from_value(value)?;
    let Some(dtype) = DType::from_safetensors(&entry.dtype) else {
      return Err(ConvertError::InvalidStructure(format!(
        "unsupported safetensors dtype {}",
        entry.dtype
      )));
    };

    let start = entry.data_offsets[0];
    let end = entry.data_offsets[1];
    if end < start || end > data.len() {
      return Err(ConvertError::InvalidStructure(format!(
        "invalid data_offsets for tensor {}",
        name
      )));
    }

    let expected_size = numel(&entry.shape)?
      .checked_mul(dtype.elem_size())
      .ok_or_else(|| ConvertError::InvalidStructure("tensor byte length overflow".to_string()))?;
    if end - start != expected_size {
      return Err(ConvertError::InvalidStructure(format!(
        "tensor {} bytes mismatch: {} != {}",
        name,
        end - start,
        expected_size
      )));
    }

    tensors.insert(
      name,
      TensorData {
        dtype,
        shape: entry.shape,
        bytes: data[start..end].to_vec(),
      },
    );
  }

  if tensors.is_empty() {
    return Err(ConvertError::InvalidStructure(
      "no tensors found in safetensors file".to_string(),
    ));
  }

  Ok(tensors)
}

fn tensor_data_to_array(tensor: &TensorData) -> Result<TensorArray> {
  let shape = IxDyn(&tensor.shape);
  match tensor.dtype {
    DType::F16 | DType::BF16 => Err(ConvertError::InvalidStructure(
      "f16/bf16 should be normalized to f32 before state_dict()".to_string(),
    )),
    DType::F32 => {
      let values = bytes_to_vec::<4, f32>(&tensor.bytes, f32::from_le_bytes)?;
      Ok(TensorArray::F32(ArrayD::from_shape_vec(shape, values)?))
    }
    DType::F64 => {
      let values = bytes_to_vec::<8, f64>(&tensor.bytes, f64::from_le_bytes)?;
      Ok(TensorArray::F64(ArrayD::from_shape_vec(shape, values)?))
    }
    DType::I8 => {
      let values = tensor.bytes.iter().map(|v| *v as i8).collect::<Vec<_>>();
      Ok(TensorArray::I8(ArrayD::from_shape_vec(shape, values)?))
    }
    DType::I16 => {
      let values = bytes_to_vec::<2, i16>(&tensor.bytes, i16::from_le_bytes)?;
      Ok(TensorArray::I16(ArrayD::from_shape_vec(shape, values)?))
    }
    DType::I32 => {
      let values = bytes_to_vec::<4, i32>(&tensor.bytes, i32::from_le_bytes)?;
      Ok(TensorArray::I32(ArrayD::from_shape_vec(shape, values)?))
    }
    DType::I64 => {
      let values = bytes_to_vec::<8, i64>(&tensor.bytes, i64::from_le_bytes)?;
      Ok(TensorArray::I64(ArrayD::from_shape_vec(shape, values)?))
    }
    DType::U8 => Ok(TensorArray::U8(ArrayD::from_shape_vec(shape, tensor.bytes.clone())?)),
    DType::Bool => {
      let values = tensor.bytes.iter().map(|v| *v != 0).collect::<Vec<_>>();
      Ok(TensorArray::Bool(ArrayD::from_shape_vec(shape, values)?))
    }
  }
}

fn bytes_to_vec<const N: usize, T>(bytes: &[u8], f: impl Fn([u8; N]) -> T) -> Result<Vec<T>> {
  if bytes.len() % N != 0 {
    return Err(ConvertError::InvalidStructure(format!(
      "tensor bytes are not divisible by {}",
      N
    )));
  }

  Ok(
    bytes
      .chunks_exact(N)
      .map(|chunk| {
        let mut arr = [0u8; N];
        arr.copy_from_slice(chunk);
        f(arr)
      })
      .collect(),
  )
}

fn normalize_tensor_dtype(dtype: DType, shape: Vec<usize>, bytes: Vec<u8>) -> Result<TensorData> {
  match dtype {
    DType::F16 => Ok(TensorData {
      dtype: DType::F32,
      shape,
      bytes: f16_bytes_to_f32_bytes(&bytes)?,
    }),
    DType::BF16 => Ok(TensorData {
      dtype: DType::F32,
      shape,
      bytes: bf16_bytes_to_f32_bytes(&bytes)?,
    }),
    _ => Ok(TensorData { dtype, shape, bytes }),
  }
}

fn f16_bytes_to_f32_bytes(input: &[u8]) -> Result<Vec<u8>> {
  if input.len() % 2 != 0 {
    return Err(ConvertError::InvalidStructure(
      "f16 tensor bytes must be even-length".to_string(),
    ));
  }
  let mut out = Vec::with_capacity(input.len() * 2);
  for chunk in input.chunks_exact(2) {
    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
    let value = f16_bits_to_f32(bits);
    out.extend_from_slice(&value.to_le_bytes());
  }
  Ok(out)
}

fn bf16_bytes_to_f32_bytes(input: &[u8]) -> Result<Vec<u8>> {
  if input.len() % 2 != 0 {
    return Err(ConvertError::InvalidStructure(
      "bf16 tensor bytes must be even-length".to_string(),
    ));
  }
  let mut out = Vec::with_capacity(input.len() * 2);
  for chunk in input.chunks_exact(2) {
    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
    let value = f32::from_bits((bits as u32) << 16);
    out.extend_from_slice(&value.to_le_bytes());
  }
  Ok(out)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
  let sign = ((bits >> 15) & 0x1) as u32;
  let exp = ((bits >> 10) & 0x1f) as u32;
  let frac = (bits & 0x03ff) as u32;

  let f32_bits = if exp == 0 {
    if frac == 0 {
      sign << 31
    } else {
      let mut mant = frac;
      let mut e = -14i32;
      while (mant & 0x0400) == 0 {
        mant <<= 1;
        e -= 1;
      }
      mant &= 0x03ff;
      let exp32 = (e + 127) as u32;
      (sign << 31) | (exp32 << 23) | (mant << 13)
    }
  } else if exp == 0x1f {
    (sign << 31) | (0xff << 23) | (frac << 13)
  } else {
    let exp32 = (exp as i32 - 15 + 127) as u32;
    (sign << 31) | (exp32 << 23) | (frac << 13)
  };

  f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::metadata::{collect_constructor_types, project_value_for_metadata};
  use crate::types::Value;
  use std::io::Write;
  use tempfile::tempdir;
  use zip::write::SimpleFileOptions;
  use zip::ZipWriter;

  #[test]
  fn converts_simple_tensor_checkpoint() {
    let tmp = tempdir().expect("tmp dir");
    let pt_path = tmp.path().join("weights.pt");
    write_fixture_checkpoint(&pt_path, false).expect("fixture checkpoint");

    let out_dir = tmp.path().join("export");
    let checkpoint = PtCheckpoint::from_pt(&pt_path, LoadOptions::default()).expect("checkpoint load should work");
    let result = checkpoint
      .export(&out_dir, ExportOptions::default())
      .expect("export should work");

    assert!(result.weights_path.exists());
    assert!(result.metadata_path.as_ref().expect("metadata path").exists());
    assert_eq!(result.tensor_count, 1);

    let yaml = fs::read_to_string(result.metadata_path.expect("metadata path")).expect("yaml readable");
    assert!(yaml.contains("layer.weight"));
    assert!(yaml.contains("dtype: F32") || yaml.contains("dtype: 'F32'"));
  }

  #[test]
  fn rejects_unsafe_global_reduce() {
    let tmp = tempdir().expect("tmp dir");
    let pt_path = tmp.path().join("unsafe.pt");
    write_fixture_checkpoint(&pt_path, true).expect("fixture checkpoint");

    let err = PtCheckpoint::from_pt(&pt_path, LoadOptions::default()).expect_err("unsafe pickle should fail");
    let msg = err.to_string();
    assert!(msg.contains("unsupported GLOBAL") || msg.contains("unsafe/unsupported pickle"));
  }

  #[test]
  fn projects_object_metadata_with_type_args_and_flattened_state() {
    let value = Value::Object {
      module: "ultralytics.nn.tasks".to_string(),
      name: "DetectionModel".to_string(),
      args: Some(Box::new(Value::Tuple(vec![
        Value::String("arg0".to_string()),
        Value::Int(42),
      ]))),
      state: Some(Box::new(Value::Dict(vec![(
        Value::String("training".to_string()),
        Value::Bool(false),
      )]))),
    };

    let projected = project_value_for_metadata(&value);
    let mapping = match projected {
      serde_yaml::Value::Mapping(map) => map,
      other => panic!("expected mapping, got {:?}", other),
    };

    let type_key = serde_yaml::Value::String("$type".to_string());
    let class_key = serde_yaml::Value::String("$class".to_string());
    let args_key = serde_yaml::Value::String("$args".to_string());
    let training_key = serde_yaml::Value::String("training".to_string());

    assert_eq!(
      mapping.get(&type_key),
      Some(&serde_yaml::Value::String("object".to_string()))
    );
    assert_eq!(
      mapping.get(&class_key),
      Some(&serde_yaml::Value::String(
        "ultralytics.nn.tasks.DetectionModel".to_string()
      ))
    );
    assert!(mapping.get(&args_key).is_some());
    assert_eq!(mapping.get(&training_key), Some(&serde_yaml::Value::Bool(false)));
  }

  #[test]
  fn omits_empty_object_args() {
    let value = Value::Object {
      module: "a".to_string(),
      name: "B".to_string(),
      args: Some(Box::new(Value::Tuple(Vec::new()))),
      state: None,
    };
    let projected = project_value_for_metadata(&value);
    let mapping = match projected {
      serde_yaml::Value::Mapping(map) => map,
      other => panic!("expected mapping, got {:?}", other),
    };

    let args_key = serde_yaml::Value::String("$args".to_string());
    assert!(!mapping.contains_key(&args_key));
  }

  #[test]
  fn collects_constructor_types_deduplicated_in_first_seen_order() {
    let tree = Value::List(vec![
      Value::Object {
        module: "a".to_string(),
        name: "One".to_string(),
        args: None,
        state: None,
      },
      Value::Dict(vec![(
        Value::String("nested".to_string()),
        Value::Object {
          module: "b".to_string(),
          name: "Two".to_string(),
          args: None,
          state: None,
        },
      )]),
      Value::Object {
        module: "a".to_string(),
        name: "One".to_string(),
        args: None,
        state: None,
      },
    ]);

    let objects = collect_constructor_types(&tree);
    assert_eq!(objects, vec!["a.One".to_string(), "b.Two".to_string()]);
  }

  fn write_fixture_checkpoint(path: &Path, unsafe_payload: bool) -> Result<()> {
    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default();

    let data_pkl = if unsafe_payload {
      build_unsafe_pickle()
    } else {
      build_safe_pickle()
    };

    zip.start_file("archive/data.pkl", options)?;
    zip.write_all(&data_pkl)?;

    let floats = [1.0f32, 2.0, 3.0, 4.0];
    let mut raw = Vec::new();
    for value in floats {
      raw.extend_from_slice(&value.to_le_bytes());
    }

    zip.start_file("archive/data/0", options)?;
    zip.write_all(&raw)?;
    zip.finish()?;
    Ok(())
  }

  fn build_safe_pickle() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&[0x80, 0x02]);

    out.push(b'}');
    out.push(b'(');

    push_binunicode(&mut out, "layer.weight");
    out.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");

    out.push(b'(');

    out.push(b'(');
    push_binunicode(&mut out, "storage");
    out.extend_from_slice(b"ctorch\nFloatStorage\n");
    push_binunicode(&mut out, "0");
    push_binunicode(&mut out, "cpu");
    out.push(b'K');
    out.push(4);
    out.push(b't');
    out.push(b'Q');

    out.push(b'K');
    out.push(0);

    out.push(b'(');
    out.push(b'K');
    out.push(2);
    out.push(b'K');
    out.push(2);
    out.push(b't');

    out.push(b'(');
    out.push(b'K');
    out.push(2);
    out.push(b'K');
    out.push(1);
    out.push(b't');

    out.push(0x89);
    out.push(b'N');

    out.push(b't');
    out.push(b'R');

    out.push(b'u');
    out.push(b'.');
    out
  }

  fn build_unsafe_pickle() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&[0x80, 0x02]);
    out.extend_from_slice(b"cos\nsystem\n");
    out.push(b'(');
    push_binunicode(&mut out, "echo hacked");
    out.push(b't');
    out.push(b'R');
    out.push(b'.');
    out
  }

  fn push_binunicode(out: &mut Vec<u8>, value: &str) {
    out.push(b'X');
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
  }
}

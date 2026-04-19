use serde_json::{json, Map, Value as JsonValue};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::types::{CheckpointMetadata, ConvertError, Result, TensorData};

pub(crate) fn write_safetensors(
  path: &Path,
  tensors: &BTreeMap<String, TensorData>,
  source_sha256: &str,
) -> Result<()> {
  let mut data = Vec::new();
  let mut tensor_meta = Map::new();

  for (name, tensor) in tensors {
    let start = data.len();
    data.extend_from_slice(&tensor.bytes);
    let end = data.len();

    tensor_meta.insert(
      name.clone(),
      json!({
        "dtype": tensor.dtype.as_safetensors(),
        "shape": tensor.shape,
        "data_offsets": [start, end],
      }),
    );
  }

  tensor_meta.insert(
    "__metadata__".to_string(),
    json!({
      "format_version": "1",
      "source_sha256": source_sha256,
      "converter": "pt-loader",
    }),
  );

  let header_bytes = serde_json::to_vec(&JsonValue::Object(tensor_meta))?;
  let mut out = File::create(path)?;
  out.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
  out.write_all(&header_bytes)?;
  out.write_all(&data)?;
  Ok(())
}

pub(crate) fn write_metadata_yaml(path: &Path, metadata: &CheckpointMetadata) -> Result<()> {
  let body = serde_yaml::to_string(metadata)
    .map_err(|error| ConvertError::InvalidStructure(format!("metadata yaml encode failed: {}", error)))?;
  std::fs::write(path, body)?;
  Ok(())
}

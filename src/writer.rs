use serde_json::{json, Map, Value as JsonValue};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::{ConvertError, Result, TensorData};

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

pub(crate) fn write_model_yaml(
  path: &Path,
  source_file: &Path,
  source_sha256: &str,
  tensor_count: usize,
  total_tensor_bytes: usize,
  metadata: &serde_yaml::Value,
  objects: &[String],
  tensors: &[(String, String, Vec<usize>, String)],
) -> Result<()> {
  let created_at = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|value| value.as_secs())
    .unwrap_or(0);

  let mut body = String::new();
  body.push_str("format_version: 1\n");
  body.push_str(&format!(
    "source_file: '{}'\n",
    yaml_quote(&source_file.display().to_string())
  ));
  body.push_str(&format!("source_sha256: '{}'\n", yaml_quote(source_sha256)));
  body.push_str("safetensors_file: 'model.safetensors'\n");
  body.push_str(&format!("created_at_unix: {}\n", created_at));
  body.push_str(&format!("tensor_count: {}\n", tensor_count));
  body.push_str(&format!("total_tensor_bytes: {}\n", total_tensor_bytes));
  let metadata_yaml = serde_yaml::to_string(metadata)
  .map_err(|error| ConvertError::InvalidStructure(format!("metadata yaml encode failed: {}", error)))?;
  body.push_str("metadata:\n");
  for line in metadata_yaml.lines() {
    body.push_str("  ");
    body.push_str(line);
    body.push('\n');
  }
  body.push_str("objects:\n");
  for item in objects {
    body.push_str(&format!("  - '{}'\n", yaml_quote(item)));
  }
  body.push_str("tensors:\n");

  for (name, dtype, shape, tensor_sha) in tensors {
    body.push_str(&format!("  - name: '{}'\n", yaml_quote(name)));
    body.push_str(&format!("    dtype: '{}'\n", yaml_quote(dtype)));
    body.push_str("    shape: [");
    for (idx, dim) in shape.iter().enumerate() {
      if idx > 0 {
        body.push_str(", ");
      }
      body.push_str(&dim.to_string());
    }
    body.push_str("]\n");
    body.push_str(&format!("    sha256: '{}'\n", yaml_quote(tensor_sha)));
  }

  fs::write(path, body)?;
  Ok(())
}

fn yaml_quote(input: &str) -> String {
  input.replace('\'', "''")
}

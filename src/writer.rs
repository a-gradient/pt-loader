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

pub fn write_metadata_yaml(path: &Path, metadata: &CheckpointMetadata) -> Result<()> {
  let raw = serde_yaml::to_string(metadata)
    .map_err(|error| ConvertError::InvalidStructure(format!("metadata yaml encode failed: {}", error)))?;
  let body = inline_known_int_vec_fields_in_tensors(&raw, &["shape", "stride"]);
  std::fs::write(path, body)?;
  Ok(())
}

pub fn inline_known_int_vec_fields_in_tensors(raw: &str, fields: &[&str]) -> String {
  let lines: Vec<&str> = raw.lines().collect();
  let mut out: Vec<String> = Vec::with_capacity(lines.len());
  let mut idx = 0usize;
  let mut in_tensors_section = false;

  while idx < lines.len() {
    let line = lines[idx];
    let indent = line.chars().take_while(|ch| ch.is_whitespace()).count();
    let trimmed = line.trim();

    if indent == 0 && trimmed == "tensors:" {
      in_tensors_section = true;
      out.push(line.to_string());
      idx += 1;
      continue;
    }

    if in_tensors_section
      && indent == 0
      && !trimmed.is_empty()
      && !trimmed.starts_with('-')
      && trimmed.ends_with(':')
      && trimmed != "tensors:"
    {
      in_tensors_section = false;
    }

    if in_tensors_section {
      let mut replaced = false;
      for field in fields {
        if trimmed == format!("{field}:") {
          let mut values: Vec<String> = Vec::new();
          let mut probe = idx + 1;
          while probe < lines.len() {
            let next = lines[probe];
            let next_indent = next.chars().take_while(|ch| ch.is_whitespace()).count();
            if next_indent < indent {
              break;
            }
            let next_trimmed = next.trim();
            let Some(value) = next_trimmed.strip_prefix("- ") else {
              break;
            };
            values.push(value.trim().to_string());
            probe += 1;
          }

          if !values.is_empty() {
            out.push(format!(
              "{}{}: {}",
              " ".repeat(indent),
              field,
              format_inline_int_vec(&values)
            ));
            idx = probe;
            replaced = true;
            break;
          }
        }
      }
      if replaced {
        continue;
      }
    }

    out.push(line.to_string());
    idx += 1;
  }

  if raw.ends_with('\n') {
    out.join("\n") + "\n"
  } else {
    out.join("\n")
  }
}

fn format_inline_int_vec(values: &[String]) -> String {
  let mut out = String::from("[");
  for (idx, value) in values.iter().enumerate() {
    if idx > 0 {
      out.push_str(", ");
    }
    out.push_str(value);
  }
  out.push(']');
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn formats_inline_int_vec() {
    let rendered = format_inline_int_vec(&["10".to_string(), "20".to_string(), "30".to_string()]);
    assert_eq!(rendered, "[10, 20, 30]");
  }

  #[test]
  fn inlines_only_known_fields_in_tensors_section() {
    let raw = r#"format_version: 1
metadata:
  shape:
  - 1
  - 2
tensors:
- name: x
  shape:
  - 10
  - 20
  - 30
  dtype: F32
"#;

    let rendered = inline_known_int_vec_fields_in_tensors(raw, &["shape", "stride"]);
    assert!(rendered.contains("metadata:\n  shape:\n  - 1\n  - 2"));
    assert!(rendered.contains("  shape: [10, 20, 30]"));
  }
}

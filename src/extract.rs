use std::collections::BTreeMap;

use crate::types::{ConvertError, LoadOptions, Result, TensorRef, Value};

pub(crate) fn extract_state_dict_tensors(
  root: &Value,
  opts: &LoadOptions,
) -> Result<BTreeMap<String, BTreeMap<String, TensorRef>>> {
  let mut groups = BTreeMap::new();

  if opts.state_dict_root_keys.is_empty() {
    let mut out = BTreeMap::new();
    collect_module_state_tensors(root, "", &mut out);
    if out.is_empty() {
      return Err(ConvertError::InvalidStructure(
        "could not find a tensor state_dict in checkpoint pickle".to_string(),
      ));
    }
    groups.insert("root".to_string(), out);
    return Ok(groups);
  }

  for key in &opts.state_dict_root_keys {
    let Some(target) = find_deep_child(root, key) else {
      if opts.state_dict_root_strict {
        return Err(ConvertError::InvalidStructure(format!(
          "could not find configured state_dict_root_key '{}' in checkpoint pickle",
          key
        )));
      }
      continue;
    };
    let mut out = BTreeMap::new();
    collect_module_state_tensors(target, "", &mut out);
    if out.is_empty() {
      if opts.state_dict_root_strict {
        return Err(ConvertError::InvalidStructure(format!(
          "could not find a tensor state_dict under configured state_dict_root_key '{}'",
          key
        )));
      }
      continue;
    }
    groups.insert(key.clone(), out);
  }

  if groups.is_empty() {
    return Err(ConvertError::InvalidStructure(
      "could not find a tensor state_dict in any configured state_dict_root_key".to_string(),
    ));
  }
  Ok(groups)
}

fn collect_module_state_tensors(value: &Value, prefix: &str, out: &mut BTreeMap<String, TensorRef>) {
  match value {
    Value::Object { state: Some(state), .. } => {
      collect_from_module_state(state, prefix, out);
    }
    Value::Dict(entries) => {
      collect_prefixed_tensor_map(value, prefix, out);
      for (key, child) in entries {
        let Some(name) = key_as_string(key) else {
          continue;
        };
        if name == "state_dict" {
          collect_prefixed_tensor_map(child, "", out);
          continue;
        }
        let next_prefix = join_name(prefix, &name);
        match child {
          Value::Object { .. } => collect_module_state_tensors(child, &next_prefix, out),
          Value::Dict(_) | Value::OrderedDict(_) | Value::List(_) | Value::Tuple(_) => {
            collect_module_state_tensors(child, &next_prefix, out)
          }
          _ => {}
        }
      }
    }
    Value::OrderedDict(entries) => {
      collect_prefixed_tensor_map(value, prefix, out);
      for (name, child) in entries {
        let next_prefix = join_name(prefix, name);
        match child {
          Value::Object { .. } => collect_module_state_tensors(child, &next_prefix, out),
          Value::Dict(_) | Value::OrderedDict(_) | Value::List(_) | Value::Tuple(_) => {
            collect_module_state_tensors(child, &next_prefix, out)
          }
          _ => {}
        }
      }
    }
    Value::List(items) | Value::Tuple(items) => {
      for (idx, child) in items.iter().enumerate() {
        let next_prefix = join_name(prefix, &idx.to_string());
        collect_module_state_tensors(child, &next_prefix, out);
      }
    }
    Value::Call { args, state, .. } => {
      for (idx, child) in args.iter().enumerate() {
        let next_prefix = join_name(prefix, &idx.to_string());
        collect_module_state_tensors(child, &next_prefix, out);
      }
      if let Some(state) = state {
        collect_module_state_tensors(state, prefix, out);
      }
    }
    _ => {}
  }
}

fn collect_from_module_state(state: &Value, prefix: &str, out: &mut BTreeMap<String, TensorRef>) {
  let Some(entries) = mapping_entries(state) else {
    return;
  };

  if let Some(parameters) = mapping_get(entries.as_slice(), "_parameters") {
    collect_named_tensors_from_mapping(parameters, prefix, out);
  }
  if let Some(buffers) = mapping_get(entries.as_slice(), "_buffers") {
    collect_named_tensors_from_mapping(buffers, prefix, out);
  }
  if let Some(modules) = mapping_get(entries.as_slice(), "_modules") {
    if let Some(module_entries) = mapping_entries(modules) {
      for (name, child) in module_entries {
        if matches!(child, Value::None) {
          continue;
        }
        let next_prefix = join_name(prefix, &name);
        collect_module_state_tensors(child, &next_prefix, out);
      }
    }
  }
}

fn collect_named_tensors_from_mapping(value: &Value, prefix: &str, out: &mut BTreeMap<String, TensorRef>) {
  let Some(entries) = mapping_entries(value) else {
    return;
  };
  for (name, child) in entries {
    let Some(tensor) = extract_tensor_ref(child) else {
      continue;
    };
    let full_name = join_name(prefix, &name);
    out.entry(full_name).or_insert(tensor);
  }
}

fn collect_prefixed_tensor_map(value: &Value, prefix: &str, out: &mut BTreeMap<String, TensorRef>) {
  let mapped = tensor_map_from_value(value);
  for (name, tensor) in mapped {
    let full_name = join_name(prefix, &name);
    out.entry(full_name).or_insert(tensor);
  }
}

fn find_named_child<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
  let entries = mapping_entries(value)?;
  for (name, child) in entries {
    if name == key {
      return Some(child);
    }
  }
  None
}

fn find_deep_child<'a>(value: &'a Value, key_path: &str) -> Option<&'a Value> {
  let mut current = value;
  for segment in key_path.split('.') {
    current = find_named_child(current, segment)?;
  }
  Some(current)
}

pub(crate) fn mapping_entries(value: &Value) -> Option<Vec<(String, &Value)>> {
  match value {
    Value::Dict(entries) => Some(
      entries
        .iter()
        .filter_map(|(k, v)| key_as_string(k).map(|name| (name, v)))
        .collect(),
    ),
    Value::OrderedDict(entries) => Some(entries.iter().map(|(k, v)| (k.clone(), v)).collect()),
    _ => None,
  }
}

fn mapping_get<'a>(entries: &'a [(String, &'a Value)], key: &str) -> Option<&'a Value> {
  entries.iter().find_map(|(name, value)| (name == key).then_some(*value))
}

pub(crate) fn key_as_string(value: &Value) -> Option<String> {
  match value {
    Value::String(v) => Some(v.clone()),
    Value::Int(v) => Some(v.to_string()),
    _ => None,
  }
}

fn join_name(prefix: &str, name: &str) -> String {
  if prefix.is_empty() {
    name.to_string()
  } else if name.is_empty() {
    prefix.to_string()
  } else {
    format!("{prefix}.{name}")
  }
}

fn extract_tensor_ref(value: &Value) -> Option<TensorRef> {
  match value {
    Value::TensorRef(spec) => Some(spec.clone()),
    Value::Tuple(items) if !items.is_empty() => match &items[0] {
      Value::TensorRef(spec) => Some(spec.clone()),
      _ => None,
    },
    _ => None,
  }
}

fn tensor_map_from_value(value: &Value) -> BTreeMap<String, TensorRef> {
  let mut out = BTreeMap::new();
  let iter: Box<dyn Iterator<Item = (&str, &Value)> + '_> = match value {
    Value::Dict(entries) => {
      let mapped = entries.iter().filter_map(|(k, v)| match k {
        Value::String(key) => Some((key.as_str(), v)),
        _ => None,
      });
      Box::new(mapped)
    }
    Value::OrderedDict(entries) => Box::new(entries.iter().map(|(k, v)| (k.as_str(), v))),
    _ => return out,
  };

  for (key, value) in iter {
    let tensor = extract_tensor_ref(value);
    if let Some(spec) = tensor {
      out.insert(key.to_string(), spec);
    }
  }

  out
}

pub(crate) fn numel(shape: &[usize]) -> Result<usize> {
  if shape.is_empty() {
    return Ok(1);
  }
  shape.iter().try_fold(1usize, |acc, dim| {
    acc
      .checked_mul(*dim)
      .ok_or_else(|| ConvertError::InvalidStructure("tensor shape product overflow".to_string()))
  })
}

pub(crate) fn contiguous_stride(shape: &[usize]) -> Vec<usize> {
  if shape.is_empty() {
    return Vec::new();
  }
  let mut stride = vec![0usize; shape.len()];
  let mut running = 1usize;
  for idx in (0..shape.len()).rev() {
    stride[idx] = running;
    running = running.saturating_mul(shape[idx]);
  }
  stride
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::{DType, StorageRef};

  fn tensor_ref_with_key(key: &str) -> Value {
    Value::TensorRef(TensorRef {
      storage: StorageRef {
        key: key.to_string(),
        dtype: DType::F32,
        size_elems: 1,
      },
      offset_elems: 0,
      shape: vec![1],
      stride: vec![1],
    })
  }

  fn module_with_named_parameter(param_name: &str, storage_key: &str) -> Value {
    Value::Object {
      module: "torch.nn.modules.linear".to_string(),
      name: "Linear".to_string(),
      args: Vec::new(),
      state: Some(Box::new(Value::Dict(vec![
        (
          Value::String("_parameters".to_string()),
          Value::Dict(vec![(Value::String(param_name.to_string()), tensor_ref_with_key(storage_key))]),
        ),
        (Value::String("_buffers".to_string()), Value::Dict(Vec::new())),
        (Value::String("_modules".to_string()), Value::Dict(Vec::new())),
      ]))),
    }
  }

  #[test]
  fn extracts_from_root_when_no_root_key_is_set() {
    let root = Value::Dict(vec![
      (
        Value::String("model".to_string()),
        module_with_named_parameter("w_model", "model_key"),
      ),
      (
        Value::String("module".to_string()),
        module_with_named_parameter("w_module", "module_key"),
      ),
    ]);

    let out = extract_state_dict_tensors(&root, &LoadOptions::default()).expect("extract tensors");
    let root_group = out.get("root").expect("root group");
    assert!(root_group.contains_key("model.w_model"));
    assert!(root_group.contains_key("module.w_module"));
  }

  #[test]
  fn supports_custom_state_dict_root_key_list() {
    let root = Value::Dict(vec![
      (
        Value::String("model".to_string()),
        module_with_named_parameter("w_model", "model_key"),
      ),
      (
        Value::String("module".to_string()),
        module_with_named_parameter("w_module", "module_key"),
      ),
    ]);

    let opts = LoadOptions {
      state_dict_root_keys: vec!["module".to_string()],
      ..LoadOptions::default()
    };
    let out = extract_state_dict_tensors(&root, &opts).expect("extract tensors");
    let module_group = out.get("module").expect("module group");
    assert!(module_group.contains_key("w_module"));
    assert!(!module_group.contains_key("w_model"));
  }

  #[test]
  fn fails_without_fallback_when_custom_root_key_is_missing_in_strict_mode() {
    let root = Value::Dict(vec![(
      Value::String("model".to_string()),
      module_with_named_parameter("w_model", "model_key"),
    )]);

    let opts = LoadOptions {
      state_dict_root_keys: vec!["ema".to_string()],
      ..LoadOptions::default()
    };
    let err = extract_state_dict_tensors(&root, &opts).expect_err("missing custom root key should fail");
    assert!(err.to_string().contains("could not find configured state_dict_root_key"));
  }

  #[test]
  fn skips_missing_when_non_strict_mode_is_enabled() {
    let root = Value::Dict(vec![(
      Value::String("model".to_string()),
      module_with_named_parameter("w_model", "model_key"),
    )]);

    let opts = LoadOptions {
      state_dict_root_keys: vec!["ema".to_string(), "model".to_string()],
      state_dict_root_strict: false,
      ..LoadOptions::default()
    };
    let out = extract_state_dict_tensors(&root, &opts).expect("should extract model key and skip ema");
    assert!(!out.contains_key("ema"));
    assert!(out.contains_key("model"));
  }

  #[test]
  fn supports_deep_key_path() {
    let root = Value::Dict(vec![(
      Value::String("ema".to_string()),
      Value::Dict(vec![(
        Value::String("model".to_string()),
        module_with_named_parameter("w_ema_model", "ema_model_key"),
      )]),
    )]);

    let opts = LoadOptions {
      state_dict_root_keys: vec!["ema.model".to_string()],
      ..LoadOptions::default()
    };
    let out = extract_state_dict_tensors(&root, &opts).expect("extract deep key path");
    let group = out.get("ema.model").expect("ema.model group");
    assert!(group.contains_key("w_ema_model"));
  }
}

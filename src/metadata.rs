use crate::extract::key_as_string;
use crate::types::Value;

fn bytes_to_hex(bytes: &[u8]) -> String {
  let mut out = String::with_capacity(bytes.len() * 2);
  for byte in bytes {
    out.push_str(&format!("{:02x}", byte));
  }
  out
}

pub(crate) fn project_root_metadata(root: &Value) -> serde_yaml::Value {
  project_value_for_metadata(root)
}

pub(crate) fn project_value_for_metadata(value: &Value) -> serde_yaml::Value {
  match value {
    Value::Marker => serde_yaml::Value::String("<marker>".to_string()),
    Value::None => serde_yaml::Value::Null,
    Value::Bool(v) => serde_yaml::Value::Bool(*v),
    Value::Int(v) => serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::String(v.to_string())),
    Value::Float(v) => serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::String(v.to_string())),
    Value::String(v) => serde_yaml::Value::String(v.clone()),
    Value::Bytes(v) => serde_yaml::Value::String(format!("0x{}", bytes_to_hex(v))),
    Value::List(items) | Value::Tuple(items) | Value::Set(items) => {
      serde_yaml::Value::Sequence(items.iter().map(project_value_for_metadata).collect())
    }
    Value::Dict(entries) => serde_yaml::Value::Mapping(
      entries
        .iter()
        .map(|(k, v)| {
          let key = key_as_string(k).unwrap_or_else(|| "<non-string-key>".to_string());
          (serde_yaml::Value::String(key), project_value_for_metadata(v))
        })
        .collect(),
    ),
    Value::OrderedDict(entries) => serde_yaml::Value::Mapping(
      entries
        .iter()
        .map(|(k, v)| (serde_yaml::Value::String(k.clone()), project_value_for_metadata(v)))
        .collect(),
    ),
    Value::Global { module, name } => serde_yaml::Value::String(format!("{module}.{name}")),
    Value::StorageRef(storage) => serde_yaml::Value::Mapping(
      [
        (
          serde_yaml::Value::String("$type".to_string()),
          serde_yaml::Value::String("storage_ref".to_string()),
        ),
        (
          serde_yaml::Value::String("dtype".to_string()),
          serde_yaml::Value::String(storage.dtype.as_safetensors().to_string()),
        ),
        (
          serde_yaml::Value::String("key".to_string()),
          serde_yaml::Value::String(storage.key.clone()),
        ),
        (
          serde_yaml::Value::String("size_elems".to_string()),
          serde_yaml::to_value(storage.size_elems).unwrap_or(serde_yaml::Value::Null),
        ),
      ]
      .into_iter()
      .collect(),
    ),
    Value::TensorRef(spec) => serde_yaml::Value::Mapping(
      [
        (
          serde_yaml::Value::String("$type".to_string()),
          serde_yaml::Value::String("tensor_ref".to_string()),
        ),
        (
          serde_yaml::Value::String("dtype".to_string()),
          serde_yaml::Value::String(spec.storage.dtype.as_safetensors().to_string()),
        ),
        (
          serde_yaml::Value::String("storage_key".to_string()),
          serde_yaml::Value::String(spec.storage.key.clone()),
        ),
        (
          serde_yaml::Value::String("offset_elems".to_string()),
          serde_yaml::to_value(spec.offset_elems).unwrap_or(serde_yaml::Value::Null),
        ),
        (
          serde_yaml::Value::String("shape".to_string()),
          serde_yaml::Value::Sequence(
            spec
              .shape
              .iter()
              .map(|v| serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::Null))
              .collect(),
          ),
        ),
        (
          serde_yaml::Value::String("stride".to_string()),
          serde_yaml::Value::Sequence(
            spec
              .stride
              .iter()
              .map(|v| serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::Null))
              .collect(),
          ),
        ),
      ]
      .into_iter()
      .collect(),
    ),
    Value::Object {
      module,
      name,
      args,
      state,
    } => {
      let mut map = serde_yaml::Mapping::new();
      map.insert(
        serde_yaml::Value::String("$type".to_string()),
        serde_yaml::Value::String("object".to_string()),
      );
      map.insert(
        serde_yaml::Value::String("$class".to_string()),
        serde_yaml::Value::String(format!("{module}.{name}")),
      );
      if !is_empty_args(args) {
        map.insert(
          serde_yaml::Value::String("$args".to_string()),
          serde_yaml::Value::Sequence(args.iter().map(project_value_for_metadata).collect()),
        );
      }
      if let Some(state) = state {
        let projected_state = project_value_for_metadata(state);
        match projected_state {
          serde_yaml::Value::Mapping(state_map) => {
            for (key, value) in state_map {
              map.insert(key, value);
            }
          }
          serde_yaml::Value::Null => {}
          other => {
            map.insert(serde_yaml::Value::String("$state".to_string()), other);
          }
        }
      }
      serde_yaml::Value::Mapping(map)
    }
    Value::Call { func, args, state } => {
      let mut map = serde_yaml::Mapping::new();
      map.insert(
        serde_yaml::Value::String("$type".to_string()),
        serde_yaml::Value::String("call".to_string()),
      );
      map.insert(
        serde_yaml::Value::String("$func".to_string()),
        serde_yaml::Value::String(func.clone()),
      );
      map.insert(
        serde_yaml::Value::String("$args".to_string()),
        serde_yaml::Value::Sequence(args.iter().map(project_value_for_metadata).collect()),
      );
      if let Some(state) = state {
        let projected_state = project_value_for_metadata(state);
        if !matches!(projected_state, serde_yaml::Value::Null) {
          map.insert(serde_yaml::Value::String("$state".to_string()), projected_state);
        }
      }
      serde_yaml::Value::Mapping(map)
    }
  }
}

fn is_empty_args(args: &[Value]) -> bool {
  args.is_empty()
}

pub(crate) fn collect_constructor_types(root: &Value) -> Vec<String> {
  let mut seen = std::collections::BTreeSet::new();
  let mut out = Vec::new();
  collect_constructor_types_inner(root, &mut seen, &mut out);
  out
}

fn collect_constructor_types_inner(
  value: &Value,
  seen: &mut std::collections::BTreeSet<String>,
  out: &mut Vec<String>,
) {
  match value {
    Value::Object {
      module,
      name,
      args,
      state,
    } => {
      let ty = format!("{module}.{name}");
      if seen.insert(ty.clone()) {
        out.push(ty);
      }
      for arg in args {
        collect_constructor_types_inner(arg, seen, out);
      }
      if let Some(state) = state {
        collect_constructor_types_inner(state, seen, out);
      }
    }
    Value::Call { args, state, .. } => {
      for item in args {
        collect_constructor_types_inner(item, seen, out);
      }
      if let Some(state) = state {
        collect_constructor_types_inner(state, seen, out);
      }
    }
    Value::Dict(entries) => {
      for (k, v) in entries {
        collect_constructor_types_inner(k, seen, out);
        collect_constructor_types_inner(v, seen, out);
      }
    }
    Value::OrderedDict(entries) => {
      for (_, v) in entries {
        collect_constructor_types_inner(v, seen, out);
      }
    }
    Value::List(items) | Value::Tuple(items) | Value::Set(items) => {
      for item in items {
        collect_constructor_types_inner(item, seen, out);
      }
    }
    _ => {}
  }
}

use crate::extract::key_as_string;
use crate::types::{NumpyEndian, NumpyScalarData, Value};

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
    Value::NumpyScalar {
      dtype,
      endian,
      shape,
      data,
    } => serde_yaml::Value::Mapping(
      [
        (
          serde_yaml::Value::String("$type".to_string()),
          serde_yaml::Value::String("numpy".to_string()),
        ),
        (
          serde_yaml::Value::String("dtype".to_string()),
          serde_yaml::Value::String(dtype.as_safetensors().to_string()),
        ),
        (
          serde_yaml::Value::String("endian".to_string()),
          project_numpy_endian(*endian),
        ),
        (
          serde_yaml::Value::String("shape".to_string()),
          serde_yaml::Value::Sequence(
            shape
              .iter()
              .map(|v| serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::Null))
              .collect(),
          ),
        ),
        (
          serde_yaml::Value::String("data".to_string()),
          project_numpy_scalar_data(data),
        ),
      ]
      .into_iter()
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

fn project_numpy_scalar_data(data: &NumpyScalarData) -> serde_yaml::Value {
  match data {
    NumpyScalarData::F32(v) => serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::String(v.to_string())),
    NumpyScalarData::F64(v) => serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::String(v.to_string())),
    NumpyScalarData::I8(v) => serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::String(v.to_string())),
    NumpyScalarData::I16(v) => serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::String(v.to_string())),
    NumpyScalarData::I32(v) => serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::String(v.to_string())),
    NumpyScalarData::I64(v) => serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::String(v.to_string())),
    NumpyScalarData::U8(v) => serde_yaml::to_value(*v).unwrap_or(serde_yaml::Value::String(v.to_string())),
    NumpyScalarData::Bool(v) => serde_yaml::Value::Bool(*v),
  }
}

fn project_numpy_endian(endian: NumpyEndian) -> serde_yaml::Value {
  match endian {
    NumpyEndian::NotApplicable => serde_yaml::Value::String("|".to_string()),
    NumpyEndian::Little => serde_yaml::Value::String("<".to_string()),
    NumpyEndian::Big => serde_yaml::Value::String(">".to_string()),
    NumpyEndian::Native => serde_yaml::Value::String("=".to_string()),
    NumpyEndian::NotSpecified => serde_yaml::Value::Null,
  }
}

pub(crate) fn collect_constructor_types(root: &Value) -> Vec<String> {
  let mut seen = std::collections::BTreeSet::new();
  let mut out = Vec::new();
  collect_constructor_types_inner(root, &mut seen, &mut out);
  out
}

pub(crate) fn collect_call_types(root: &Value) -> Vec<String> {
  let mut seen = std::collections::BTreeSet::new();
  let mut out = Vec::new();
  collect_call_types_inner(root, &mut seen, &mut out);
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

fn collect_call_types_inner(value: &Value, seen: &mut std::collections::BTreeSet<String>, out: &mut Vec<String>) {
  match value {
    Value::Object { args, state, .. } => {
      for arg in args {
        collect_call_types_inner(arg, seen, out);
      }
      if let Some(state) = state {
        collect_call_types_inner(state, seen, out);
      }
    }
    Value::Call { func, args, state } => {
      if seen.insert(func.clone()) {
        out.push(func.clone());
      }
      for item in args {
        collect_call_types_inner(item, seen, out);
      }
      if let Some(state) = state {
        collect_call_types_inner(state, seen, out);
      }
    }
    Value::Dict(entries) => {
      for (k, v) in entries {
        collect_call_types_inner(k, seen, out);
        collect_call_types_inner(v, seen, out);
      }
    }
    Value::OrderedDict(entries) => {
      for (_, v) in entries {
        collect_call_types_inner(v, seen, out);
      }
    }
    Value::List(items) | Value::Tuple(items) | Value::Set(items) => {
      for item in items {
        collect_call_types_inner(item, seen, out);
      }
    }
    _ => {}
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::{DType, NumpyEndian};

  #[test]
  fn projects_recovered_numpy_scalar_metadata() {
    let value = Value::NumpyScalar {
      dtype: DType::F32,
      endian: NumpyEndian::Little,
      shape: Vec::new(),
      data: NumpyScalarData::F32(3.5),
    };

    let projected = project_value_for_metadata(&value);
    let serde_yaml::Value::Mapping(map) = projected else {
      panic!("expected mapping");
    };

    assert_eq!(
      map.get(serde_yaml::Value::String("$type".to_string())),
      Some(&serde_yaml::Value::String("numpy".to_string()))
    );
    assert_eq!(
      map.get(serde_yaml::Value::String("dtype".to_string())),
      Some(&serde_yaml::Value::String("F32".to_string()))
    );
    assert_eq!(
      map.get(serde_yaml::Value::String("endian".to_string())),
      Some(&serde_yaml::Value::String("<".to_string()))
    );
    assert_eq!(
      map.get(serde_yaml::Value::String("shape".to_string())),
      Some(&serde_yaml::Value::Sequence(Vec::new()))
    );
    assert_eq!(
      map.get(serde_yaml::Value::String("data".to_string())),
      Some(&serde_yaml::to_value(3.5f32).expect("f32 value"))
    );
  }

  #[test]
  fn keeps_call_projection_for_unrecognized_calls() {
    let value = Value::Call {
      func: "numpy._core.multiarray.scalar".to_string(),
      args: vec![Value::String("x9".to_string())],
      state: None,
    };

    let projected = project_value_for_metadata(&value);
    let serde_yaml::Value::Mapping(map) = projected else {
      panic!("expected mapping");
    };
    assert_eq!(
      map.get(serde_yaml::Value::String("$type".to_string())),
      Some(&serde_yaml::Value::String("call".to_string()))
    );
    assert_eq!(
      map.get(serde_yaml::Value::String("$func".to_string())),
      Some(&serde_yaml::Value::String("numpy._core.multiarray.scalar".to_string()))
    );
  }

  #[test]
  fn projects_numpy_endian_symbols() {
    let cases = [
      (NumpyEndian::NotApplicable, serde_yaml::Value::String("|".to_string())),
      (NumpyEndian::Little, serde_yaml::Value::String("<".to_string())),
      (NumpyEndian::Big, serde_yaml::Value::String(">".to_string())),
      (NumpyEndian::Native, serde_yaml::Value::String("=".to_string())),
      (NumpyEndian::NotSpecified, serde_yaml::Value::Null),
    ];

    for (endian, expected) in cases {
      let value = Value::NumpyScalar {
        dtype: DType::F64,
        endian,
        shape: Vec::new(),
        data: NumpyScalarData::F64(1.0),
      };
      let projected = project_value_for_metadata(&value);
      let serde_yaml::Value::Mapping(map) = projected else {
        panic!("expected mapping");
      };
      assert_eq!(
        map.get(serde_yaml::Value::String("endian".to_string())),
        Some(&expected)
      );
    }
  }
}

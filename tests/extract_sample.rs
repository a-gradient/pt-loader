use pt_loader::{writer::inline_known_int_vec_fields_in_tensors, ExportOptions, LoadOptions, PtCheckpoint};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelYaml {
  format_version: usize,
  source_file: String,
  source_sha256: String,
  safetensors_file: String,
  created_at_unix: u64,
  tensor_count: usize,
  total_tensor_bytes: usize,
  #[serde(default)]
  metadata: serde_yaml::Value,
  #[serde(default)]
  objects: Vec<String>,
  tensors: Vec<ModelYamlTensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelYamlTensor {
  name: String,
  dtype: String,
  shape: Vec<usize>,
  sha256: String,
}

#[test]
fn extracts_sample_yolo26n_pt() {
  let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
  let input = crate_root.join("samples").join("yolo26n.pt");
  let reference_safetensors = crate_root.join("samples").join("yolo26n.safetensors");
  let reference_yaml = crate_root.join("samples").join("yolo26n.yaml");
  assert!(input.exists(), "expected sample checkpoint at {}", input.display());
  let needs_regen = reference_yaml_needs_regeneration(&reference_yaml);
  if needs_regen {
    assert!(
      reference_safetensors.exists(),
      "expected sample safetensors at {} to regenerate {}",
      reference_safetensors.display(),
      reference_yaml.display()
    );
    generate_yaml_from_safetensors(&reference_safetensors, &reference_yaml);
  }

  let out_dir = crate_root.join("out");
  if out_dir.exists() {
    std::fs::remove_dir_all(&out_dir).expect("remove old out dir");
  }
  std::fs::create_dir_all(&out_dir).expect("create out dir");

  let checkpoint = PtCheckpoint::load(&input, LoadOptions::default()).expect("sample checkpoint should load");
  let result = checkpoint
    .export(&out_dir, ExportOptions::default())
    .expect("sample checkpoint should convert");

  assert!(result.weights_path.exists());
  assert!(result.metadata_path.as_ref().expect("metadata path").exists());
  assert!(result.tensor_count > 0);
  assert!(result.total_tensor_bytes > 0);

  let converted_shapes = read_safetensors_shapes(&result.weights_path);
  assert!(!converted_shapes.is_empty(), "converted safetensors is empty");
  if reference_safetensors.exists() {
    let reference_shapes = read_safetensors_shapes(&reference_safetensors);
    assert!(!reference_shapes.is_empty(), "reference safetensors is empty");
    assert_eq!(
      converted_shapes, reference_shapes,
      "converted out/model.safetensors shapes must match samples/yolo26n.safetensors"
    );
  }

  let out_yaml_tensors = read_model_yaml_tensors(result.metadata_path.as_ref().expect("metadata"));
  let reference_yaml_tensors = read_model_yaml_tensors(&reference_yaml);
  assert_eq!(
    out_yaml_tensors, reference_yaml_tensors,
    "converted out/model.yaml tensors must match samples/yolo26n.yaml"
  );

  let out_yaml_path = result.metadata_path.as_ref().expect("metadata");
  let out_yaml = read_model_yaml(out_yaml_path);
  let out_yaml_raw = std::fs::read_to_string(out_yaml_path).expect("read output yaml");
  assert!(
    out_yaml_raw.contains("shape: ["),
    "expected inline flow sequence for tensor shape"
  );
  let metadata_len = match &out_yaml.metadata {
    serde_yaml::Value::Mapping(map) => map.len(),
    _ => 0,
  };
  assert!(
    metadata_len > 10,
    "expected extracted metadata map to contain many top-level fields, got {}",
    metadata_len
  );
  assert!(
    out_yaml
      .objects
      .iter()
      .any(|item| item == "ultralytics.nn.tasks.DetectionModel"),
    "expected objects to include ultralytics.nn.tasks.DetectionModel"
  );
}

fn reference_yaml_needs_regeneration(yaml_path: &Path) -> bool {
  if !yaml_path.exists() {
    true
  } else {
    let parsed = read_model_yaml(yaml_path);
    let raw = std::fs::read_to_string(yaml_path).expect("read existing yaml");
    parsed.source_sha256.trim().is_empty()
      || parsed.tensors.iter().any(|item| item.sha256.trim().is_empty())
      || raw.contains("\n    shape:\n")
  }
}

fn read_safetensors_shapes(path: &Path) -> BTreeMap<String, Vec<usize>> {
  let bytes = std::fs::read(path).expect("read safetensors bytes");
  let tensors = SafeTensors::deserialize(&bytes).expect("parse safetensors");
  let mut shapes = BTreeMap::new();
  for name in tensors.names() {
    let view = tensors.tensor(name).expect("tensor view should exist");
    shapes.insert(name.to_string(), view.shape().iter().map(|dim| *dim as usize).collect());
  }
  shapes
}

fn generate_yaml_from_safetensors(safetensors_path: &Path, yaml_path: &Path) {
  let bytes = std::fs::read(safetensors_path).expect("read safetensors bytes");
  let tensors = SafeTensors::deserialize(&bytes).expect("parse safetensors");

  let mut names = tensors
    .names()
    .iter()
    .map(|value| value.to_string())
    .collect::<Vec<_>>();
  names.sort();

  let mut total_tensor_bytes = 0usize;
  let mut yaml_tensors = Vec::with_capacity(names.len());
  for name in names {
    let view = tensors.tensor(&name).expect("tensor view should exist");
    let raw = view.data();
    total_tensor_bytes += raw.len();
    yaml_tensors.push(ModelYamlTensor {
      name,
      dtype: format!("{:?}", view.dtype()),
      shape: view.shape().iter().map(|dim| *dim as usize).collect(),
      sha256: sha256_bytes(raw),
    });
  }

  let payload = ModelYaml {
    format_version: 1,
    source_file: safetensors_path.display().to_string(),
    source_sha256: sha256_file(safetensors_path),
    safetensors_file: safetensors_path
      .file_name()
      .map(|value| value.to_string_lossy().into_owned())
      .unwrap_or_else(|| "yolo26n.safetensors".to_string()),
    created_at_unix: 0,
    tensor_count: yaml_tensors.len(),
    total_tensor_bytes,
    metadata: serde_yaml::Value::Null,
    objects: Vec::new(),
    tensors: yaml_tensors,
  };

  let encoded = serde_yaml::to_string(&payload).expect("encode yaml");
  let encoded = inline_known_int_vec_fields_in_tensors(&encoded, &["shape", "stride"]);
  std::fs::write(yaml_path, encoded).expect("write generated yaml");
}

fn read_model_yaml(path: &Path) -> ModelYaml {
  let raw = std::fs::read_to_string(path).expect("read yaml");
  serde_yaml::from_str::<ModelYaml>(&raw).expect("parse model yaml")
}

fn read_model_yaml_tensors(path: &Path) -> BTreeMap<String, (String, Vec<usize>, String)> {
  let parsed = read_model_yaml(path);
  let mut map = BTreeMap::new();
  for item in parsed.tensors {
    map.insert(item.name, (item.dtype, item.shape, item.sha256));
  }
  map
}

fn sha256_file(path: &Path) -> String {
  let bytes = std::fs::read(path).expect("read file for sha256");
  sha256_bytes(&bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
  let mut hasher = Sha256::new();
  hasher.update(bytes);
  hasher
    .finalize()
    .as_slice()
    .iter()
    .map(|byte| format!("{byte:02x}"))
    .collect()
}

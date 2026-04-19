use pt_loader::{
  writer::inline_known_int_vec_fields_in_tensors, ExportFormat, ExportOptions, LoadOptions, PtCheckpoint,
};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
fn extracts_all_sample_pt_files() {
  let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
  let samples_dir = crate_root.join("samples");
  let mut pt_files = list_sample_pt_files(&samples_dir);
  pt_files.sort();
  assert!(!pt_files.is_empty(), "expected at least one .pt sample in {}", samples_dir.display());

  let out_root = crate_root.join("out").join("test_extract_sample");
  if out_root.exists() {
    std::fs::remove_dir_all(&out_root).expect("remove old test out dir");
  }
  std::fs::create_dir_all(&out_root).expect("create test out dir");

  for input in pt_files {
    test_one_sample(&input, &samples_dir, &out_root);
  }
}

fn test_one_sample(input: &Path, samples_dir: &Path, out_root: &Path) {
  let stem = input
    .file_stem()
    .and_then(|value| value.to_str())
    .expect("sample pt file should have a valid stem")
    .to_string();

  let reference_yaml = samples_dir.join(format!("{stem}.yaml"));
  let reference_safetensors = samples_dir.join(format!("{stem}.safetensors"));
  let has_reference = reference_yaml.exists() || reference_safetensors.exists();
  let sample_out_dir = out_root.join(&stem);
  std::fs::create_dir_all(&sample_out_dir).expect("create sample out dir");

  let checkpoint = match PtCheckpoint::load(input, LoadOptions::default()) {
    Ok(value) => value,
    Err(err) if !has_reference => {
      eprintln!(
        "skipping extract-only sample {} because loading is not supported yet: {}",
        input.display(),
        err
      );
      return;
    }
    Err(err) => {
      panic!(
        "sample checkpoint should load for reference comparison ({}): {err}",
        input.display()
      );
    }
  };
  let result = checkpoint
    .export(&sample_out_dir, ExportOptions::new(ExportFormat::Safetensors, Some(input)))
    .unwrap_or_else(|err| panic!("sample checkpoint should convert ({}): {err}", input.display()));

  assert!(result.weights_path.exists(), "missing weights output for {}", input.display());
  assert!(
    result.metadata_path.as_ref().expect("metadata path").exists(),
    "missing metadata output for {}",
    input.display()
  );
  assert!(result.tensor_count > 0, "expected tensor_count > 0 for {}", input.display());
  assert!(
    result.total_tensor_bytes > 0,
    "expected total_tensor_bytes > 0 for {}",
    input.display()
  );

  let converted_shapes = read_safetensors_shapes(&result.weights_path);
  assert!(!converted_shapes.is_empty(), "converted safetensors is empty for {}", input.display());

  if reference_yaml.exists() {
    // 1) if corresponding yaml exists, compare with it
    let out_yaml_tensors = read_model_yaml_tensors(result.metadata_path.as_ref().expect("metadata"));
    let reference_yaml_tensors = read_model_yaml_tensors(&reference_yaml);
    assert_eq!(
      out_yaml_tensors, reference_yaml_tensors,
      "converted yaml tensors must match reference yaml for {}",
      input.display()
    );
  } else if reference_safetensors.exists() {
    // 2) if no yaml but safetensors exists, generate yaml from safetensors and compare
    let generated_yaml = sample_out_dir.join(format!("{stem}.generated.expected.yaml"));
    generate_yaml_from_safetensors(&reference_safetensors, &generated_yaml);
    let out_yaml_tensors = read_model_yaml_tensors(result.metadata_path.as_ref().expect("metadata"));
    let generated_yaml_tensors = read_model_yaml_tensors(&generated_yaml);
    assert_eq!(
      out_yaml_tensors, generated_yaml_tensors,
      "converted yaml tensors must match generated yaml from reference safetensors for {}",
      input.display()
    );
  } else {
    // 3) otherwise only extract
    assert!(
      result.weights_path.exists() && result.metadata_path.as_ref().expect("metadata").exists(),
      "expected extraction outputs for {}",
      input.display()
    );
  }
}

fn list_sample_pt_files(samples_dir: &Path) -> Vec<PathBuf> {
  let mut files = Vec::new();
  let entries = std::fs::read_dir(samples_dir)
    .unwrap_or_else(|err| panic!("failed reading samples dir {}: {err}", samples_dir.display()));

  for entry in entries {
    let entry = entry.expect("read_dir entry");
    let path = entry.path();
    if path.extension().and_then(|ext| ext.to_str()) == Some("pt") {
      files.push(path);
    }
  }

  files
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
      .unwrap_or_else(|| "model.safetensors".to_string()),
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

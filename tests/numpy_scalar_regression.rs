use pt_loader::{LoadOptions, PtCheckpoint, TensorManifest};
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

#[test]
fn recovers_numpy_scalars_from_pt_checkpoint_metadata() {
  let tmp = tempdir().expect("temp dir");
  let pt_path = tmp.path().join("numpy_scalars.pt");
  write_numpy_scalar_fixture(&pt_path).expect("fixture write");

  let ckpt = PtCheckpoint::load(&pt_path, LoadOptions::default()).expect("checkpoint load");
  assert!(ckpt.tensor_count() > 0);
  assert!(matches!(ckpt.metadata().tensors, TensorManifest::List(_)));

  let root = &ckpt.metadata().metadata;
  let optimizer = yaml_get(root, "optimizer");

  // Exact shape reported by user: f8 + _codecs.encode(..., latin1)
  let momentum = yaml_get(optimizer, "momentum");
  assert_eq!(yaml_string(momentum, "$type"), "numpy");
  assert_eq!(yaml_string(momentum, "dtype"), "F64");
  assert_eq!(yaml_string(momentum, "endian"), "<");
  assert_eq!(yaml_f64(momentum, "data"), 0.937);
  assert!(yaml_seq_len(momentum, "shape") == 0);

  let lr = yaml_get(optimizer, "lr");
  assert_eq!(yaml_string(lr, "$type"), "numpy");
  assert_eq!(yaml_string(lr, "dtype"), "F32");
  assert_eq!(yaml_string(lr, "endian"), "<");
  assert_eq!(yaml_f64(lr, "data"), 0.125);

  let steps = yaml_get(optimizer, "steps");
  assert_eq!(yaml_string(steps, "$type"), "numpy");
  assert_eq!(yaml_string(steps, "dtype"), "I64");
  assert_eq!(yaml_string(steps, "endian"), "<");
  assert_eq!(yaml_i64(steps, "data"), 42);

  let enabled = yaml_get(optimizer, "enabled");
  assert_eq!(yaml_string(enabled, "$type"), "numpy");
  assert_eq!(yaml_string(enabled, "dtype"), "BOOL");
  assert_eq!(yaml_string(enabled, "endian"), "|");
  assert!(yaml_bool(enabled, "data"));

  // Missing order resolves to NotSpecified and projects as null.
  let no_order = yaml_get(optimizer, "no_order");
  assert_eq!(yaml_string(no_order, "$type"), "numpy");
  assert_eq!(yaml_string(no_order, "dtype"), "F64");
  assert!(yaml_get(no_order, "endian").is_null());
  assert_eq!(yaml_f64(no_order, "data"), 0.5);

  // Unknown dtype remains an untouched call.
  let unknown = yaml_get(optimizer, "unknown");
  assert_eq!(yaml_string(unknown, "$type"), "call");
  assert_eq!(yaml_string(unknown, "$func"), "numpy._core.multiarray.scalar");

  // Unknown endian marker remains an untouched call.
  let unknown_order = yaml_get(optimizer, "unknown_order");
  assert_eq!(yaml_string(unknown_order, "$type"), "call");
  assert_eq!(yaml_string(unknown_order, "$func"), "numpy._core.multiarray.scalar");
}

fn write_numpy_scalar_fixture(path: &std::path::Path) -> pt_loader::Result<()> {
  let file = File::create(path)?;
  let mut zip = ZipWriter::new(file);
  let options = SimpleFileOptions::default();

  let data_pkl = build_numpy_scalar_pickle();
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

fn build_numpy_scalar_pickle() -> Vec<u8> {
  let mut out = Vec::new();
  out.extend_from_slice(&[0x80, 0x02]); // PROTO 2
  out.push(b'}'); // EMPTY_DICT
  out.push(b'('); // MARK for SETITEMS

  // root["layer.weight"] = rebuild_tensor(...)
  push_binunicode(&mut out, "layer.weight");
  push_tensor_rebuild_value(&mut out);

  // root["optimizer"] = {... numpy scalars ...}
  push_binunicode(&mut out, "optimizer");
  out.push(b'}'); // EMPTY_DICT (optimizer)
  out.push(b'('); // MARK for optimizer SETITEMS

  push_binunicode(&mut out, "momentum");
  push_numpy_scalar_codec(
    &mut out,
    "f8",
    "<",
    &[0x96, 0x43, 0x8b, 0x6c, 0xe7, 0xfb, 0xed, 0x3f],
    "latin1",
  );

  push_binunicode(&mut out, "lr");
  push_numpy_scalar_bytes(&mut out, "f4", "<", &0.125f32.to_le_bytes());

  push_binunicode(&mut out, "steps");
  push_numpy_scalar_codec(&mut out, "i8", "<", &42i64.to_le_bytes(), "latin-1");

  push_binunicode(&mut out, "enabled");
  push_numpy_scalar_bytes(&mut out, "b1", "|", &[1u8]);

  push_binunicode(&mut out, "unknown");
  push_numpy_scalar_bytes(&mut out, "x9", "<", &[1, 2, 3, 4]);

  push_binunicode(&mut out, "no_order");
  push_numpy_scalar_bytes_no_order(&mut out, "f8", &0.5f64.to_ne_bytes());

  push_binunicode(&mut out, "unknown_order");
  push_numpy_scalar_bytes(&mut out, "f8", "?", &[0x96, 0x43, 0x8b, 0x6c, 0xe7, 0xfb, 0xed, 0x3f]);

  out.push(b'u'); // optimizer SETITEMS
  out.push(b'u'); // root SETITEMS
  out.push(b'.'); // STOP
  out
}

fn push_tensor_rebuild_value(out: &mut Vec<u8>) {
  out.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
  out.push(b'('); // MARK args
  out.push(b'('); // MARK persistent id tuple
  push_binunicode(out, "storage");
  out.extend_from_slice(b"ctorch\nFloatStorage\n");
  push_binunicode(out, "0");
  push_binunicode(out, "cpu");
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
  out.push(0x89); // NEWFALSE
  out.push(b'N');
  out.push(b't');
  out.push(b'R');
}

fn push_numpy_scalar_bytes(out: &mut Vec<u8>, dtype_code: &str, order: &str, raw: &[u8]) {
  push_numpy_scalar_dtype(out, dtype_code, order);
  out.push(b'U');
  out.push(raw.len() as u8);
  out.extend_from_slice(raw);
  out.push(0x86); // TUPLE2(dtype,data)
  out.push(b'R'); // REDUCE scalar(dtype, data)
}

fn push_numpy_scalar_codec(out: &mut Vec<u8>, dtype_code: &str, order: &str, raw: &[u8], encoding: &str) {
  push_numpy_scalar_dtype(out, dtype_code, order);
  out.extend_from_slice(b"c_codecs\nencode\n");
  let text = raw.iter().map(|byte| char::from(*byte)).collect::<String>();
  push_binunicode(out, &text);
  push_binunicode(out, encoding);
  out.push(0x86); // TUPLE2(text, encoding)
  out.push(b'R'); // REDUCE _codecs.encode(...)
  out.push(0x86); // TUPLE2(dtype,data)
  out.push(b'R'); // REDUCE scalar(dtype, data)
}

fn push_numpy_scalar_bytes_no_order(out: &mut Vec<u8>, dtype_code: &str, raw: &[u8]) {
  out.extend_from_slice(b"cnumpy._core.multiarray\nscalar\n");
  out.extend_from_slice(b"cnumpy\ndtype\n");
  push_binunicode(out, dtype_code);
  out.push(0x89); // NEWFALSE
  out.push(0x88); // NEWTRUE
  out.push(0x87); // TUPLE3
  out.push(b'R'); // REDUCE numpy.dtype(...)
  out.push(b'U');
  out.push(raw.len() as u8);
  out.extend_from_slice(raw);
  out.push(0x86); // TUPLE2(dtype,data)
  out.push(b'R'); // REDUCE scalar(dtype, data)
}

fn push_numpy_scalar_dtype(out: &mut Vec<u8>, dtype_code: &str, order: &str) {
  out.extend_from_slice(b"cnumpy._core.multiarray\nscalar\n");
  out.extend_from_slice(b"cnumpy\ndtype\n");
  push_binunicode(out, dtype_code);
  out.push(0x89); // NEWFALSE
  out.push(0x88); // NEWTRUE
  out.push(0x87); // TUPLE3
  out.push(b'R'); // REDUCE numpy.dtype(...)
  out.push(b'('); // MARK for BUILD state tuple
  out.push(b'K');
  out.push(3);
  push_binunicode(out, order);
  out.push(b'N');
  out.push(b'N');
  out.push(b'N');
  out.push(b'J');
  out.extend_from_slice(&(-1i32).to_le_bytes());
  out.push(b'J');
  out.extend_from_slice(&(-1i32).to_le_bytes());
  out.push(b'K');
  out.push(0);
  out.push(b't');
  out.push(b'b'); // BUILD dtype state
}

fn push_binunicode(out: &mut Vec<u8>, value: &str) {
  out.push(b'X');
  out.extend_from_slice(&(value.len() as u32).to_le_bytes());
  out.extend_from_slice(value.as_bytes());
}

fn yaml_get<'a>(value: &'a serde_yaml::Value, key: &str) -> &'a serde_yaml::Value {
  let map = value.as_mapping().expect("expected mapping");
  map
    .get(serde_yaml::Value::String(key.to_string()))
    .unwrap_or_else(|| panic!("missing key {}", key))
}

fn yaml_string(value: &serde_yaml::Value, key: &str) -> String {
  yaml_get(value, key)
    .as_str()
    .unwrap_or_else(|| panic!("{} is not string", key))
    .to_string()
}

fn yaml_f64(value: &serde_yaml::Value, key: &str) -> f64 {
  yaml_get(value, key)
    .as_f64()
    .unwrap_or_else(|| panic!("{} is not f64-compatible", key))
}

fn yaml_i64(value: &serde_yaml::Value, key: &str) -> i64 {
  yaml_get(value, key)
    .as_i64()
    .unwrap_or_else(|| panic!("{} is not i64", key))
}

fn yaml_bool(value: &serde_yaml::Value, key: &str) -> bool {
  yaml_get(value, key)
    .as_bool()
    .unwrap_or_else(|| panic!("{} is not bool", key))
}

fn yaml_seq_len(value: &serde_yaml::Value, key: &str) -> usize {
  yaml_get(value, key)
    .as_sequence()
    .unwrap_or_else(|| panic!("{} is not sequence", key))
    .len()
}

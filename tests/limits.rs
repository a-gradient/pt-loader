use model_converter::{convert_pt_to_safetensors, ConvertOptions};
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

#[test]
fn rejects_non_zip_pt() {
  let dir = tempdir().expect("tempdir");
  let pt = dir.path().join("legacy.pt");
  std::fs::write(&pt, b"not a zip checkpoint").expect("write");

  let err =
    convert_pt_to_safetensors(&pt, dir.path(), ConvertOptions::default()).expect_err("should fail");
  assert!(err.to_string().contains("torch zip checkpoints"));
}

#[test]
fn enforces_pickle_size_limit() {
  let dir = tempdir().expect("tempdir");
  let pt = dir.path().join("large.pt");

  let file = File::create(&pt).expect("create");
  let mut zip = ZipWriter::new(file);
  zip
    .start_file("archive/data.pkl", SimpleFileOptions::default())
    .expect("start");
  zip.write_all(&vec![0u8; 1024]).expect("write");
  zip
    .start_file("archive/data/0", SimpleFileOptions::default())
    .expect("start data");
  zip.write_all(&[0u8; 16]).expect("write data");
  zip.finish().expect("finish");

  let opts = ConvertOptions {
    max_pickle_bytes: 32,
    ..ConvertOptions::default()
  };

  let err = convert_pt_to_safetensors(&pt, dir.path(), opts).expect_err("should fail");
  assert!(err.to_string().contains("data.pkl"));
}

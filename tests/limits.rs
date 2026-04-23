use pt_loader::{LoadOptions, PtCheckpoint};
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

  let err = PtCheckpoint::load(&pt, LoadOptions::default()).expect_err("should fail");
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

  let opts = LoadOptions {
    max_pickle_bytes: 32,
    ..LoadOptions::default()
  };

  let err = PtCheckpoint::load(&pt, opts).expect_err("should fail");
  assert!(err.to_string().contains("data.pkl"));
}

#[test]
fn sample_last_pt_no_longer_fails_on_call_build_state() {
  let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
  let path = crate_root.join("samples").join("last.pt");
  if !path.exists() {
    return;
  }

  let err = match PtCheckpoint::load(&path, LoadOptions::default()) {
    Ok(_) => return,
    Err(err) => err,
  };
  assert!(
    !err.to_string().contains("BUILD with non-empty state is not supported"),
    "regression: parser still fails with unsupported call BUILD state"
  );
}

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use zip::read::ZipArchive;

use crate::types::{ConvertError, Result};

pub(crate) fn find_data_pkl_name<R: Read + std::io::Seek>(archive: &mut ZipArchive<R>) -> Result<String> {
  let mut found: Option<String> = None;
  for index in 0..archive.len() {
    let file = archive.by_index(index)?;
    let name = file.name();
    if name == "data.pkl" || name.ends_with("/data.pkl") {
      if found.is_some() {
        return Err(ConvertError::InvalidStructure(
          "multiple data.pkl entries found in archive".to_string(),
        ));
      }
      found = Some(name.to_string());
    }
  }
  found.ok_or_else(|| ConvertError::InvalidStructure("missing data.pkl in archive".to_string()))
}

pub(crate) fn read_storage_blob<R: Read + std::io::Seek>(
  archive: &mut ZipArchive<R>,
  prefix: &str,
  key: &str,
) -> Result<Vec<u8>> {
  let candidates = [
    format!("{}data/{}", prefix, key),
    format!("data/{}", key),
    format!("{}{}", prefix, key),
    key.to_string(),
  ];

  for name in candidates {
    if let Ok(bytes) = read_zip_entry(archive, &name) {
      return Ok(bytes);
    }
  }

  Err(ConvertError::InvalidStructure(format!(
    "missing tensor storage data for key {}",
    key
  )))
}

pub(crate) fn read_zip_entry<R: Read + std::io::Seek>(
  archive: &mut ZipArchive<R>,
  name: &str,
) -> Result<Vec<u8>> {
  let mut file = archive.by_name(name)?;
  let mut out = Vec::new();
  file.read_to_end(&mut out)?;
  Ok(out)
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
  let mut file = File::open(path)?;
  let mut hasher = Sha256::new();
  let mut buf = [0u8; 8192];
  loop {
    let read = file.read(&mut buf)?;
    if read == 0 {
      break;
    }
    hasher.update(&buf[..read]);
  }
  let digest = hasher.finalize();
  Ok(bytes_to_lower_hex(digest.as_slice()))
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
  let mut hasher = Sha256::new();
  hasher.update(bytes);
  let digest = hasher.finalize();
  bytes_to_lower_hex(digest.as_slice())
}

fn bytes_to_lower_hex(bytes: &[u8]) -> String {
  bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

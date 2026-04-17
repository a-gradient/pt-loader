# pt-loader

Safe parser-based PyTorch checkpoint converter to safetensors with both Rust and Python APIs.

## Features

- Parses torch zip `.pt` checkpoints with strict safety limits.
- Converts checkpoints to `model.safetensors` + `model.yaml`.
- Inspects checkpoint metadata and tensor summaries.
- Loads tensors directly into Python as NumPy arrays.

## Python Usage

Install from source (local repo):

```bash
uv sync --group dev
uv run pytest -q
```

Example:

```python
import pt_loader

report = pt_loader.inspect("samples/yolo26n.pt")
print(report["tensor_count"])

result = pt_loader.convert("samples/yolo26n.pt", out_dir="out")
print(result["safetensors_path"])

tensors = pt_loader.load_pt("samples/yolo26n.pt")
print(next(iter(tensors.values())).shape)
```

## Rust Usage

```rust
use pt_loader::{convert_pt_to_safetensors, inspect_pt, ConvertOptions};
use std::path::Path;

let report = inspect_pt(Path::new("samples/yolo26n.pt"))?;
let result = convert_pt_to_safetensors(
    Path::new("samples/yolo26n.pt"),
    Path::new("out"),
    ConvertOptions::default(),
)?;
```

## Development

```bash
cargo test
cargo check --features pyo3
uv run pytest -q
```

## Releasing

- Tag a release as `vX.Y.Z`.
- GitHub Actions workflow `.github/workflows/release.yml` will:
  - publish to crates.io using `CRATES_IO_TOKEN`
  - build and publish to PyPI via trusted publishing

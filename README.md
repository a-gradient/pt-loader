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
from pt_loader import PtCheckpoint

ckpt = PtCheckpoint.load("samples/yolo26n.pt")
print(ckpt.metadata()["tensor_count"])

result = ckpt.export("out")
print(result["weights_path"])

tensors = ckpt.state_dict(backend="numpy")
print(next(iter(tensors.values())).shape)
```

## Rust Usage

```rust
use pt_loader::{ExportOptions, LoadOptions, PtCheckpoint};

let ckpt = PtCheckpoint::load("samples/yolo26n.pt", LoadOptions::default())?;
let result = ckpt.export("out", ExportOptions::default())?;
println!("{}", result.weights_path.display());
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

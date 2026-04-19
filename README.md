# pt-loader

Safe parser-based PyTorch checkpoint converter to safetensors with both Rust and Python APIs.

## Active Development Notice

`pt-loader` is under active development. During `v0.1.x`, we may introduce breaking API and behavior changes in both Rust and Python interfaces without backward-compatibility guarantees.

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

result = ckpt.export(format="safetensors", dir="out")
print(result["weights_path"])

tensors = ckpt.state_dict(backend="numpy")
print(next(iter(tensors.values())).shape)
```

## Rust Usage

```rust
use pt_loader::{ExportFormat, ExportOptions, LoadOptions, PtCheckpoint};
use std::path::Path;

let input = "samples/yolo26n.pt";
let ckpt = PtCheckpoint::load(input, LoadOptions::default())?;
let result = ckpt.export("out", ExportOptions::new(ExportFormat::Safetensors, Some(Path::new(input))))?;
println!("{}", result.weights_path.display());
```

## Development

```bash
cargo test
cargo check --features pyo3
uv run pytest -q
```

# pt-loader

[![CI](https://github.com/a-gradient/pt-loader/actions/workflows/ci.yml/badge.svg)](https://github.com/a-gradient/pt-loader/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/pt-loader.svg)](https://crates.io/crates/pt-loader)
[![docs.rs](https://img.shields.io/docsrs/pt-loader)](https://docs.rs/pt-loader)
[![PyPI](https://img.shields.io/pypi/v/pt-safe-loader.svg)](https://pypi.org/project/pt-safe-loader/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pt-safe-loader.svg)](https://pypi.org/project/pt-safe-loader/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Safe parser-based PyTorch checkpoint converter to safetensors with both Rust and Python APIs.

## Active Development Notice

`pt-loader` is under active development. During `v0.1.x`, we may introduce breaking API and behavior changes in both Rust and Python interfaces without backward-compatibility guarantees.

## Features

- Parses torch zip `.pt` checkpoints with strict safety limits.
- Converts checkpoints to `model.safetensors` + `model.yaml`.
- Inspects checkpoint metadata and tensor summaries.
- Loads tensors directly into Python as NumPy arrays.

## Installation

### Python

Install from PyPI:

```bash
pip install pt-safe-loader
```

Install from source (local repo):

```bash
uv sync --group dev
uv run maturin develop --features pyo3
```

### Rust

Add dependency:

```toml
[dependencies]
pt-loader = "0.1"
```

## Python Usage

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

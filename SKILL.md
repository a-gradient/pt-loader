# pt-loader User Skill

Use this package to safely read Torch zip `.pt` checkpoints and export tensors to safetensors.

This skill is for end users consuming the package APIs. It is not a contributor/developer workflow guide.

## Python

Install:

```bash
pip install pt-safe-loader
```

Basic usage:

```python
from pt_loader import PtCheckpoint

ckpt = PtCheckpoint.load("model.pt")

# Inspect metadata
meta = ckpt.metadata()
print(meta["tensor_count"])
print(meta["security"]["objects"])
print(meta["security"]["calls"])

# Export safetensors + yaml
result = ckpt.export(format="safetensors", dir="out")
print(result["weights_path"])
print(result["metadata_path"])

# Load tensors in memory
tensors = ckpt.state_dict(backend="numpy")
print(next(iter(tensors.items()))[0], next(iter(tensors.values())).shape)
```

Advanced root-key extraction:

```python
ckpt = PtCheckpoint.load(
  "model.pt",
  state_dict_root_keys=["model", "optimizer", "ema.model"],  # deep keys supported
  state_dict_root_strict=False,  # skip missing keys; fail if none succeed
)

result = ckpt.export(format="safetensors", dir="out")
print(result["weights_paths"])  # per-root files, e.g. *.model.safetensors
```

Notes:
- `state_dict_root_key="model"` is still accepted for backward compatibility.
- `state_dict_root_keys=[]` means extract from root only.

## Rust

Add dependency:

```toml
[dependencies]
pt-loader = "0.1"
```

Basic usage:

```rust
use pt_loader::{ExportFormat, ExportOptions, LoadOptions, PtCheckpoint};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = "model.pt";
    let ckpt = PtCheckpoint::load(input, LoadOptions::default())?;

    let meta = ckpt.metadata();
    println!("tensor_count={}", meta.tensor_count);
    println!("objects={:?}", meta.security.objects);
    println!("calls={:?}", meta.security.calls);

    let out = ckpt.export(
        "out",
        ExportOptions::new(ExportFormat::Safetensors, Some(Path::new(input))),
    )?;
    println!("weights={}", out.weights_path.display());
    println!("metadata={}", out.metadata_path.unwrap().display());
    Ok(())
}
```

Root-key extraction options:

```rust
use pt_loader::LoadOptions;

let opts = LoadOptions {
    state_dict_root_keys: vec![
        "model".to_string(),
        "optimizer".to_string(),
        "ema.model".to_string(), // deep key
    ],
    state_dict_root_strict: false,
    ..LoadOptions::default()
};
```

Notes:
- Empty `state_dict_root_keys` extracts from root only.
- Non-empty keys extract only those roots (no fallback scan).
- Strict mode fails fast on missing/invalid roots.

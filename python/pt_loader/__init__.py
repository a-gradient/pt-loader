from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._core import __version__, PtCheckpointCore

_DTYPE_TO_NUMPY: dict[str, str] = {
  "F16": "float16",
  "BF16": "float32",
  "F32": "float32",
  "F64": "float64",
  "I8": "int8",
  "I16": "int16",
  "I32": "int32",
  "I64": "int64",
  "U8": "uint8",
  "BOOL": "bool",
}

_NUMPY_TO_DTYPE: dict[str, str] = {
  "float16": "F16",
  "float32": "F32",
  "float64": "F64",
  "int8": "I8",
  "int16": "I16",
  "int32": "I32",
  "int64": "I64",
  "uint8": "U8",
  "bool": "BOOL",
}


class PtCheckpoint:
  def __init__(self, core: PtCheckpointCore):
    self._core = core

  @classmethod
  def from_pt(
    cls,
    pt_path: str | Path,
    *,
    max_archive_bytes: int | None = None,
    max_tensor_count: int | None = None,
    max_tensor_bytes: int | None = None,
    max_pickle_bytes: int | None = None,
    strict_contiguous: bool | None = None,
  ) -> "PtCheckpoint":
    core = PtCheckpointCore.from_pt(
      str(pt_path),
      max_archive_bytes=max_archive_bytes,
      max_tensor_count=max_tensor_count,
      max_tensor_bytes=max_tensor_bytes,
      max_pickle_bytes=max_pickle_bytes,
      strict_contiguous=strict_contiguous,
    )
    return cls(core)

  @classmethod
  def from_metadata(
    cls,
    metadata: dict[str, Any] | str | Path,
    *,
    state_dict: dict[str, Any] | None = None,
    weights_path: str | Path | None = None,
    backend: str = "numpy",
  ) -> "PtCheckpoint":
    metadata_yaml = _metadata_to_yaml(metadata)

    if state_dict is not None and weights_path is not None:
      raise ValueError("provide either state_dict or weights_path, not both")
    if state_dict is None and weights_path is None:
      raise ValueError("one of state_dict or weights_path is required")

    payload = None
    if state_dict is not None:
      payload = _state_dict_to_raw_payload(state_dict, backend=backend)

    core = PtCheckpointCore.from_metadata_yaml(
      metadata_yaml,
      weights_path=None if weights_path is None else str(weights_path),
      state_dict=payload,
    )
    return cls(core)

  def export(
    self,
    out_dir: str | Path,
    *,
    format: str = "safetensors",
    weights_filename: str = "model.safetensors",
    metadata_filename: str = "model.yaml",
    include_metadata: bool = True,
    overwrite: bool = False,
  ) -> dict[str, Any]:
    return json.loads(
      self._core.export_json(
        str(out_dir),
        format=format,
        weights_filename=weights_filename,
        metadata_filename=metadata_filename,
        include_metadata=include_metadata,
        overwrite=overwrite,
      )
    )

  def state_dict(self, *, backend: str = "numpy", copy: bool = True) -> dict[str, Any]:
    raw = self._core.state_dict_raw()
    tensors: dict[str, Any] = {}

    if backend == "numpy":
      import numpy as np

      for name, payload in raw.items():
        dtype_name = _DTYPE_TO_NUMPY[payload["dtype"]]
        shape = tuple(payload["shape"])
        data = payload["data"]
        array = np.frombuffer(data, dtype=dtype_name).reshape(shape)
        tensors[name] = array.copy() if copy else array
      return tensors

    if backend == "torch":
      import numpy as np
      import torch

      for name, payload in raw.items():
        dtype_name = _DTYPE_TO_NUMPY[payload["dtype"]]
        shape = tuple(payload["shape"])
        data = payload["data"]
        array = np.frombuffer(data, dtype=dtype_name).reshape(shape)
        value = torch.from_numpy(array.copy() if copy else array)
        tensors[name] = value
      return tensors

    raise ValueError("backend must be 'numpy' or 'torch'")

  def metadata(self) -> dict[str, Any]:
    return json.loads(self._core.metadata_json())



def _metadata_to_yaml(metadata: dict[str, Any] | str | Path) -> str:
  if isinstance(metadata, dict):
    return json.dumps(metadata)

  if isinstance(metadata, Path):
    return metadata.read_text(encoding="utf-8")

  maybe_path = Path(metadata)
  if maybe_path.exists():
    return maybe_path.read_text(encoding="utf-8")
  return metadata



def _state_dict_to_raw_payload(state_dict: dict[str, Any], *, backend: str) -> dict[str, Any]:
  if backend == "numpy":
    import numpy as np

    payload: dict[str, Any] = {}
    for name, tensor in state_dict.items():
      arr = np.asarray(tensor)
      payload[name] = _numpy_tensor_payload(arr)
    return payload

  if backend == "torch":
    import torch

    payload = {}
    for name, tensor in state_dict.items():
      if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
      else:
        arr = torch.as_tensor(tensor).detach().cpu().numpy()
      payload[name] = _numpy_tensor_payload(arr)
    return payload

  raise ValueError("backend must be 'numpy' or 'torch'")



def _numpy_tensor_payload(array: "Any") -> dict[str, Any]:
  import numpy as np

  arr = np.ascontiguousarray(array)
  dtype_name = str(arr.dtype)
  if dtype_name not in _NUMPY_TO_DTYPE:
    raise ValueError(f"unsupported dtype for from_metadata state_dict: {dtype_name}")

  return {
    "dtype": _NUMPY_TO_DTYPE[dtype_name],
    "shape": list(arr.shape),
    "data": arr.tobytes(order="C"),
  }


__all__ = ["__version__", "PtCheckpoint"]

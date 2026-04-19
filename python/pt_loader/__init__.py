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

_FORMAT_TO_EXT: dict[str, str] = {
  "safetensors": ".safetensors",
}

_EXT_TO_FORMAT: dict[str, str] = {
  ".safetensors": "safetensors",
}


class PtCheckpoint:
  def __init__(self, core: PtCheckpointCore, *, input_path: Path | None):
    self._core = core
    self._input_path = input_path

  @classmethod
  def load(
    cls,
    pt_path: str | Path,
    *,
    max_archive_bytes: int | None = None,
    max_tensor_count: int | None = None,
    max_tensor_bytes: int | None = None,
    max_pickle_bytes: int | None = None,
    strict_contiguous: bool | None = None,
  ) -> "PtCheckpoint":
    pt_path_obj = Path(pt_path)
    core = PtCheckpointCore.load(
      str(pt_path_obj),
      max_archive_bytes=max_archive_bytes,
      max_tensor_count=max_tensor_count,
      max_tensor_bytes=max_tensor_bytes,
      max_pickle_bytes=max_pickle_bytes,
      strict_contiguous=strict_contiguous,
    )
    return cls(core, input_path=pt_path_obj)

  def export(
    self,
    filename: str | Path | None = None,
    *,
    dir: str | Path | None = None,
    format: str | None = None,
    include_metadata: bool = True,
    overwrite: bool = False,
  ) -> dict[str, Any]:
    resolved_format, final_weights_path = self._resolve_export_target(
      filename=filename,
      dir=dir,
      format=format,
    )
    metadata_filename = f"{final_weights_path.stem}.yaml"

    return json.loads(
      self._core.export_json(
        str(final_weights_path.parent),
        format=resolved_format,
        weights_filename=final_weights_path.name,
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

  def _resolve_export_target(
    self,
    *,
    filename: str | Path | None,
    dir: str | Path | None,
    format: str | None,
  ) -> tuple[str, Path]:
    if filename is None and format is None:
      raise ValueError("format and filename cannot both be None")
    if filename is None and self._input_path is None:
      raise ValueError("filename is required when input path is unknown")

    resolved_format: str
    filename_path: Path
    if filename is None:
      assert self._input_path is not None
      resolved_format = _normalize_export_format(format)
      ext = _FORMAT_TO_EXT[resolved_format]
      filename_path = _replace_pt_suffix(self._input_path.name, ext)
    else:
      filename_path = Path(filename)
      if format is None:
        resolved_format = _infer_format_from_filename(filename_path)
      else:
        resolved_format = _normalize_export_format(format)

    if dir is not None:
      base_dir = Path(dir)
      final_weights_path = base_dir / filename_path
    elif filename_path.is_absolute():
      final_weights_path = filename_path
    elif self._input_path is not None:
      final_weights_path = self._input_path.parent / filename_path
    else:
      final_weights_path = Path.cwd() / filename_path

    return resolved_format, final_weights_path

def _normalize_export_format(value: str | None) -> str:
  if value is None:
    raise ValueError("format is required")
  if value not in _FORMAT_TO_EXT:
    raise ValueError(f"unsupported format: {value}")
  return value


def _infer_format_from_filename(filename: Path) -> str:
  ext = filename.suffix.lower()
  try:
    return _EXT_TO_FORMAT[ext]
  except KeyError as exc:
    raise ValueError(f"cannot infer format from filename suffix: {filename.suffix}") from exc


def _replace_pt_suffix(name: str, ext: str) -> Path:
  path = Path(name)
  if path.suffix.lower() == ".pt":
    return path.with_suffix(ext)
  return Path(path.stem + ext)


__all__ = ["__version__", "PtCheckpoint"]

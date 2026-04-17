from __future__ import annotations

import json
from typing import Any

from ._core import __version__, convert_json, inspect_json, load_pt_raw

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


def inspect(input_pt: str) -> dict[str, Any]:
  return json.loads(inspect_json(input_pt))


def convert(
  input_pt: str,
  out_dir: str = "out",
  *,
  max_archive_bytes: int | None = None,
  max_tensor_count: int | None = None,
  max_tensor_bytes: int | None = None,
  max_pickle_bytes: int | None = None,
  strict_contiguous: bool | None = None,
) -> dict[str, Any]:
  return json.loads(
    convert_json(
      input_pt,
      out_dir,
      max_archive_bytes=max_archive_bytes,
      max_tensor_count=max_tensor_count,
      max_tensor_bytes=max_tensor_bytes,
      max_pickle_bytes=max_pickle_bytes,
      strict_contiguous=strict_contiguous,
    )
  )


def load_pt(
  input_pt: str,
  *,
  max_archive_bytes: int | None = None,
  max_tensor_count: int | None = None,
  max_tensor_bytes: int | None = None,
  max_pickle_bytes: int | None = None,
  strict_contiguous: bool | None = None,
) -> dict[str, "Any"]:
  import numpy as np

  raw = load_pt_raw(
    input_pt,
    max_archive_bytes=max_archive_bytes,
    max_tensor_count=max_tensor_count,
    max_tensor_bytes=max_tensor_bytes,
    max_pickle_bytes=max_pickle_bytes,
    strict_contiguous=strict_contiguous,
  )

  tensors: dict[str, Any] = {}
  for name, payload in raw.items():
    dtype_name = _DTYPE_TO_NUMPY[payload["dtype"]]
    shape = tuple(payload["shape"])
    data = payload["data"]
    array = np.frombuffer(data, dtype=dtype_name).reshape(shape)
    tensors[name] = array.copy()

  return tensors


__all__ = ["__version__", "inspect", "convert", "load_pt"]

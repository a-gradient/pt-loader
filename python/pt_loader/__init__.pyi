from __future__ import annotations

from typing import Any
from numpy.typing import NDArray


def inspect(input_pt: str) -> dict[str, Any]: ...


def convert(
  input_pt: str,
  out_dir: str = "out",
  *,
  max_archive_bytes: int | None = None,
  max_tensor_count: int | None = None,
  max_tensor_bytes: int | None = None,
  max_pickle_bytes: int | None = None,
  strict_contiguous: bool | None = None,
) -> dict[str, Any]: ...


def load_pt(
  input_pt: str,
  *,
  max_archive_bytes: int | None = None,
  max_tensor_count: int | None = None,
  max_tensor_bytes: int | None = None,
  max_pickle_bytes: int | None = None,
  strict_contiguous: bool | None = None,
) -> dict[str, NDArray[Any]]: ...


__version__: str

__all__: list[str]

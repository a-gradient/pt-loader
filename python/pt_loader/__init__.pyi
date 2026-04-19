from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from numpy.typing import NDArray


class PtCheckpoint:
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
  ) -> PtCheckpoint: ...

  @classmethod
  def from_metadata(
    cls,
    metadata: dict[str, Any] | str | Path,
    *,
    state_dict: dict[str, Any] | None = None,
    weights_path: str | Path | None = None,
    backend: Literal["numpy", "torch"] = "numpy",
  ) -> PtCheckpoint: ...

  def export(
    self,
    out_dir: str | Path,
    *,
    format: Literal["safetensors"] = "safetensors",
    weights_filename: str = "model.safetensors",
    metadata_filename: str = "model.yaml",
    include_metadata: bool = True,
    overwrite: bool = False,
  ) -> dict[str, Any]: ...

  def state_dict(
    self,
    *,
    backend: Literal["numpy", "torch"] = "numpy",
    copy: bool = True,
  ) -> dict[str, NDArray[Any] | Any]: ...

  def metadata(self) -> dict[str, Any]: ...


__version__: str

__all__: list[str]

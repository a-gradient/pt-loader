from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from numpy.typing import NDArray


class PtCheckpoint:
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
    state_dict_root_key: str | None = None,
  ) -> PtCheckpoint: ...

  def export(
    self,
    filename: str | Path | None = None,
    *,
    dir: str | Path | None = None,
    format: Literal["safetensors"] | None = None,
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

import os
import shutil
from pathlib import Path

import pytest

mc = pytest.importorskip("pt_loader")
np = pytest.importorskip("numpy")


def _sample_path(name: str) -> str:
  root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  return os.path.join(root, "samples", name)


def test_from_pt_returns_checkpoint_and_metadata() -> None:
  ckpt = mc.PtCheckpoint.load(_sample_path("yolo26n.pt"))
  metadata = ckpt.metadata()

  assert isinstance(metadata, dict)
  assert metadata["tensor_count"] > 0
  assert isinstance(metadata["tensors"], list)


def test_export_writes_outputs(tmp_path) -> None:
  ckpt = mc.PtCheckpoint.load(_sample_path("yolo26n.pt"))
  result = ckpt.export(format="safetensors", dir=tmp_path)

  assert isinstance(result, dict)
  assert result["tensor_count"] > 0
  assert os.path.exists(result["weights_path"])
  assert os.path.exists(result["metadata_path"])


def test_export_filename_none_uses_input_name_with_format(tmp_path) -> None:
  sample = _sample_path("yolo26n.pt")
  local_pt = tmp_path / "weights.pt"
  shutil.copy(sample, local_pt)

  ckpt = mc.PtCheckpoint.load(local_pt)
  result = ckpt.export(format="safetensors")

  assert Path(result["weights_path"]) == (tmp_path / "weights.safetensors")


def test_export_relative_filename_defaults_to_input_folder_when_dir_none(tmp_path) -> None:
  sample = _sample_path("yolo26n.pt")
  local_pt = tmp_path / "nested" / "weights.pt"
  local_pt.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy(sample, local_pt)

  ckpt = mc.PtCheckpoint.load(local_pt)
  result = ckpt.export("custom.safetensors")

  assert Path(result["weights_path"]) == (local_pt.parent / "custom.safetensors")


def test_export_requires_filename_or_format() -> None:
  ckpt = mc.PtCheckpoint.load(_sample_path("yolo26n.pt"))
  with pytest.raises(ValueError, match="cannot both be None"):
    ckpt.export()


def test_state_dict_returns_numpy_tensors() -> None:
  ckpt = mc.PtCheckpoint.load(_sample_path("yolo26n.pt"))
  tensors = ckpt.state_dict(backend="numpy")
  assert isinstance(tensors, dict)
  assert len(tensors) > 0

  first = next(iter(tensors.values()))
  assert isinstance(first, np.ndarray)
  assert first.size > 0


def test_roundtrip_from_metadata_with_state_dict() -> None:
  ckpt = mc.PtCheckpoint.load(_sample_path("yolo26n.pt"))
  metadata = ckpt.metadata()
  tensors = ckpt.state_dict(backend="numpy")

  reconstructed = mc.PtCheckpoint.from_metadata(
    metadata,
    state_dict=tensors,
    backend="numpy",
  )

  reconstructed_tensors = reconstructed.state_dict(backend="numpy")
  assert reconstructed_tensors.keys() == tensors.keys()

  first_name = next(iter(tensors.keys()))
  assert reconstructed_tensors[first_name].shape == tensors[first_name].shape

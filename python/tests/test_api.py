import os
import shutil
from pathlib import Path

import pytest
import numpy as np

import pt_loader as ptl


def _sample_path(name: str) -> str:
  root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  return os.path.join(root, "samples", name)


def test_from_pt_returns_checkpoint_and_metadata() -> None:
  ckpt = ptl.PtCheckpoint.load(_sample_path("yolo26n.pt"))
  metadata = ckpt.metadata()

  assert isinstance(metadata, dict)
  assert metadata["tensor_count"] > 0
  assert isinstance(metadata["tensors"], list)


def test_export_writes_outputs(tmp_path) -> None:
  ckpt = ptl.PtCheckpoint.load(_sample_path("yolo26n.pt"))
  result = ckpt.export(format="safetensors", dir=tmp_path)

  assert isinstance(result, dict)
  assert result["tensor_count"] > 0
  assert os.path.exists(result["weights_path"])
  assert os.path.exists(result["metadata_path"])


def test_export_filename_none_uses_input_name_with_format(tmp_path) -> None:
  sample = _sample_path("yolo26n.pt")
  local_pt = tmp_path / "weights.pt"
  shutil.copy(sample, local_pt)

  ckpt = ptl.PtCheckpoint.load(local_pt)
  result = ckpt.export(format="safetensors")

  assert Path(result["weights_path"]) == (tmp_path / "weights.safetensors")


def test_export_relative_filename_defaults_to_input_folder_when_dir_none(tmp_path) -> None:
  sample = _sample_path("yolo26n.pt")
  local_pt = tmp_path / "nested" / "weights.pt"
  local_pt.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy(sample, local_pt)

  ckpt = ptl.PtCheckpoint.load(local_pt)
  result = ckpt.export("custom.safetensors")

  assert Path(result["weights_path"]) == (local_pt.parent / "custom.safetensors")


def test_export_requires_filename_or_format() -> None:
  ckpt = ptl.PtCheckpoint.load(_sample_path("yolo26n.pt"))
  with pytest.raises(ValueError, match="cannot both be None"):
    ckpt.export()


def test_state_dict_returns_numpy_tensors() -> None:
  ckpt = ptl.PtCheckpoint.load(_sample_path("yolo26n.pt"))
  tensors = ckpt.state_dict(backend="numpy")
  assert isinstance(tensors, dict)
  assert len(tensors) > 0

  first = next(iter(tensors.values()))
  assert isinstance(first, np.ndarray)
  assert first.size > 0


def test_load_state_dict_root_keys_non_strict_skips_missing() -> None:
  ckpt = ptl.PtCheckpoint.load(
    _sample_path("yolo26n.pt"),
    state_dict_root_keys=["ema", "model"],
    state_dict_root_strict=False,
  )
  assert ckpt.metadata()["tensor_count"] > 0


def test_load_state_dict_root_key_strict_missing_fails() -> None:
  with pytest.raises(
    RuntimeError,
    match="could not find a tensor state_dict under configured state_dict_root_key 'ema'",
  ):
    ptl.PtCheckpoint.load(
      _sample_path("yolo26n.pt"),
      state_dict_root_keys=["ema"],
      state_dict_root_strict=True,
    )


def test_load_single_root_key_backward_compat_matches_key_list() -> None:
  legacy = ptl.PtCheckpoint.load(_sample_path("yolo26n.pt"), state_dict_root_key="model")
  keyed = ptl.PtCheckpoint.load(_sample_path("yolo26n.pt"), state_dict_root_keys=["model"])

  assert legacy.metadata()["tensor_count"] == keyed.metadata()["tensor_count"]
  assert set(legacy.state_dict(backend="numpy").keys()) == set(keyed.state_dict(backend="numpy").keys())


def test_export_with_keyed_root_writes_per_key_output_and_grouped_manifest(tmp_path) -> None:
  ckpt = ptl.PtCheckpoint.load(
    _sample_path("yolo26n.pt"),
    state_dict_root_keys=["model"],
    state_dict_root_strict=True,
  )
  result = ckpt.export(format="safetensors", dir=tmp_path)
  metadata_text = Path(result["metadata_path"]).read_text(encoding="utf-8")

  assert "weights_paths" in result
  assert "model" in result["weights_paths"]
  assert Path(result["weights_paths"]["model"]).exists()
  assert "safetensors_files:" in metadata_text
  assert "model:" in metadata_text
  assert "tensors:" in metadata_text

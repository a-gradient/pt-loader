import os

import pytest

mc = pytest.importorskip("pt_loader")
np = pytest.importorskip("numpy")


def _sample_path(name: str) -> str:
  root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  return os.path.join(root, "samples", name)


def test_inspect_returns_tensor_summary() -> None:
  report = mc.inspect(_sample_path("yolo26n.pt"))
  assert isinstance(report, dict)
  assert report["tensor_count"] > 0
  assert isinstance(report["tensors"], list)


def test_convert_writes_outputs(tmp_path) -> None:
  result = mc.convert(_sample_path("yolo26n.pt"), out_dir=str(tmp_path))
  assert isinstance(result, dict)
  assert result["tensor_count"] > 0
  assert os.path.exists(result["safetensors_path"])
  assert os.path.exists(result["model_yaml_path"])


def test_load_pt_returns_numpy_tensors() -> None:
  tensors = mc.load_pt(_sample_path("yolo26n.pt"))
  assert isinstance(tensors, dict)
  assert len(tensors) > 0

  first = next(iter(tensors.values()))
  assert isinstance(first, np.ndarray)
  assert first.size > 0

import importlib.util

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from apertus_bench.effort import REPO  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "effort_weight_deltas", REPO / "tools/effort-weight-deltas.py"
)
deltas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deltas)


def test_relative_norms_cosines_and_frozen_tensors():
    reference = {"w": torch.ones(4), "frozen": torch.ones(2)}
    heads = {
        "low": {"w": torch.tensor([2.0, 1, 1, 1]), "frozen": torch.ones(2)},
        "high": {"w": torch.tensor([3.0, 1, 1, 1]), "frozen": torch.ones(2)},
        "mixed": {"w": torch.tensor([1.0, 0, 1, 1]), "frozen": torch.ones(2)},
    }
    result = deltas.compare(reference, heads)
    assert result["unchanged_tensors"] == ["frozen"]
    row = result["per_tensor"]["w"]
    assert abs(row["relative_frobenius"]["low"] - 0.5) < 1e-9
    assert abs(row["relative_frobenius"]["high"] - 1.0) < 1e-9
    assert abs(row["delta_cosine"]["low-high"] - 1.0) < 1e-9
    assert abs(row["delta_cosine"]["low-mixed"]) < 1e-9
    assert result["aggregate"]["relative_frobenius"]["low"] == row["relative_frobenius"]["low"]

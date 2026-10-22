import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "methods/dspark/scheduler/depth_ceiling.py"
spec = importlib.util.spec_from_file_location("depth_ceiling", SCRIPT)
assert spec and spec.loader
depth_ceiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_ceiling)


def test_interp_clamps_and_interpolates() -> None:
    xs, ys = [1, 64, 256], [1.0, 2.0, 4.0]
    assert depth_ceiling.interp(0, xs, ys) == 1.0
    assert depth_ceiling.interp(160, xs, ys) == pytest.approx(3.0)
    assert depth_ceiling.interp(999, xs, ys) == 4.0


def test_fit_line_recovers_step_and_prefill_cost() -> None:
    v, p = depth_ceiling.fit_line([(x, 80 * x + 0.035) for x in (0.001, 0.002, 0.003)])
    assert v == pytest.approx(80) and p == pytest.approx(0.035)

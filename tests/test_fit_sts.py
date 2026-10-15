import importlib.util
import json
from pathlib import Path

import pytest

# The fit runs in the serving container; locally numpy comes with the analysis
# extra (uv run --extra analysis pytest).
np = pytest.importorskip("numpy")

SCRIPT = Path(__file__).resolve().parents[1] / "methods/dspark/scheduler/fit_sts.py"
spec = importlib.util.spec_from_file_location("fit_sts", SCRIPT)
assert spec and spec.loader
fit_sts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit_sts)


def _simulate(true_temperatures: list[float], rows: int, seed: int = 0):
    """Drafts whose slot j is accepted, given its prefix was, with probability
    sigmoid(z_j / T_j): a head overconfident by T_j."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(1.5, 2.0, size=(rows, len(true_temperatures)))
    accept = rng.random(logits.shape) < fit_sts.sigmoid(logits / np.array(true_temperatures))
    # Accepted run: slots up to the first rejection.
    accepted = np.where(accept.all(axis=1), accept.shape[1], np.argmin(accept, axis=1))
    verified = np.full(rows, len(true_temperatures))
    return logits, verified, accepted


def test_fit_recovers_the_temperatures_slot_by_slot() -> None:
    logits, verified, accepted = _simulate([2.0, 1.5, 3.0], rows=60000)
    result = fit_sts.fit(logits, verified, accepted)
    assert np.allclose(result["temperatures"], [2.0, 1.5, 3.0], rtol=0.15)
    for slot in result["slots"]:
        assert slot["ece_sts"] < slot["ece_raw"]
        assert abs(slot["pred_sts"] - slot["observed"]) < 0.01


def test_unverified_slots_carry_no_label() -> None:
    logits, verified, accepted = _simulate([1.0, 1.0], rows=20000)
    # Only the first slot verified: the second has no samples, the first is
    # unchanged.
    verified[:] = 1
    accepted = np.minimum(accepted, 1)
    result = fit_sts.fit(logits, verified, accepted)
    assert result["slots"][1]["samples"] == 0
    assert result["slots"][0]["samples"] == 20000


def test_load_drops_each_requests_last_line(tmp_path: Path) -> None:
    dump = tmp_path / "conf.jsonl"
    lines = [
        {"req": "a", "z": [0.1, 0.2], "n": 2, "a": 1},
        {"req": "b", "z": [0.3, 0.4], "n": 2, "a": 2},
        {"req": "a", "z": [0.5, 0.6], "n": 2, "a": 0},
        {"req": "b", "z": [0.7, 0.8], "n": 1, "a": 0},
    ]
    dump.write_text("".join(json.dumps(line) + "\n" for line in lines))
    logits, verified, accepted = fit_sts.load([dump])
    assert logits.tolist() == [[0.1, 0.2], [0.3, 0.4]]
    assert verified.tolist() == [2, 2]
    assert accepted.tolist() == [1, 2]


def test_ece_is_zero_when_calibrated_per_bin() -> None:
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    labels = np.array([1.0, 0.0, 0.0, 0.0])
    assert fit_sts.ece(probs, labels) == 0.0
    assert fit_sts.ece(np.array([0.9]), np.array([0.0])) == 0.9

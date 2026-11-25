import importlib.util
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

SCRIPT = Path(__file__).resolve().parents[1] / "methods/dspark/scheduler/headroom_sim.py"
spec = importlib.util.spec_from_file_location("headroom_sim", SCRIPT)
assert spec and spec.loader
sim = importlib.util.module_from_spec(spec)
sys.modules["headroom_sim"] = sim  # dataclasses resolve the module by name
spec.loader.exec_module(sim)


def _table() -> dict:
    reqs = [1, 8, 32, 64, 128, 256]
    tokens = [1, 64, 256, 512, 1024, 2048]
    return {
        "verify_tokens": tokens,
        "verify_ms": [5.0 + 0.03 * t for t in tokens],
        "draft_reqs": reqs,
        "draft_ms_by_len": {
            str(length): [1.3 + 0.003 * r + length * (0.006 + 0.008 * r) for r in reqs]
            for length in range(1, 8)
        },
        "kv_ms_per_token": 3.6e-5,
    }


def _dump(path: Path, groups: dict[str, float], requests: int, steps: int) -> None:
    """Requests whose accepted run follows a per-label acceptance rate, with a
    head logit that tells the truth (high when accepted)."""
    rng = np.random.default_rng(0)
    with path.open("w") as handle:
        for label, rate in groups.items():
            for r in range(requests):
                for _ in range(steps):
                    accept = rng.random(7) < rate
                    a = 7 if accept.all() else int(np.argmin(accept))
                    z = [3.0 if j < a else -3.0 for j in range(7)]
                    row = {"req": f"{label}-{r}", "z": z, "n": 7, "a": a, "ctx": 500}
                    handle.write(json.dumps({**row, "label": label}) + "\n")


def test_cost_model_pair_fit_recovers_the_table() -> None:
    cost = sim.CostModel(_table())
    a0, a_r, o, c = cost.pair_coef
    assert a0 == pytest.approx(1.3, abs=1e-6)
    assert c == pytest.approx(0.008, abs=1e-6)
    assert cost.draft_batch(3, 64) == pytest.approx(1.3 + 0.192 + 3 * (0.006 + 0.512))


def test_budget_lengths_are_nested_and_follow_survival() -> None:
    surv = np.cumprod(np.array([[0.9] * 7, [0.5] * 7, [0.99] * 7]), axis=1)
    lengths = sim.budget_lengths(surv, np.array([1.0, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 0, 0]))
    assert lengths.tolist() == [3, 1, 5]


def test_oracle_and_per_request_beat_fixed_on_mixed_requests(tmp_path: Path) -> None:
    dump = tmp_path / "dump.jsonl"
    _dump(dump, {"easy": 0.95, "hard": 0.3}, requests=40, steps=30)
    trajectories = sim.load([dump], [1.0] * 7)
    assert {t.label for t in trajectories} == {"easy", "hard"}
    assert all(t.prompt == 500 for t in trajectories)
    cost = sim.CostModel(_table(), overhead_ms=2.0, prefill_us=30.0)

    def rate(**spec) -> float:
        return sim.simulate(trajectories, cost, 32, **spec)["tokens_per_s"]

    fixed = max(rate(rule="fixed", param=k, draft="batch") for k in range(1, 8))
    oracle = rate(rule="topk", draft="pair", signal="oracle")
    budget = rate(rule="budget", draft="pair")
    assert oracle > budget > fixed

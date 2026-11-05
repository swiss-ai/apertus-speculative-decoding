"""Continuing an independent layout never promotes failed scientific gates."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("continue_remaining.py")
SPEC = importlib.util.spec_from_file_location("remaining_layouts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(kind="cell", success=0.989, state="COMPLETED"):
    return {
        "slurm": {"state": state},
        "gate": {
            "passed": False,
            "checks": [
                {"kind": "arm_status", "passed": True},
                {
                    "kind": kind,
                    "passed": False,
                    "requests": {"attempted": 1000, "success_rate": success},
                },
            ],
        },
    }


def test_reliability_rejection_is_retained_not_promoted():
    r = record()
    assert MODULE.request_rejections_only(r)
    assert r["gate"]["passed"] is False
    assert r["gate"]["checks"][-1]["requests"]["success_rate"] < 0.99


def test_other_failures_never_enable_continuation():
    assert not MODULE.request_rejections_only(record(kind="provenance_counter_continuity"))
    assert not MODULE.request_rejections_only(record(state="FAILED"))
    assert not MODULE.request_rejections_only(record(success=0.999))


def test_continuation_scope_is_only_the_original_two_layouts():
    assert MODULE.REMAINING == ["tp2dp2", "pd-off"]

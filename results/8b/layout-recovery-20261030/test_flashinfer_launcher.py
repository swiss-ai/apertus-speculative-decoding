"""Home quota remedies route only this continuation's runtime writes."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location(
    "scratch_layout_launcher_v5", HERE / "continue_remaining_v5.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_only_owned_submit_runtime_moves_and_home_is_preserved(tmp_path):
    seen = {}

    def real(command, **kwargs):
        seen.update(kwargs)
        return "submitted"

    launch = MODULE.scratch_launcher(real, tmp_path)
    env = {
        "HOME": "/users/example",
        "SWEEP_OUTPUT": "/scratch/runs/tp2dp2-unique",
        "VLLM_PATCH_DIR": "/users/example/.sml/old",
        "EXTRA_ENV": "OTHER=keep",
    }
    result = launch([str(MODULE.recovery.REPO / "serving/node-sweep.sh"), "spec.yaml"], env=env)
    assert result == "submitted"
    assert env["HOME"] == "/users/example"
    assert env["VLLM_PATCH_DIR"] == str(tmp_path / "tp2dp2-unique/vllm-patch")
    assert env["SML_ENV_DIR"] == str(tmp_path / "tp2dp2-unique/sml-env")
    assert "OTHER=keep" in env["EXTRA_ENV"]
    assert str(tmp_path / "tp2dp2-unique/vllm-config") in env["EXTRA_ENV"]
    assert (
        "FLASHINFER_WORKSPACE_BASE=" + str(tmp_path / "tp2dp2-unique/flashinfer-workspace")
        in env["EXTRA_ENV"]
    )


def test_status_and_source_commands_are_unchanged(tmp_path):
    seen = []

    def real(command, **kwargs):
        seen.append((command, kwargs))
        return 0

    launch = MODULE.scratch_launcher(real, tmp_path)
    env = {"HOME": "/users/example"}
    launch(["git", "rev-parse", "HEAD"], env=env)
    assert seen == [(["git", "rev-parse", "HEAD"], {"env": env})]
    assert env == {"HOME": "/users/example"}

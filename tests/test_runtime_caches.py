import json
import os
import subprocess
import sys
from pathlib import Path

from apertus_bench.runtime_caches import cache_environment, prepare


def test_private_cache_paths_preserve_home_and_separate_deployments(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/existing-home")
    a, b = prepare(tmp_path / "a"), prepare(tmp_path / "b")
    assert os.environ["HOME"] == "/existing-home"
    assert a["home_unchanged"] == "/existing-home"
    assert a["directory_write_probes_passed"]
    assert "HOME" not in a["environment"]
    assert set(a["environment"].values()).isdisjoint(b["environment"].values())
    assert all(Path(p).is_dir() for p in a["environment"].values())
    assert a["environment"]["FLASHINFER_WORKSPACE_BASE"].startswith(str(tmp_path / "a"))
    assert a["environment"]["VLLM_CONFIG_ROOT"].startswith(str(tmp_path / "a"))


def test_wrapper_passes_private_environment_before_exec_and_records_it(tmp_path):
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "apertus_bench.runtime_caches",
            "--root",
            str(tmp_path / "cache"),
            "--report",
            str(report),
            "--exec",
            sys.executable,
            "-c",
            "import os,json; keys=('HOME','FLASHINFER_WORKSPACE_BASE',"
            "'VLLM_CONFIG_ROOT','T5_RUNTIME_CACHE_REPORT'); "
            "print(json.dumps({k:os.environ[k] for k in keys}))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    child = json.loads(result.stdout.splitlines()[-1])
    assert child["HOME"] == os.environ["HOME"]
    assert (
        child["FLASHINFER_WORKSPACE_BASE"]
        == cache_environment(tmp_path / "cache")["FLASHINFER_WORKSPACE_BASE"]
    )
    assert child["VLLM_CONFIG_ROOT"] == cache_environment(tmp_path / "cache")["VLLM_CONFIG_ROOT"]
    assert child["T5_RUNTIME_CACHE_REPORT"] == str(report)
    saved = json.loads(report.read_text())
    assert saved["environment"]["VLLM_CONFIG_ROOT"] == child["VLLM_CONFIG_ROOT"]

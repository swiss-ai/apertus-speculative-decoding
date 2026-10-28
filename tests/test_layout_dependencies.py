"""Pinned NIXL factory path and retained native-import failure evidence."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

PATH = Path(__file__).parents[1] / "experiments/high-concurrency-8b/recovery/check_dependencies.py"
SPEC = importlib.util.spec_from_file_location("layout_dependency_check", PATH)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_imports_the_pinned_factory_module_and_hashes_each_source(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("native wrapper")
    seen = []

    def loader(name):
        seen.append(name)
        return SimpleNamespace(__file__=str(source), NixlConnector=object())

    result = CHECK.inspect_dependencies(tmp_path, loader)
    assert seen[-1] == "vllm.distributed.kv_transfer.kv_connector.v1.nixl"
    assert result["passed"] is True
    assert result["connector_class"] == "NixlConnector"
    assert all(m["sha256"] for m in result["modules"].values())


def test_failed_connector_retains_successful_native_imports(tmp_path):
    def loader(name):
        if name == CHECK.NIXL_CONNECTOR_MODULE:
            raise ModuleNotFoundError("connector absent")
        return SimpleNamespace(__file__=None)

    result = CHECK.inspect_dependencies(tmp_path, loader)
    assert result["passed"] is False
    assert len(result["modules"]) == 4
    assert "ModuleNotFoundError" in result["error"]

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _watch():
    path = ROOT / "experiments/k-headline-live/gpu-watch.py"
    spec = importlib.util.spec_from_file_location("gpu_watch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pod(name, app, node, gpus=0, phase="Running"):
    resources = {"requests": {"nvidia.com/gpu": gpus}} if gpus else {}
    return {
        "metadata": {"name": name, "labels": {"app": app}},
        "spec": {"nodeName": node, "containers": [{"resources": resources}]},
        "status": {"phase": phase},
    }


def test_gpu_watch_skips_protected_nodes_and_counts_gpus():
    watch = _watch()
    pods = [
        _pod("a", "vllm-apertus-v1p5-70b-fixed-nov23", "n1", 4),
        _pod("b", "vllm-apertus-v1p5-8b-fixed-nov23", "n2", 1),
        _pod("c", "vllm-apertus2-chonk-head", "n3", 4),
        _pod("d", "vllm-qwen38-27b", "n4", 2),  # someone else's, but not protected
        _pod("e", "vllm-other", "n5", 4),
        _pod("f", "vllm-apertus-v1p5-8b-dspark-e10", "n6", 1),  # our own pod frees nothing
        _pod("g", "vllm-old", "n7", 4, phase="Succeeded"),
    ]
    result = watch.check(pods, pool=("n1", "n8"))
    assert set(result["protected"]) == {"n1", "n2", "n3"}
    assert result["candidates"] == ["n4", "n6", "n8"]  # n7: finished pods are ignored
    assert result["gpus_used"] == {"n1": 4, "n2": 1, "n3": 4, "n4": 2, "n5": 4}

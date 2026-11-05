from pathlib import Path
from types import SimpleNamespace

from apertus_bench import vllm_nvtx


def test_ranged_pushes_one_range_per_outer_call() -> None:
    events: list[str] = []
    push, pop = events.append, lambda: events.append("pop")

    class Base:
        def run(self, n: int) -> int:
            return n

    class Child(Base):
        def run(self, n: int) -> int:
            return super().run(n) + 1

    assert vllm_nvtx.wrap_method(Base, "run", "x", push, pop)
    assert vllm_nvtx.wrap_method(Child, "run", "x", push, pop)
    assert not vllm_nvtx.wrap_method(Child, "run", "x", push, pop)  # already wrapped
    assert not vllm_nvtx.wrap_method(Child, "missing", "x", push, pop)
    assert Child().run(1) == 2
    assert events == ["x", "pop"]


def test_step_kind_separates_prefill_from_decode_steps() -> None:
    decode = SimpleNamespace(
        scheduled_new_reqs=[],
        num_scheduled_tokens={"a": 8, "b": 1},
        scheduled_spec_decode_tokens={"a": [1] * 7},
    )
    assert vllm_nvtx.step_kind(None, decode) == "target:execute[decode]"
    chunk = SimpleNamespace(
        scheduled_new_reqs=[], num_scheduled_tokens={"a": 512}, scheduled_spec_decode_tokens={}
    )
    assert vllm_nvtx.step_kind(None, chunk) == "target:execute[mixed]"
    new = SimpleNamespace(scheduled_new_reqs=[object()], num_scheduled_tokens={})
    assert vllm_nvtx.step_kind(None, new) == "target:execute[mixed]"


def test_nvtx_patch_only_hooks_the_worker_module() -> None:
    patch = Path(__file__).resolve().parents[1] / "serving/patches/vllm-apertus-nvtx.patch"
    text = patch.read_text()
    targets = [line.split()[1] for line in text.splitlines() if line.startswith("+++ ")]
    assert targets == ["b/workspace/vllm/vllm/v1/worker/gpu_worker.py"]
    assert 'os.environ.get("VLLM_APERTUS_NVTX") == "1"' in text

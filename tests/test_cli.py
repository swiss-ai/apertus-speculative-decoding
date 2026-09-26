import pytest

from apertus_bench.cli import (
    _metadata,
    _require_eagle_provenance,
    _variant,
    build_parser,
)


def test_eagle3_requires_algorithm() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--base-url",
            "http://x",
            "--model",
            "m",
            "--workloads",
            "w.jsonl",
            "--workload",
            "chat",
            "--concurrency",
            "1",
            "--output",
            "out",
            "--variant",
            "e31-k3",
            "--method",
            "eagle3",
            "--num-speculative-tokens",
            "3",
        ]
    )
    with pytest.raises(ValueError, match="--algorithm is required"):
        _variant(args)


def test_eagle31_algorithm_is_recorded_separately_from_engine_method() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--base-url",
            "http://x",
            "--model",
            "m",
            "--workloads",
            "w.jsonl",
            "--workload",
            "chat",
            "--concurrency",
            "1",
            "--output",
            "out",
            "--variant",
            "e31-k3",
            "--method",
            "eagle3",
            "--algorithm",
            "eagle31",
            "--num-speculative-tokens",
            "3",
            "--draft-tensor-parallel-size",
            "4",
        ]
    )
    variant = _variant(args)
    assert variant.method == "eagle3"
    assert variant.algorithm == "eagle31"
    assert variant.parallel_drafting is False


def test_peagle_requires_parallel_drafting_flag() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--base-url",
            "http://x",
            "--model",
            "m",
            "--workloads",
            "w.jsonl",
            "--workload",
            "chat",
            "--concurrency",
            "1",
            "--output",
            "out",
            "--variant",
            "peagle",
            "--method",
            "eagle3",
            "--algorithm",
            "peagle",
            "--num-speculative-tokens",
            "3",
        ]
    )
    with pytest.raises(ValueError, match="parallel-drafting"):
        _variant(args)


def test_eagle3_cells_require_checkpoint_and_deployment_ids() -> None:
    extra = _metadata(["deployment_id=job-1"])
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _require_eagle_provenance("eagle3", extra)
    with pytest.raises(ValueError, match="target_revision"):
        _require_eagle_provenance(
            "eagle3",
            _metadata(["deployment_id=job-1", "checkpoint_sha256=abc", "target_model=m"]),
        )
    _require_eagle_provenance(
        "eagle3",
        _metadata(
            [
                "deployment_id=job-1",
                "checkpoint_sha256=abc",
                "target_model=swiss-ai/Apertus-v1.5-8B",
                "target_revision=a411d83",
                "target_tensor_parallel_size=1",
            ]
        ),
    )
    _require_eagle_provenance("none", {})

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from apertus_eagle.build_adaptation_pools import (
    Guard,
    RankedCandidates,
    language_quotas,
    read_json_rows,
    sharegpt_messages,
)


def test_duplicate_source_ids_do_not_compare_dictionary_payloads() -> None:
    candidates = RankedCandidates(3)
    candidates.add(10, {"id": "duplicate", "text": "first shard version"})
    candidates.add(10, {"id": "duplicate", "text": "second shard version"})
    candidates.add(2, {"id": "better", "text": "low rank"})
    candidates.add(1, {"id": "best", "text": "lowest rank"})
    # Bounded selection must survive equal ranks/IDs during both heap insertion
    # and final sorting, retain the best ranks, and be deterministic.
    assert [row["id"] for row in candidates.rows()] == ["best", "better", "duplicate"]
    assert candidates.rows()[-1]["text"] == "second shard version"
    with pytest.raises(ValueError, match="positive"):
        RankedCandidates(0)


def test_summary_candidates_keep_the_actual_source_shard(tmp_path, monkeypatch) -> None:
    from apertus_eagle import build_adaptation_pools as pools

    for i in range(6):
        (tmp_path / f"{i}.parquet").touch()

    class ParquetFile:
        def __init__(self, path):
            self.path = path

        def iter_batches(self, **_kwargs):
            doc = {
                "id": "repeated-id",
                "text": self.path.name,
                "token_count": 2500,
                "is_truncated": False,
                "fw_edu_scores": [3.0],
            }
            yield SimpleNamespace(to_pylist=lambda: [doc])

    parquet = SimpleNamespace(ParquetFile=ParquetFile)
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(parquet=parquet))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    monkeypatch.setattr(pools, "FINEPDFS", {"eng": tmp_path})
    monkeypatch.setattr(pools, "SUMMARY_SHARE", {"eng": 1.0})
    result = pools.summarization_candidates(2, 2.0, 2)["eng"]
    assert {row["source_file"] for row in result} == {"4.parquet", "5.parquet"}
    assert all(row["source_file"] in row["messages"][0]["content"] for row in result)


def _row(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def test_sharegpt_rows_become_prompts_without_the_answer() -> None:
    row = {"conversations": [{"from": "human", "value": "Q?"}, {"from": "gpt", "value": "A."}]}
    assert sharegpt_messages(row) == [{"role": "user", "content": "Q?"}]


def test_language_quotas_sum_to_the_total() -> None:
    quota = language_quotas(255, {"eng": 0.4, "deu": 0.2, "fra": 0.2, "ita": 0.2})
    assert sum(quota.values()) == 255
    assert quota["deu"] == 51


def test_json_rows_read_arrays_and_lines(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps([{"id": 1}, {"id": 2}]))
    (tmp_path / "b.jsonl").write_text('{"id": 1}\n\n{"id": 2}\n')
    assert [r["id"] for r in read_json_rows(tmp_path / "a.json")] == [1, 2]
    assert [r["id"] for r in read_json_rows(tmp_path / "b.jsonl")] == [1, 2]


def test_guard_rejects_test_prompts_and_near_duplicates() -> None:
    words = " ".join(f"word{i}" for i in range(40))
    guard = Guard([_row(words)], {"digest-of-test"})
    assert not guard.admit(_row("anything"), "digest-of-test")
    assert not guard.admit(_row(words.upper()), "d1")  # same text after normalization
    assert not guard.admit(_row(words + " one more"), "d2")  # 13-gram near duplicate
    assert guard.admit(_row("a different prompt entirely"), "d3")
    assert not guard.admit(_row("a different prompt entirely"), "d3")  # repeat
    assert guard.rejected == {
        "protected_digest": 1,
        "protected_text": 1,
        "near_dup": 1,
        "repeat": 1,
    }
    other = " ".join(f"other{i}" for i in range(40))
    guard.protect([_row(other)])
    assert not guard.admit(_row(other + " tail"), "d4")


def test_trace_seeds_depend_on_the_row_not_its_position() -> None:
    from apertus_eagle.generate_targets import row_seed

    assert row_seed("chat-1", 0) == row_seed("chat-1", 0)
    assert row_seed("chat-1", 0) != row_seed("chat-2", 0)
    assert 0 <= row_seed("x", 2**31 - 1) < 2**31


def test_swap_summary_helpers() -> None:
    from apertus_bench.drafter_swap import acceptance_between, compare_greedy, swap_stall

    samples = [
        {"t": 0.0, "drafts": 0, "accepted_tokens": 0},
        {"t": 10.0, "drafts": 100, "accepted_tokens": 250},
        {"t": 20.0, "drafts": 200, "accepted_tokens": 600},
    ]
    assert acceptance_between(samples, 0, 10)["mean_acceptance_length"] == 3.5
    assert acceptance_between(samples, 10, 20)["mean_acceptance_length"] == 4.5
    assert acceptance_between(samples, 30, 40)["mean_acceptance_length"] is None
    # Two streams ticking every 0.1 s; one stalls 0.7 s (4.9 to 5.6) across the swap.
    steady = [i * 0.1 for i in range(100)]
    stalled = [t for t in steady if not 4.95 < t < 5.55]
    stall = swap_stall([steady, stalled], 5.0, 5.4)
    assert stall["max_gap_across_swap_s"] == 0.7
    assert stall["streams_across_swap"] >= 2
    left = [{"id": "a", "sha256": "1", "text": "same"}, {"id": "b", "sha256": "2", "text": "abc"}]
    right = [{"id": "a", "sha256": "1", "text": "same"}, {"id": "b", "sha256": "3", "text": "abd"}]
    assert compare_greedy(left, right) == {
        "prompts": 2,
        "exact": 1,
        "exact_rate": 0.5,
        "first_divergence_chars": [2],
    }


def test_drafter_reload_refuses_another_architecture(tmp_path: Path) -> None:
    from apertus_bench.drafter_reload import architecture_mismatch, checkpoint_files, is_shared_name

    old = {"hidden_size": 4096, "num_hidden_layers": 1, "fc_norm": True, "torch_dtype": "bf16"}
    assert architecture_mismatch(old, {**old, "torch_dtype": "fp16"}) == {}
    assert architecture_mismatch(old, {**old, "num_hidden_layers": 2}) == {
        "num_hidden_layers": [1, 2]
    }
    assert is_shared_name("lm_head.weight") and not is_shared_name("layers.0.mlp.up_proj.weight")
    with pytest.raises(FileNotFoundError):
        checkpoint_files(tmp_path)

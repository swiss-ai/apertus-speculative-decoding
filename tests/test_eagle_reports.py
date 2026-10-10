import json
from pathlib import Path

import pytest
from apertus_eagle.a5_summary import load_cells, summarize
from apertus_eagle.a6_report import build_report


def _cell(
    root: Path,
    deployment: str,
    *,
    block: str,
    depth: int | None,
    workload: str,
    tpot: float,
    outputs: dict[str, str],
) -> None:
    cell = root / deployment / "cells" / workload / "c1"
    cell.mkdir(parents=True)
    method = "none" if depth is None else "eagle3"
    name = "baseline" if depth is None else f"eagle31-k{depth}"
    (cell / "metadata.json").write_text(
        json.dumps(
            {
                "variant": {"name": name, "method": method, "num_speculative_tokens": depth},
                "cell": {"workload": workload, "concurrency": 1},
                "extra": {"block_id": block, "deployment_id": deployment},
            }
        )
    )
    spec = {"drafts": 100, "accepted_tokens": 80, "acceptance_rate": 0.4}
    spec["acceptance_rate_per_position"] = {"0": 0.6, "1": 0.2}
    latency = {name: {"p50": tpot, "p95": 2 * tpot} for name in ("tpot", "ttft", "e2e")}
    (cell / "summary.json").write_text(
        json.dumps(
            {
                "requests": {"success_rate": 1.0, "successful": len(outputs)},
                "tokens": {"output_tokens_per_second": 1000 / tpot, "completion": 10},
                "latency_ms": latency,
                "speculative_decoding": spec if depth else {},
            }
        )
    )
    (cell / "requests.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "prompt_id": prompt,
                    "output_sha256": digest,
                    "success": True,
                    "completion_tokens": 5,
                    "finish_reason": "stop",
                }
            )
            + "\n"
            for prompt, digest in outputs.items()
        )
    )
    (root / deployment / "engine-excerpt.txt").write_text(
        "Model loading took 17.78 GiB\nAvailable KV cache memory: 55.24 GiB\n"
        "GPU KV cache size: 438,784 tokens\n"
    )


def _campaign(root: Path) -> None:
    same = {"p1": "a", "p2": "b"}
    for block, base_tpot, eagle_tpot in (("c1", 6.0, 4.0), ("c2", 6.0, 5.0)):
        _cell(
            root,
            f"base-{block}",
            block=block,
            depth=None,
            workload="code",
            tpot=base_tpot,
            outputs=same,
        )
        # c2's EAGLE deployment diverges on p2; the plain deployments agree.
        eagle_outputs = same if block == "c1" else {"p1": "a", "p2": "x"}
        _cell(
            root,
            f"k2-{block}",
            block=block,
            depth=2,
            workload="code",
            tpot=eagle_tpot,
            outputs=eagle_outputs,
        )


def test_a6_report_pairs_within_blocks_and_counts_agreement(tmp_path: Path) -> None:
    _campaign(tmp_path)
    report = build_report(tmp_path)
    cell = report["per_cell"]["k2/code/c1"]
    assert cell["blocks"] == {"c1": 1.5, "c2": 1.2}
    assert cell["geomean"] == pytest.approx((1.5 * 1.2) ** 0.5)
    assert cell["acceptance_per_position_mean"] == pytest.approx([0.6, 0.2])
    assert cell["g_mean"] == pytest.approx(1.8)
    agreement = report["greedy_agreement"]
    assert agreement["k2 vs baseline, same block/code/c1"] == {
        "identical": 3,
        "compared": 4,
        "share": 0.75,
    }
    assert agreement["baseline vs baseline, across blocks/code/c1"]["share"] == 1.0
    assert report["outputs"]["k2/code/c1"]["finish_reasons"] == {"stop": 4}
    assert report["deployments"]["k2-c1"]["kv_cache_tokens"] == 438784
    assert report["latency_p95"]["k2/code/c1/c2"]["tpot_p95_ratio"] == pytest.approx(5 / 6)


def test_a5_summary_refuses_a_cell_without_its_block_baseline(tmp_path: Path) -> None:
    _cell(tmp_path, "k2-c9", block="c9", depth=2, workload="code", tpot=4.0, outputs={"p": "a"})
    with pytest.raises(SystemExit, match="no same-block baseline"):
        summarize(load_cells(tmp_path))


def test_perfectblend_rows_convert_and_flag_problems() -> None:
    from apertus_eagle.perfectblend_check import to_messages

    good = {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}
    messages, problems = to_messages(good)
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert problems == []
    bad = {"conversations": [{"from": "gpt", "value": ""}, {"from": "gpt", "value": "x"}]}
    _, problems = to_messages(bad)
    assert {"empty_turn", "roles_not_alternating", "starts_with_assistant"} <= set(problems)


def test_perfectblend_shards_dedupe_prompts_and_keep_open_user_turns(tmp_path: Path) -> None:
    from apertus_eagle.perfectblend_shards import main

    def row(i: int, messages: list[dict[str, str]]) -> str:
        return json.dumps({"id": f"r{i}", "source": "s", "messages": messages}) + "\n"

    user = {"role": "user", "content": "q"}
    (tmp_path / "conversations.jsonl").write_text(
        row(0, [user, {"role": "assistant", "content": "a"}])
        + row(1, [user, {"role": "assistant", "content": "other"}])  # same prompt
        + row(2, [{"role": "user", "content": "open"}])  # ends with user: keep
        + row(3, [{"role": "user", "content": "leak"}, {"role": "assistant", "content": "x"}])
    )
    (tmp_path / "drop-ids.json").write_text(
        json.dumps({"r2": ["no_final_assistant"], "r3": ["benchmark_overlap"]})
    )
    main(["--data-dir", str(tmp_path), "--shards", "2", "--report", str(tmp_path / "r.json")])
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["counts"] == {"generate": 2, "duplicate_prompt": 1, "dropped": 1}
    assert (tmp_path / "duplicates.jsonl").read_text().strip() == json.dumps(
        {"id": "r1", "answer_from": "r0"}
    )


def test_online_rows_accept_both_formats_and_read_lazily(tmp_path: Path) -> None:
    from apertus_eagle.online_features import LineIndex, row_sequence

    generated = {"prompt_ids": [1, 2, 3], "generated_ids": [4, 5]}
    assert row_sequence(generated, 16) == ([1, 2, 3, 4, 5], [0, 0, 0, 1, 0])
    tokenized = {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]}
    assert row_sequence(tokenized, 3) == ([1, 2, 3], [0, 1, 0])
    with pytest.raises(ValueError, match="lengths differ"):
        row_sequence({"input_ids": [1, 2], "loss_mask": [1]}, 8)

    first = tmp_path / "a.jsonl"
    first.write_text('{"id": 0}\n\n{"id": 1}\n')
    second = tmp_path / "b.jsonl"
    second.write_text('{"id": 2}\n')
    index = LineIndex([first, second])
    assert [index.read(i)["id"] for i in range(len(index))] == [0, 1, 2]
    assert len(LineIndex([first, second], limit=2)) == 2


def test_micro_batches_group_by_length_under_the_token_budget() -> None:
    from apertus_eagle.train_rollout import micro_batches

    class Ids:
        def __init__(self, n: int) -> None:
            self.n = n

        def numel(self) -> int:
            return self.n

    records = [{"input_ids": Ids(n), "id": n} for n in (900, 100, 300, 120, 2000)]
    groups = micro_batches(records, 1000)
    assert [[r["id"] for r in g] for g in groups] == [[100, 120, 300], [900], [2000]]
    assert all(max(r["id"] for r in g) * len(g) <= 1000 or len(g) == 1 for g in groups)
    assert [len(g) for g in micro_batches(records, 0)] == [1, 1, 1, 1, 1]
    # Attention bound: 3 rows of 300 cost 3 * 300^2 > 400^2, so they split.
    squared = micro_batches(records, 1000, max_square=400 * 400)
    assert [[r["id"] for r in g] for g in squared] == [[100, 120], [300], [900], [2000]]


def test_rows_per_step_keeps_the_global_batch_across_rank_counts() -> None:
    from apertus_eagle.train_rollout import accumulation_steps

    assert accumulation_steps({"draft_accumulation_steps": 16}, 8) == 16
    assert accumulation_steps({"rows_per_step": 64, "draft_accumulation_steps": 16}, 4) == 16
    assert accumulation_steps({"rows_per_step": 64}, 8) == 8
    # Each step takes the next accum * world rows of the epoch order, sliced by rank,
    # so after s steps both rank counts have consumed the same rows.
    order = list(range(640))
    for world in (4, 8):
        accum = accumulation_steps({"rows_per_step": 64}, world)
        seen = {i for r in range(world) for i in order[r::world][: 3 * accum]}
        assert seen == set(order[:192])
    with pytest.raises(SystemExit):
        accumulation_steps({"rows_per_step": 64}, 12)


def test_benchmark_prompt_text_handles_the_common_layouts() -> None:
    from apertus_eagle.benchmark_overlap import prompt_text

    assert prompt_text({"question": "What is 2+2?", "answer": "4"}) == "What is 2+2?"
    assert prompt_text({"problem": "Solve x"}) == "Solve x"
    assert prompt_text({"turns": [{"content": "hi"}, {"content": "more"}]}) == "hi\nmore"
    assert prompt_text({"prompt": ["first", "second"]}) == "first\nsecond"
    assert prompt_text({"answer": "only"}) is None


def _speculators_row(conv: str, k: int, prompt: list[int], completion: list[int]) -> dict:
    # Shape written by speculators regenerate-responses (_sample_from_response).
    return {
        "id": f"{conv}_gen{k}",
        "primary_id": conv,
        "input_ids": prompt + completion,
        "loss_mask": [0] * len(prompt) + [1] * len(completion),
        "text": "review only",
        "metadata": {"idx": 0, "finish_reason": "stop", "usage": {}},
    }


def test_speculators_rows_check_split_and_train_as_is(tmp_path: Path) -> None:
    from apertus_eagle.online_features import row_sequence
    from apertus_eagle.speculators_corpus import check, main, row_problems

    rows = [
        _speculators_row("opb-1", 0, [1, 2, 3], [4, 5]),
        _speculators_row("opb-1", 1, [1, 2, 3, 4, 5, 6], [7]),
        _speculators_row("opb-2", 0, [1, 9], [8, 8, 8]),
    ]
    broken = _speculators_row("opb-3", 0, [1], [2])
    broken["loss_mask"] = [1, 0]
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [*rows, broken]))

    assert row_problems(rows[0]) == []
    assert row_problems(broken) == ["loss mask not a single trailing span"]
    report = check([path], max_seq_length=6)
    assert report["rows"] == 4 and report["rows_with_problems"] == 1
    assert report["conversations"] == 2
    assert report["tokens"] == 5 + 7 + 5 and report["loss_tokens"] == 2 + 1 + 3
    assert report["rows_over_max_seq_length"] == 1

    # online_features trains on the rows unchanged (last position has no target).
    assert row_sequence(rows[0], 16) == ([1, 2, 3, 4, 5], [0, 0, 0, 1, 0])

    ids = tmp_path / "heldout.txt"
    ids.write_text("opb-1\n")
    main(
        [
            "split",
            "--input",
            str(path),
            "--heldout-ids",
            str(ids),
            "--by",
            "conversation",
            "--output-dir",
            str(tmp_path / "s"),
        ]
    )
    held = (tmp_path / "s" / "heldout.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in held] == ["opb-1_gen0", "opb-1_gen1"]
    assert len((tmp_path / "s" / "train.jsonl").read_text().splitlines()) == 2


def test_speculators_split_by_row_ids_like_the_dspark_validation(tmp_path: Path) -> None:
    from apertus_eagle.speculators_corpus import main

    rows = [
        _speculators_row("opb-1", 0, [1], [2]),
        _speculators_row("opb-1", 1, [1, 2], [3]),
        _speculators_row("opb-2", 0, [1], [4]),
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ids = tmp_path / "val_ids.txt"
    ids.write_text("opb-1_gen1\n")
    args = ["split", "--input", str(path), "--heldout-ids", str(ids), "--output-dir"]
    main([*args, str(tmp_path / "s")])
    held = (tmp_path / "s" / "heldout.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in held] == ["opb-1_gen1"]
    # An id that is not in the corpus means the held-out set is not the intended one.
    ids.write_text("opb-1_gen1\nopb-9_gen0\n")
    with pytest.raises(SystemExit, match="matched 1 of 2"):
        main([*args, str(tmp_path / "t")])


def test_speculators_prompts_are_the_tokens_before_the_completion(tmp_path: Path) -> None:
    from apertus_eagle.speculators_corpus import main

    rows = [
        _speculators_row("opb-1", 0, [1, 2, 3], [4, 5]),
        _speculators_row("opb-2", 0, list(range(10)), [6]),
        _speculators_row("opb-3", 0, [7], [8]),
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "prompts.jsonl"
    main(["prompts", "--input", str(path), "--output", str(out), "--max-prompt-tokens", "5"])
    written = [json.loads(line) for line in out.read_text().splitlines()]
    # The 10-token prompt does not fit; the other two keep their prompt only.
    assert written == [
        {"id": "opb-1_gen0", "prompt_ids": [1, 2, 3]},
        {"id": "opb-3_gen0", "prompt_ids": [7]},
    ]
    main(["prompts", "--input", str(path), "--output", str(out), "--limit", "1"])
    assert len(out.read_text().splitlines()) == 1

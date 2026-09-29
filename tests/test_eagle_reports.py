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

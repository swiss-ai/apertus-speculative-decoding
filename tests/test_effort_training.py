import copy
import json

import pytest

from apertus_bench.effort_training import (
    REPO,
    TRAINING_CONFIG,
    budget_examples,
    load_config,
    resolve_budget,
    retain_responses,
    select_prompts,
    validate_parent,
    validate_reused_data,
)
from apertus_bench.workloads import Prompt, file_sha256


def samples():
    return [
        {
            "id": f"p-{i}",
            "effort": effort,
            "token_ids": list(range(2 + length)),
            "prompt_tokens": 2,
            "hidden_states_path": f"/{effort}/{i}",
        }
        for effort in ("low", "medium", "high")
        for i, length in enumerate((3, 5, 7))
    ]


def test_completed_data_reuse_requires_identical_inputs_and_budget(tmp_path):
    config = load_config(TRAINING_CONFIG)
    reference, campaign = tmp_path / "previous", tmp_path / "new"
    for path in (reference, campaign):
        (path / "inputs").mkdir(parents=True)
        lock = {
            key: key
            for key in (
                "spec",
                "target",
                "drafter",
                "workload",
                "image_sha256",
                "template_sha256",
                "rendering",
            )
        }
        (path / "inputs/lock.json").write_text(json.dumps(lock))
    (reference / "data-status.json").write_text('{"status":"complete"}')
    import yaml

    (reference / "phase2.yaml").write_text(
        yaml.safe_dump({key: value for key, value in config.items() if key != "optimizer"})
    )
    data = reference / "training-data"
    data.mkdir()
    manifest = {}
    for split in ("train", "development"):
        f = data / f"{split}-prompts.jsonl"
        f.write_text("[]\n")
        manifest[f"{split}_sha256"] = file_sha256(f)
    (data / "prompt-manifest.json").write_text(json.dumps(manifest))
    validate_reused_data(campaign, reference, config)
    with pytest.raises(ValueError, match="configuration or training budget"):
        validate_reused_data(campaign, reference, {**config, "supervised_tokens_per_head": 3})
    (data / "train-prompts.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="prompts changed"):
        validate_reused_data(campaign, reference, config)


@pytest.mark.parametrize("role", ("low", "medium", "high", "mixed"))
def test_exact_training_budget_and_balanced_mixture_keep_causal_context(role):
    source = samples()
    before = copy.deepcopy(source)
    rows = budget_examples(source, role, 120, 0)
    assert sum(sum(row["loss_mask"]) for row in rows) == 120
    assert source == before
    for effort in ("low", "medium", "high"):
        count = sum(sum(r["loss_mask"]) for r in rows if r["effort"] == effort)
        assert count == (40 if role == "mixed" else 120 if role == effort else 0)
    for row in rows:
        original = next(
            s for s in source if s["id"] == row["source_id"] and s["effort"] == row["effort"]
        )
        assert row["input_ids"] == original["token_ids"]
        assert len(row["loss_mask"]) == len(row["input_ids"])
        assert row["loss_mask"][:2] == [0, 0]
    assert rows == budget_examples(source, role, 120, 0)


def test_mixed_budget_requires_exact_division():
    with pytest.raises(ValueError, match="budget"):
        budget_examples(samples(), "mixed", 121, 0)


def test_prompt_splits_exclude_normalized_evaluation_tasks_and_cross_domain_duplicates():
    evaluation = [Prompt("eval", "math", [{"role": "user", "content": "same task"}], 10)]
    domains = {
        domain: [
            {"id": f"{domain}-{i}", "messages": [{"role": "user", "content": text}]}
            for i, text in enumerate(texts)
        ]
        for domain, texts in {
            "math": ["same   task", "math one", "math two", "math three"],
            "code": ["math one", "code one", "code two", "code three"],
        }.items()
    }
    config = {"seed": 0, "train_prompts_per_domain": 1, "development_prompts_per_domain": 1}
    train, dev = select_prompts(domains, evaluation, config)
    all_tasks = [p["messages"][0]["content"] for p in train + dev]
    assert len(train) == len(dev) == 2
    assert len(set(all_tasks)) == 4
    assert "same   task" not in all_tasks


def test_filter_excludes_a_task_at_all_efforts_and_requires_raw_native_ids():
    config = {
        "maximum_training_sequence": 100,
        "minimum_retained_train_prompts": 1,
        "minimum_retained_development_prompts": 1,
    }
    records = []
    for split in ("train", "development"):
        for identifier in ("good", "bad"):
            for effort in ("low", "medium", "high"):
                records.append(
                    {
                        "id": f"{split}-{identifier}",
                        "effort": effort,
                        "split": split,
                        "response": {
                            "prompt_token_ids": [1, 2],
                            "choices": [
                                {
                                    "token_ids": [3, 4],
                                    "finish_reason": "length"
                                    if identifier == "bad" and effort == "high"
                                    else "stop",
                                    "message": {"content": "answer"},
                                }
                            ],
                        },
                    }
                )
    retained, excluded = retain_responses(records, config)
    assert len(retained) == 6
    assert set(excluded) == {"train-bad", "development-bad"}
    assert all(r["token_ids"] == [1, 2, 3, 4] and r["prompt_tokens"] == 2 for r in retained)
    records[0]["response"]["prompt_token_ids"] = None
    with pytest.raises(ValueError, match="native"):
        retain_responses(records, config)


def test_phase2_can_use_new_source_but_requires_same_target_head_corpus_and_serving(
    monkeypatch, tmp_path
):
    import apertus_bench.effort_training as module

    monkeypatch.setattr(module, "build_report", lambda *a, **k: {"decision": "proceed"})
    before = {
        "target": {"sha256": "target"},
        "drafter": {"sha256": "drafter"},
        "workload": {"sha256": "corpus"},
        "source": {"revision": "old"},
        "spec": {"source_workload": "relative/path", "max_tokens": 32768},
    }
    after = copy.deepcopy(before)
    after["source"]["revision"] = "new"
    after["spec"]["source_workload"] = "/absolute/path"
    parent, child = tmp_path / "parent", tmp_path / "child"
    for root, lock in ((parent, before), (child, after)):
        (root / "inputs").mkdir(parents=True)
        (root / "inputs/lock.json").write_text(json.dumps(lock))
    assert validate_parent(child, parent) == before
    after["target"]["sha256"] = "different"
    (child / "inputs/lock.json").write_text(json.dumps(after))
    with pytest.raises(ValueError, match="target"):
        validate_parent(child, parent)


def test_frozen_training_configuration_has_matched_integral_budgets():
    config = load_config(TRAINING_CONFIG)
    assert config["epochs"] == 4
    assert config["supervised_tokens_per_head"] % (3 * config["epochs"]) == 0
    assert config["checkpoint_selection"] == "lowest_development_loss"
    assert config["upstream_revision"] == "e3041ddb4e58d8d0fad86f938f44423bdf424ae9"


SCRATCH = REPO / "experiments/thinking-effort/phase2-scratch.yaml"
FINETUNE = REPO / "experiments/thinking-effort/phase2-scratch-finetune.yaml"


def test_scratch_and_matched_finetune_differ_only_in_training_settings():
    scratch, finetune = load_config(SCRATCH), load_config(FINETUNE)
    assert scratch["initialization"] == "scratch" and finetune["initialization"] == "released"
    assert scratch["scheduler"] == "cosine" and finetune["scheduler"] == "none"
    assert finetune["learning_rate"] == load_config(TRAINING_CONFIG)["learning_rate"]
    data_keys = set(scratch) - {"initialization", "learning_rate", "scheduler", "warmup_ratio"}
    assert {k: scratch[k] for k in data_keys} == {k: finetune[k] for k in data_keys}
    assert scratch["domains"] == ["math"] and scratch["epochs"] == 1


def test_original_config_keeps_released_constant_math_and_code_defaults():
    config = load_config(TRAINING_CONFIG)
    assert config["initialization"] == "released"
    assert config["scheduler"] == "none"
    assert config["domains"] == ["math", "code"]


def test_smallest_pool_budget_gives_each_head_at_most_one_pass():
    config = {"supervised_tokens_per_head": "smallest_effort_pool"}
    source = [{**s, "split": "train"} for s in samples()]
    budget = resolve_budget(config, source)
    # Every effort pool is 3 + 5 + 7 = 15 assistant tokens.
    assert budget == 15
    for role in ("low", "medium", "high", "mixed"):
        rows = budget_examples(source, role, budget, 0)
        keys = [(r["source_id"], r["effort"]) for r in rows]
        assert len(keys) == len(set(keys))
        assert sum(sum(r["loss_mask"]) for r in rows) == budget
    assert resolve_budget({"supervised_tokens_per_head": 99}, source) == 99


def test_invalid_scratch_settings_are_rejected(tmp_path):
    import yaml

    base = yaml.safe_load(SCRATCH.read_text())
    for change in (
        {"initialization": "random"},
        {"scheduler": "linear"},
        {"domains": ["chat"]},
        {"epochs": 2},
    ):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**base, **change}))
        with pytest.raises(ValueError):
            load_config(path)


def test_all_remaining_tasks_train_after_the_development_split():
    evaluation = [Prompt("eval", "math", [{"role": "user", "content": "held out"}], 10)]
    domains = {
        "math": [
            {"id": f"math-{i}", "messages": [{"role": "user", "content": f"task {i}"}]}
            for i in range(6)
        ]
        + [{"id": "dup", "messages": [{"role": "user", "content": "held  out"}]}]
    }
    config = {
        "seed": 0,
        "domains": ["math"],
        "train_prompts_per_domain": "all",
        "development_prompts_per_domain": 2,
    }
    train, dev = select_prompts(domains, evaluation, config)
    assert len(dev) == 2 and len(train) == 4
    assert {p["id"] for p in train}.isdisjoint({p["id"] for p in dev})
    assert "dup" not in {p["id"] for p in train + dev}

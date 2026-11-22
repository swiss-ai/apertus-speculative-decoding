import copy
import json

import pytest

from apertus_bench.effort_training import (
    TRAINING_CONFIG,
    budget_examples,
    load_config,
    retain_responses,
    select_prompts,
    validate_parent,
)
from apertus_bench.workloads import Prompt


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

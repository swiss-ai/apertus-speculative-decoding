import json
from pathlib import Path

import pytest
from apertus_eagle.contract import (
    ContractError,
    aux_layer_ids,
    derive_aux_layers,
    load_contract,
    target_keys,
    validate_draft_config_path,
)
from apertus_eagle.parity import serving_bos_policy
from apertus_eagle.parse_rendered import parse_rendered_text, prompt_and_last_assistant
from apertus_eagle.prepare_data import assign_split
from apertus_eagle.renderer import ASSISTANT_END_ID, ASSISTANT_START_ID, BOS_ID, assistant_loss_mask
from apertus_eagle.target_adapter import torchspec_model_overrides

from apertus_bench.eagle import count_draft_parameters

REPO = Path(__file__).resolve().parents[1]
CONTRACT_8B = REPO / "targets/8b/contract.json"
CONTRACT_70B = REPO / "targets/70b/contract.json"


def test_assistant_loss_mask_skips_structural_tokens() -> None:
    ids = [BOS_ID, 65, 10, 66, ASSISTANT_START_ID, 11, 12, ASSISTANT_END_ID, 99]
    mask = assistant_loss_mask(ids)
    assert mask == [0, 0, 0, 0, 0, 1, 1, 0, 0]


def test_split_assignment_is_stable() -> None:
    first = assign_split("smoltalk2", "row-1", seed=1)
    second = assign_split("smoltalk2", "row-1", seed=1)
    other = assign_split("smoltalk2", "row-2", seed=1)
    assert first == second
    assert first in {"overfit", "validation", "heldout_test", "train"}
    assert other in {"overfit", "validation", "heldout_test", "train"}


def test_bounded_sampler_keeps_smallest_ids() -> None:
    from apertus_eagle.prepare_data import _consider, _heap_rows

    heap: list = []
    for index in range(20, 0, -1):
        _consider(heap, {"id": f"{index:02d}", "source": "s"}, limit=3)
    rows = _heap_rows(heap)
    assert [row["id"] for row in rows] == ["01", "02", "03"]


@pytest.mark.parametrize(
    ("contract_path", "hidden", "vllm_aux", "hf_aux"),
    [(CONTRACT_8B, 4096, [2, 16, 29], [1, 15, 28]), (CONTRACT_70B, 8192, [2, 40, 77], [1, 39, 76])],
)
def test_torchspec_overrides_come_from_the_selected_contract(
    contract_path: Path, hidden: int, vllm_aux: list[int], hf_aux: list[int]
) -> None:
    contract = load_contract(contract_path)
    keys = target_keys(contract)
    overrides = torchspec_model_overrides(contract)
    assert keys["output_vocab_size"] == 131072
    assert keys["input_vocab_size"] == 266752
    assert keys["lm_head_shape"] == [131072, hidden]
    assert overrides["lm_head_key"] == "lm_head.weight"
    assert overrides["norm_key"] == "model.language_model.norm.weight"
    assert overrides["embedding_key"] == "model.language_model.embed_tokens.weight"
    assert overrides["aux_hidden_states_layers_vllm"] == vllm_aux
    assert overrides["aux_hidden_states_layers_torchspec"] == hf_aux


def test_draft_templates_satisfy_their_own_contract() -> None:
    e31_8b = validate_draft_config_path(
        REPO / "methods/eagle/configs/8b/draft-e31-config.json", CONTRACT_8B
    )
    e31 = validate_draft_config_path(
        REPO / "methods/eagle/configs/70b/draft-e31-config.json", CONTRACT_70B
    )
    e3 = validate_draft_config_path(
        REPO / "methods/eagle/configs/70b/draft-e3-config.json", CONTRACT_70B
    )
    assert e31_8b["algorithm"] == "eagle31"
    assert e31["algorithm"] == "eagle31"
    assert e3["algorithm"] == "eagle3"


def test_double_bos_fails_the_serving_policy() -> None:
    assert serving_bos_policy([BOS_ID, 65]).get("pass") is True
    assert serving_bos_policy([BOS_ID, BOS_ID, 65])["double_bos"] is True
    assert serving_bos_policy([BOS_ID, BOS_ID, 65])["pass"] is False


def test_mix_omitting_bos_does_not_block_training() -> None:
    from apertus_eagle.parity import mix_omits_leading_bos
    from apertus_eagle.renderer import prepare_serving_messages

    rendered = [BOS_ID, 65, 10, 66]
    stored = [65, 10, 66]
    assert mix_omits_leading_bos(rendered, stored) is True
    assert mix_omits_leading_bos(rendered, rendered) is False
    assert mix_omits_leading_bos(rendered, [BOS_ID, 99]) is False
    filtered, thinking = prepare_serving_messages(
        [
            {"role": "developer", "content": "Deliberation: enabled\nTool Capabilities: disabled"},
            {"role": "user", "content": "Hi"},
        ]
    )
    assert thinking is True
    assert [m["role"] for m in filtered] == ["user"]


def test_conversations_export_restores_last_assistant(tmp_path: Path) -> None:
    from apertus_eagle.prepare_data import conversations_from_sampled

    source = tmp_path / "overfit.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "row-1",
                "messages": [
                    {"role": "developer", "content": "Deliberation: disabled"},
                    {"role": "user", "content": "Name the Swiss capital."},
                ],
                "original_assistant": "Bern",
            }
        )
        + "\n"
    )
    dest = tmp_path / "conversations.jsonl"
    assert conversations_from_sampled(source, dest) == 1
    row = json.loads(dest.read_text())
    assert [m["role"] for m in row["messages"]] == ["developer", "user", "assistant"]
    assert row["messages"][-1]["content"] == "Bern"


def test_parse_rendered_mix_row_keeps_prompt_only() -> None:
    text = (
        "<|developer_start|>Deliberation: disabled<|developer_end|>"
        "<|user_start|>Name the Swiss capital.<|user_end|>"
        "<|assistant_start|>Bern<|assistant_end|>"
    )
    messages = parse_rendered_text(text)
    prompt, assistant = prompt_and_last_assistant(messages)
    assert [m["role"] for m in prompt] == ["developer", "user"]
    assert assistant == "Bern"
    assert parse_rendered_text("<s>" + text)[0]["role"] == "developer"


def test_torchspec_conversation_keeps_assistant() -> None:
    from apertus_eagle.to_torchspec import sampled_row_to_conversations

    row = sampled_row_to_conversations(
        {
            "id": "abc",
            "messages": [{"role": "user", "content": "Hi"}],
            "original_assistant": "Hello",
            "source": "x",
            "domain": "chat_qa",
            "split": "overfit",
        }
    )
    assert [m["role"] for m in row["conversations"]] == ["user", "assistant"]
    assert row["conversations"][-1]["content"] == "Hello"


def test_reconstructed_mix_text_has_no_bos() -> None:
    from apertus_eagle.probe_renderer import reconstruct_mix_text

    text = reconstruct_mix_text(
        {
            "messages": [{"role": "user", "content": "Hi"}],
            "original_assistant": "Hello",
        }
    )
    assert text is not None
    assert not text.startswith("<s>")
    assert text.startswith("<|user_start|>")


def test_chat_template_token_ids_ignores_batch_encoding() -> None:
    from apertus_eagle.renderer import chat_template_token_ids

    class FakeTok:
        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            del messages, kwargs
            if tokenize:
                return {"input_ids": [99, 99], "attention_mask": [1, 1]}
            return "<s>hello"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            assert text == "<s>hello"
            return [1, 29706]

    assert chat_template_token_ids(FakeTok(), [{"role": "user", "content": "hello"}]) == [
        1,
        29706,
    ]


def test_draft_parameter_count_is_not_a_200m_head() -> None:
    config = _load_template("70b/draft-e31-config.json")
    parts = count_draft_parameters(config)
    # Target-width single-layer Llama EAGLE head, full output vocab.
    assert parts["embed_tokens"] == 131072 * 8192
    assert parts["lm_head"] == 131072 * 8192
    assert parts["fc"] == 3 * 8192 * 8192
    assert parts["excluding_embed_and_lm_head"] == 1_140_908_032
    assert parts["total"] == 3_288_391_680


def test_8b_draft_parameter_count_is_derived_from_the_8b_contract() -> None:
    parts = count_draft_parameters(_load_template("8b/draft-e31-config.json"))
    assert parts["embed_tokens"] == 131072 * 4096
    assert parts["fc"] == 3 * 4096 * 4096
    assert parts["excluding_embed_and_lm_head"] == 293_629_952


def _load_template(name: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "methods" / "eagle" / "configs" / name
    return json.loads(path.read_text())


def _drop_stub_modules(*prefixes: str) -> dict:
    import sys

    saved = {}
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            saved[name] = sys.modules.pop(name)
    return saved


def _restore_stub_modules(saved: dict, prefixes: tuple[str, ...]) -> None:
    import sys

    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)
    sys.modules.update(saved)


def test_import_stubs_survive_accelerate_find_spec() -> None:
    """Job 3491232: find_spec('wandb') raised because __spec__ was None."""
    import contextlib
    import importlib.metadata
    import importlib.util

    from apertus_eagle.import_stubs import ensure_import_stubs

    prefixes = ("wandb", "ray", "datasets", "pydantic", "numba")
    saved = _drop_stub_modules(*prefixes)
    try:
        ensure_import_stubs()
        for name in prefixes:
            spec = importlib.util.find_spec(name)
            assert spec is not None
            with contextlib.suppress(importlib.metadata.PackageNotFoundError):
                importlib.metadata.metadata(name)
        assert importlib.util.find_spec("ray.util.scheduling_strategies") is not None
    finally:
        _restore_stub_modules(saved, prefixes)


def test_import_stubs_keep_a_spec_bearing_wandb() -> None:
    import importlib.machinery
    import sys
    import types

    from apertus_eagle.import_stubs import ensure_import_stubs

    prefixes = ("wandb", "ray", "datasets", "pydantic", "numba")
    saved = _drop_stub_modules(*prefixes)
    try:
        kept = types.ModuleType("wandb")
        kept.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None, is_package=True)
        kept.__path__ = []
        kept.__version__ = "kept"
        sys.modules["wandb"] = kept
        ensure_import_stubs()
        assert sys.modules["wandb"] is kept
        assert sys.modules["wandb"].__version__ == "kept"
    finally:
        _restore_stub_modules(saved, prefixes)


def test_wandb_offline_stub_children_have_specs(tmp_path: Path) -> None:
    import importlib.util
    import sys

    pkg = tmp_path / "wandb"
    pkg.mkdir()
    source = (
        Path(__file__).resolve().parents[1] / "methods/eagle/apertus_eagle/wandb_offline_stub.py"
    )
    (pkg / "__init__.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    saved = _drop_stub_modules("wandb")
    sys.path.insert(0, str(tmp_path))
    try:
        import wandb

        assert wandb.__version__ == "0.0.0-stub"
        for name in ("wandb", "wandb.util", "wandb.env", "wandb.sdk", "wandb.sdk.lib"):
            spec = importlib.util.find_spec(name)
            assert spec is not None
    finally:
        if sys.path and sys.path[0] == str(tmp_path):
            sys.path.pop(0)
        _restore_stub_modules(saved, ("wandb",))


def test_aux_layers_follow_the_pinned_vllm_rule() -> None:
    assert derive_aux_layers(32) == {"vllm": [2, 16, 29], "hf_decoder_layer": [1, 15, 28]}
    assert derive_aux_layers(80) == {"vllm": [2, 40, 77], "hf_decoder_layer": [1, 39, 76]}


def test_contract_aux_layers_are_cross_checked_against_depth() -> None:
    contract = load_contract(CONTRACT_8B)
    assert aux_layer_ids(contract, "vllm") == [2, 16, 29]
    assert aux_layer_ids(contract, "hf") == [1, 15, 28]
    contract["target"]["eagle"]["default_aux_hidden_state_layer_ids"] = [2, 40, 77]
    with pytest.raises(ContractError, match="depth-derived"):
        aux_layer_ids(contract, "vllm")


def test_contract_selection_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APERTUS_EAGLE_CONTRACT", raising=False)
    with pytest.raises(ContractError, match="no target contract selected"):
        load_contract()
    monkeypatch.setenv("APERTUS_EAGLE_CONTRACT", str(CONTRACT_8B))
    assert load_contract()["target"]["text"]["num_hidden_layers"] == 32


def test_committed_8b_draft_config_is_the_contract_derivation() -> None:
    from apertus_eagle.draft_config import build_draft_config

    derived = build_draft_config(load_contract(CONTRACT_8B), "eagle31")
    assert derived == _load_template("8b/draft-e31-config.json")
    e3 = build_draft_config(load_contract(CONTRACT_8B), "eagle3")
    assert (e3["fc_norm"], e3["norm_output"]) == (False, False)
    assert {
        k: v for k, v in e3.items() if k not in ("fc_norm", "norm_output", "apertus_target")
    } == {k: v for k, v in derived.items() if k not in ("fc_norm", "norm_output", "apertus_target")}


def test_feature_cache_is_refused_after_target_layout_or_corpus_change() -> None:
    from apertus_eagle.contract import target_identity
    from apertus_eagle.features import check_cache, feature_contract

    contract = load_contract(CONTRACT_8B)
    manifest = {
        "path": "cache",
        "status": "complete",
        "target": target_identity(contract),
        "feature_contract": feature_contract(contract),
        "corpus_sha256": "abc",
    }
    check_cache(manifest, contract, "abc")
    with pytest.raises(ValueError, match="corpus digest"):
        check_cache(manifest, contract, "other")
    with pytest.raises(ValueError, match="target identity"):
        check_cache(manifest, load_contract(CONTRACT_70B), "abc")
    stale = {
        **manifest,
        "feature_contract": {**manifest["feature_contract"], "vllm_aux_ids": [2, 40, 77]},
    }
    with pytest.raises(ValueError, match="feature contract"):
        check_cache(stale, contract, "abc")
    with pytest.raises(ValueError, match="status"):
        check_cache({**manifest, "status": "partial"}, contract, "abc")


def test_training_sequence_is_the_served_stream_with_generated_loss() -> None:
    from apertus_eagle.features import training_sequence

    row = {
        "prompt_ids": [BOS_ID, 65, 10, 66, ASSISTANT_START_ID],
        "generated_ids": [11, 12, ASSISTANT_END_ID],
    }
    ids, mask = training_sequence(row, max_seq_length=4096)
    assert ids == [BOS_ID, 65, 10, 66, ASSISTANT_START_ID, 11, 12, ASSISTANT_END_ID]
    # TorchSpec preprocessing zeroes the final position: nothing follows it.
    assert mask == [0, 0, 0, 0, 0, 1, 1, 0]
    ids, mask = training_sequence(row, max_seq_length=6)
    assert len(ids) == 6 and mask == [0, 0, 0, 0, 0, 0]

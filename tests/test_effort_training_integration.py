"""Validate the data contract against the actual optional upstream trainer."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("speculators")

from safetensors.torch import save_file  # noqa: E402
from speculators.train.data import ArrowDataset  # noqa: E402

from apertus_bench.effort_training import package_datasets  # noqa: E402


def test_all_role_datasets_load_through_upstream_with_exact_budgets_and_dev_boundary(tmp_path):
    (tmp_path / "training-data").mkdir()
    samples = []
    for split in ("train", "development"):
        for effort in ("low", "medium", "high"):
            path = tmp_path / f"{split}-{effort}.safetensors"
            save_file(
                {
                    "token_ids": torch.arange(10),
                    "hidden_states": torch.zeros(10, 6, 16, dtype=torch.bfloat16),
                },
                str(path),
            )
            samples.append(
                {
                    "id": split,
                    "effort": effort,
                    "split": split,
                    "token_ids": list(range(10)),
                    "prompt_tokens": 2,
                    "hidden_states_path": str(path),
                }
            )
    config = {"supervised_tokens_per_head": 120, "epochs": 4, "seed": 0}
    package_datasets(tmp_path, config, samples)
    for role in ("low", "medium", "high", "mixed"):
        path = tmp_path / "training-data/roles" / role
        manifest = json.loads((path / "training-manifest.json").read_text())
        train = ArrowDataset(
            8192, path, on_missing="raise", train_ratio=manifest["train_data_ratio"], split="train"
        )
        dev = ArrowDataset(
            8192, path, on_missing="raise", train_ratio=manifest["train_data_ratio"], split="val"
        )
        assert sum(int(train[i]["loss_mask"].sum()) for i in range(len(train))) == 30
        assert len(dev) == (3 if role == "mixed" else 1)
        assert train[0]["hidden_states"].shape == (10, 80)
        assert train[0]["verifier_last_hidden_states"].shape == (10, 16)


def test_sdpa_preserves_dflash_loss_and_backward():
    from speculators.losses import resolve_loss_config
    from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
    from transformers import Qwen3Config

    def model(backend):
        config = DFlashSpeculatorConfig(
            transformer_layer_config=Qwen3Config(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=64,
                _attn_implementation=backend,
            ),
            draft_vocab_size=64,
            block_size=4,
            aux_hidden_state_layer_ids=[0, 1],
            mask_token_id=0,
        )
        result = DFlashDraftModel(config).train()
        torch.nn.init.normal_(result.embed_tokens.weight)
        torch.nn.init.normal_(result.lm_head.weight)
        torch.nn.init.normal_(result.verifier_lm_head.weight)
        torch.nn.init.ones_(result.verifier_norm.weight)
        return result

    torch.manual_seed(7)
    eager, sdpa = model("eager"), model("sdpa")
    sdpa.load_state_dict(eager.state_dict())
    batch = {
        "hidden_states": torch.randn(1, 16, 32),
        "verifier_last_hidden_states": torch.randn(1, 16, 16),
        "input_ids": torch.randint(0, 64, (1, 16)),
        "loss_mask": torch.ones(1, 16),
        "document_ids": torch.zeros(1, 16, dtype=torch.long),
    }
    losses = []
    for draft in (eager, sdpa):
        torch.manual_seed(0)
        _, loss, _ = draft(
            **batch,
            loss_config=resolve_loss_config("ce", "eager"),
            max_anchors=4,
            per_position_loss_weight="fixed-exp-decay",
            gamma=4.0,
        )
        loss.backward()
        losses.append(loss.detach())
        assert torch.isfinite(loss)
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(eager.fc.weight.grad, sdpa.fc.weight.grad, atol=1e-6, rtol=1e-4)

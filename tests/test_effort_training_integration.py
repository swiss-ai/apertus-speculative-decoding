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

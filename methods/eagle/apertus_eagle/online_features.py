"""Target features computed on the fly instead of read from a disk cache.

``apertus_eagle.features`` stores the target's auxiliary and final hidden
states for every training token (32.8 KB/token on the 8B). That is fine for
10k conversations and impossible for Open-PerfectBlend (~0.8B tokens/epoch,
~25 TB). This source runs the frozen target on the training GPU right before
each draft step and hands the trainer the same record a cache file holds, so
``train_rollout`` trains on it unchanged:

    {"input_ids", "loss_mask", "hidden_states", "last_hidden_states"}

The target is the contract checkpoint, captured with the same forward hooks
(``features.FeatureCapture``) whose alignment with vLLM the parity step
verified. Rows are read lazily through a byte-offset index, so a corpus far
larger than host memory is fine.

Accepted row formats, one JSON object per line:
- ``generate_targets`` output: ``prompt_ids`` + ``generated_ids`` (loss on the
  generated turn only, as ``features.training_sequence``);
- pre-tokenized: ``input_ids`` + ``loss_mask`` (e.g. every assistant turn).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from apertus_eagle.contract import aux_layer_ids, num_decoder_layers
from apertus_eagle.features import FeatureCapture, load_teacher, training_sequence


def row_sequence(row: dict[str, Any], max_seq_length: int) -> tuple[list[int], list[int]]:
    if "input_ids" in row:
        ids = list(row["input_ids"])[:max_seq_length]
        mask = list(row["loss_mask"])[:max_seq_length]
        if len(ids) != len(mask):
            raise ValueError(f"row {row.get('id')!r}: input_ids and loss_mask lengths differ")
        if mask:
            mask[-1] = 0  # the last position has no next token to predict
        return ids, mask
    return training_sequence(row, max_seq_length)


class LineIndex:
    """Byte offsets of the non-empty lines of several JSONL files."""

    def __init__(self, paths: list[Path], limit: int | None = None):
        self.paths = paths
        self.entries: list[tuple[int, int]] = []
        self.digests = {}
        for file_index, path in enumerate(paths):
            digest = hashlib.sha256()
            offset = 0
            with path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    if line.strip() and (limit is None or len(self.entries) < limit):
                        self.entries.append((file_index, offset))
                    offset += len(line)
            self.digests[str(path)] = digest.hexdigest()
        self.handles = [path.open("rb") for path in paths]

    def __len__(self) -> int:
        return len(self.entries)

    def read(self, index: int) -> dict[str, Any]:
        file_index, offset = self.entries[index]
        handle = self.handles[file_index]
        handle.seek(offset)
        return json.loads(handle.readline())


class Teacher:
    """The frozen target with the parity-checked capture hooks, shared by all sources."""

    def __init__(self, contract: dict[str, Any], device: str = "cuda"):
        import torch

        self.model, self.loader_class = load_teacher(contract["source"]["authorized_checkpoint"])
        self.model.eval()
        self.model.requires_grad_(False)
        self.capture = FeatureCapture(
            self.model, aux_layer_ids(contract, "hf"), num_decoder_layers(contract)
        )
        self.device = torch.device(device)

    def features(self, ids: list[int]):
        import torch

        with torch.inference_mode():
            input_ids = torch.tensor([ids], device=self.device)
            hidden, last, _logits = self.capture.run(input_ids)
        # Leave inference mode: the trainer builds autograd graphs on top of these.
        return hidden.clone().to(torch.bfloat16), last.clone().to(torch.bfloat16)


class OnlineFeatures:
    """Drop-in replacement for ``train_rollout.FeatureCache``."""

    def __init__(
        self,
        paths: list[Path],
        teacher: Teacher,
        *,
        max_seq_length: int,
        limit: int | None = None,
    ):
        self.root = paths[0].parent
        self.index = LineIndex(paths, limit)
        self.teacher = teacher
        self.max_seq_length = max_seq_length
        self.skipped: dict[str, int] = {"no_loss_tokens": 0}
        digest = hashlib.sha256(json.dumps(self.index.digests, sort_keys=True).encode()).hexdigest()
        self.manifest = {"corpus_sha256": digest, "files": self.index.digests, "mode": "online"}

    def __len__(self) -> int:
        return len(self.index)

    def load(self, index: int) -> dict[str, Any]:
        import torch

        ids, mask = row_sequence(self.index.read(index), self.max_seq_length)
        if not any(mask):
            self.skipped["no_loss_tokens"] += 1
        hidden, last = self.teacher.features(ids)
        device = self.teacher.device
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
            "loss_mask": torch.tensor(mask, dtype=torch.long, device=device),
            "hidden_states": hidden,
            "last_hidden_states": last,
        }

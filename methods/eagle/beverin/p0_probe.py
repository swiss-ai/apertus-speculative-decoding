"""P0 probe for EAGLE training on Beverin (MI300A): imports, teacher, BLAS.

Run inside the ROCm container with the training PYTHONPATH (p0-node.sh):
    python3 p0_probe.py --contract targets/8b/contract.json --rows HELDOUT.jsonl --output P0.json

1. Imports: the Swiss AI transformers fork (Apertus1p5ForConditionalGeneration),
   the pinned TorchSpec draft/objective/optimizer, flex_attention, and a small
   torch.compile (Triton) kernel on the GPU.
2. Loads the Apertus 1.5 8B teacher as training does (online_features.Teacher).
3. Computes teacher features for the first --samples held-out rows with the
   training batch size under hipBLASLt (default) and rocBLAS, and reports the
   time per backend and their agreement (relative L2, bf16 noise expected).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {"machine": platform.machine()}

    from apertus_eagle.train_rollout import _prepare_imports

    report["torchspec_root"] = str(_prepare_imports())
    import torch
    import transformers

    report["versions"] = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "transformers": transformers.__version__,
        "transformers_file": transformers.__file__,
        "gpu": torch.cuda.get_device_name(0),
        "gpus": torch.cuda.device_count(),
    }
    from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
    from torchspec.models.draft import AutoEagle3DraftModel  # noqa: F401
    from torchspec.models.eagle3 import Eagle3Model  # noqa: F401
    from torchspec.training.optimizer import BF16Optimizer  # noqa: F401

    compiled = torch.compile(lambda x: torch.nn.functional.silu(x) * 2 + 1)
    x = torch.randn(4096, device="cuda")
    started = time.time()
    torch.testing.assert_close(compiled(x), torch.nn.functional.silu(x) * 2 + 1)
    report["torch_compile_s"] = round(time.time() - started, 1)
    report["fork_class"] = hasattr(transformers, "Apertus1p5ForConditionalGeneration")
    report["imports"] = "ok"
    print(json.dumps(report), flush=True)

    from apertus_eagle.contract import load_contract
    from apertus_eagle.online_features import OnlineFeatures, Teacher

    contract = load_contract(args.contract)
    started = time.time()
    teacher = Teacher(contract)
    report["teacher_load_s"] = round(time.time() - started, 1)
    report["teacher_class"] = teacher.loader_class
    online = OnlineFeatures(
        [args.rows],
        teacher,
        max_seq_length=args.max_seq_length,
        limit=args.samples,
        max_batch_tokens=args.batch_tokens,
    )
    indices = list(range(len(online)))
    results = {}
    for backend in ("cublaslt", "cublas"):
        torch.backends.cuda.preferred_blas_library(backend)
        online.load_many(indices[:4])  # warm-up
        torch.cuda.synchronize()
        started = time.time()
        records = online.load_many(indices)
        torch.cuda.synchronize()
        seconds = time.time() - started
        tokens = sum(int(r["input_ids"].numel()) for r in records)
        results[backend] = records
        report[f"teacher_{backend}"] = {
            "seconds": round(seconds, 2),
            "tokens": tokens,
            "tokens_per_s": round(tokens / seconds, 1),
            "active": str(torch.backends.cuda.preferred_blas_library()),
        }
        print(json.dumps({backend: report[f"teacher_{backend}"]}), flush=True)
    for key in ("hidden_states", "last_hidden_states"):
        num = sum(
            float((a[key].float() - b[key].float()).norm() ** 2)
            for a, b in zip(results["cublaslt"], results["cublas"], strict=True)
        )
        den = sum(float(a[key].float().norm() ** 2) for a in results["cublaslt"])
        report[f"relative_l2_lt_vs_rb_{key}"] = (num / den) ** 0.5
    report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

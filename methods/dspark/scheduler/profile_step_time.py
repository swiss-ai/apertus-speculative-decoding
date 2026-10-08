"""Forward-pass time of the target against batch tokens, for the DSpark
confidence-scheduled verification (serving/patches/vllm-apertus-dspark-confidence-verify.patch).

Builds the plain target with the serving engine settings and times the model
runner's dummy step (short contexts, so mostly the GEMMs) at each batch size.
The scheduler interpolates this table and learns the rest of the step time
(attention over long contexts, drafting, scheduling) online.

    python3 profile_step_time.py OUT.json [--model PATH] [--repeats 20]

Runs inside the serving container on one GPU (a node-sweep `command` arm).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

SIZES = [1, 8, 16, 32, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024,
         1280, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]  # fmt: skip
DEFAULT_MODEL = "/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B"


def time_steps(worker, sizes: list[int], repeats: int) -> dict[int, float]:
    """Runs in the worker process: mean wall time of one dummy step per size."""
    import torch

    runner = worker.model_runner
    timings: dict[int, float] = {}
    for num_tokens in sizes:
        for _ in range(3):
            runner._dummy_run(num_tokens)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            runner._dummy_run(num_tokens)
        torch.cuda.synchronize()
        timings[num_tokens] = (time.perf_counter() - started) * 1000.0 / repeats
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    args = parser.parse_args()

    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=32768,
        enable_prefix_caching=False,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )
    sizes = [size for size in SIZES if size <= args.max_num_batched_tokens]
    (timings,) = llm.collective_rpc(time_steps, args=(sizes, args.repeats))
    import torch
    import vllm

    result = {
        "tokens": sizes,
        "step_ms": [round(timings[size], 4) for size in sizes],
        "model": args.model,
        "repeats": args.repeats,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu": torch.cuda.get_device_name(0),
        "vllm": vllm.__version__,
        "host": platform.node(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

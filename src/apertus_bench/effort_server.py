"""Launch vLLM with a reproducible date in its native GPT-OSS Harmony renderer.

GPT-OSS chat requests bypass the Hugging Face chat template. This small, recorded
overlay changes only get_system_message's default start_date, not model sampling.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import runpy
import sys
from datetime import datetime
from pathlib import Path


def pin_harmony_date(module, date: str):
    datetime.strptime(date, "%Y-%m-%d")
    original = module.get_system_message
    signature = inspect.signature(original)
    if "start_date" not in signature.parameters:
        raise ValueError("this vLLM Harmony renderer has no supported start_date override")

    @functools.wraps(original)
    def fixed(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        if bound.arguments.get("start_date") is None:
            bound.arguments["start_date"] = date
        return original(*bound.args, **bound.kwargs)

    module.get_system_message = fixed
    return original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-date", required=True)
    parser.add_argument("--provenance-file", type=Path, required=True)
    args, vllm_args = parser.parse_known_args()
    if vllm_args and vllm_args[0] == "--":
        vllm_args = vllm_args[1:]
    from vllm.entrypoints.openai.parser import harmony_utils

    original = pin_harmony_date(harmony_utils, args.fixed_date)
    with args.provenance_file.open("x") as sink:
        json.dump(
            {
                "fixed_date": args.fixed_date,
                "function": f"{original.__module__}.{original.__name__}",
                "original_source_sha256": hashlib.sha256(
                    inspect.getsource(original).encode()
                ).hexdigest(),
                "overlay_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            sink,
            indent=2,
        )
    sys.argv = ["vllm.entrypoints.openai.api_server", *vllm_args]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()

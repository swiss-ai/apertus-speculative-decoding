"""Size Qwen3.5-9B TP1 GDN and BF16 target+MTP KV pools for the pinned SGLang image."""

import argparse
import json
from pathlib import Path


def ledger(mode, budget_gib):
    layers, slots, value_heads, key_heads, dim = 24, 65, 32, 16, 128
    verify_slots, ring_length, conv_history = 4, 16, 3
    temporal = layers * slots * value_heads * dim * dim * 4
    conv_channels = (2 * key_heads + value_heads) * dim
    conv_position = layers * slots * conv_channels * 2
    state = {
        "persistent_temporal": temporal,
        "persistent_conv": conv_position * conv_history,
        "intermediate_conv": conv_position * (conv_history + verify_slots - 1),
    }
    if mode == "snapshot":
        state["intermediate_temporal"] = verify_slots * temporal
    elif mode == "compact":
        # High and low BF16 parts for keys and updates, plus FP32 log decay.
        record_size = (key_heads + value_heads) * dim * 2 * 2 + value_heads * 4
        state["replay_records"] = layers * slots * ring_length * record_size
    else:
        raise ValueError(f"unknown verifier: {mode}")
    # Eight target attention layers and one MTP layer, each with 4 KV heads of dim 256.
    kv_bytes_per_token = (8 + 1) * 2 * 4 * 256 * 2
    budget = budget_gib * 2**30
    # With page_size=1 SGLang allocates one extra KV slot beyond the user cap.
    max_tokens = (budget - sum(state.values())) // kv_bytes_per_token - 1
    if max_tokens < 3072:
        raise ValueError("budget is too small to run a full context")
    kv_total = (max_tokens + 1) * kv_bytes_per_token
    return dict(
        mode=mode,
        budget_gib=budget_gib,
        state_bytes=state,
        kv_bytes_per_token=kv_bytes_per_token,
        kv_bytes=kv_total,
        max_total_tokens=max_tokens,
        unallocated_bytes=budget - sum(state.values()) - kv_total,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["snapshot", "compact"])
    parser.add_argument("budget_gib", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = ledger(args.mode, args.budget_gib)
    with args.output.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(data["max_total_tokens"])

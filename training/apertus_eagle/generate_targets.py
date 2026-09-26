"""A2: regenerate assistant turns with the frozen target (greedy, natural EOS).

Runs inside the serving image on one allocated GPU. For every split row the
prompt is the row's messages up to the final assistant turn, rendered by the
serving chat template with the serving default (``Deliberation: disabled``,
no tools). The training sequence is the exact served token stream,
``prompt_ids + generated_ids``; nothing is detokenized and re-encoded.

Before generating, the first ``--render-checks`` prompts are also sent
through ``LLM.chat`` (the OpenAI server's rendering path) and their prompt
token ids must equal ours exactly.

    python3 -m apertus_eagle.generate_targets --contract .../compatibility.json \
      --input results/eagle/data/train.jsonl --output results/eagle/8b/data/generated/train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from apertus_eagle.contract import load_contract, target_identity
from apertus_eagle.renderer import (
    ASSISTANT_START_ID,
    BOS_ID,
    chat_template_token_ids,
    prepare_serving_messages,
)

GENERATION_POLICY = {
    "temperature": 0.0,
    "top_p": 1.0,
    "stop": "natural EOS (generation_config eos ids)",
    "enable_thinking": False,
    "thinking_policy": "serving default: chat template renders 'Deliberation: disabled'",
    "tools": None,
    "developer_messages": "dropped; the template injects its own developer block",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Messages before the final assistant turn, developer spans removed."""
    messages = list(row.get("messages") or row.get("conversations") or [])
    while messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    filtered, _thinking = prepare_serving_messages(messages)
    if not filtered or filtered[-1].get("role") != "user":
        raise ValueError(f"row {row.get('id')!r} does not end with a user turn")
    return filtered


def render_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    ids = chat_template_token_ids(
        tokenizer, messages, add_generation_prompt=True, enable_thinking=False
    )
    if not ids or ids[0] != BOS_ID or (len(ids) > 1 and ids[1] == BOS_ID):
        raise ValueError("rendered prompt must start with exactly one BOS")
    if ids[-1] != ASSISTANT_START_ID:
        raise ValueError(f"generation prompt ends with {ids[-1]}, not <|assistant_start|>")
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-generate-targets")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--render-checks", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    contract = load_contract(args.contract)
    identity = target_identity(contract)
    model_path = contract["source"]["authorized_checkpoint"]
    eos_ids = contract["target"]["tokens"]["generation_eos_token_id"]

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_seq_length + 64,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        seed=0,
    )
    tokenizer = llm.get_tokenizer()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "target": identity,
        "contract": contract["_path"],
        "policy": GENERATION_POLICY,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "eos_token_ids": eos_ids,
        "splits": {},
    }

    render_checked = 0
    render_failures: list[dict[str, Any]] = []
    for input_path in args.input:
        rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
        if args.limit:
            rows = rows[: args.limit]
        prepared: list[tuple[dict[str, Any], list[dict[str, Any]], list[int]]] = []
        skipped: dict[str, int] = {"prompt_too_long": 0, "unrenderable": 0}
        for row in rows:
            try:
                messages = prompt_messages(row)
                ids = render_prompt(tokenizer, messages)
            except (ValueError, TypeError) as error:
                skipped["unrenderable"] += 1
                print(json.dumps({"skip": row.get("id"), "error": str(error)}), flush=True)
                continue
            if len(ids) > args.max_prompt_tokens:
                skipped["prompt_too_long"] += 1
                continue
            prepared.append((row, messages, ids))

        # Serving-renderer parity: LLM.chat renders like the OpenAI server.
        budget = max(0, args.render_checks - render_checked)
        multi = [item for item in prepared if len(item[1]) > 1]
        single = [item for item in prepared if len(item[1]) == 1]
        checks = (multi[: budget // 2] + single)[:budget]
        checks += [item for item in multi[budget // 2 :]][: budget - len(checks)]
        if checks:
            outputs = llm.chat(
                [messages for _row, messages, _ids in checks],
                SamplingParams(max_tokens=1, temperature=0.0),
                chat_template_content_format="string",
                chat_template_kwargs={"enable_thinking": False},
                use_tqdm=False,
            )
            for (row, messages, ids), output in zip(checks, outputs):
                served = list(output.prompt_token_ids)
                render_checked += 1
                if served != ids:
                    first = next(
                        (i for i, (a, b) in enumerate(zip(served, ids)) if a != b),
                        min(len(served), len(ids)),
                    )
                    render_failures.append(
                        {"id": row.get("id"), "first_diff": first,
                         "served_len": len(served), "ours_len": len(ids),
                         "n_turns": len(messages)}
                    )
        if render_failures:
            (args.output_dir / "render-failures.json").write_text(
                json.dumps(render_failures, indent=2) + "\n"
            )
            raise SystemExit(f"serving renderer mismatch on {len(render_failures)} prompts")

        params = [
            SamplingParams(
                temperature=0.0,
                max_tokens=min(args.max_new_tokens, args.max_seq_length - len(ids)),
                stop_token_ids=list(eos_ids),
                skip_special_tokens=False,
                seed=0,
            )
            for _row, _messages, ids in prepared
        ]
        started = time.time()
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for _row, _messages, ids in prepared],
            params,
            use_tqdm=True,
        )
        elapsed = time.time() - started
        dest = args.output_dir / input_path.name
        finish: dict[str, int] = {}
        generated_tokens = 0
        with dest.open("w", encoding="utf-8") as handle:
            for (row, messages, ids), output in zip(prepared, outputs):
                completion = output.outputs[0]
                gen_ids = list(completion.token_ids)
                reason = str(completion.finish_reason)
                finish[reason] = finish.get(reason, 0) + 1
                generated_tokens += len(gen_ids)
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "split": row.get("split") or input_path.stem,
                            "domain": row.get("domain"),
                            "source": row.get("dataset_name") or row.get("source"),
                            "messages": messages,
                            "prompt_ids": ids,
                            "generated_ids": gen_ids,
                            "generated_text": completion.text,
                            "finish_reason": reason,
                            "stop_reason": completion.stop_reason,
                        }
                    )
                    + "\n"
                )
        summary["splits"][input_path.stem] = {
            "input": str(input_path),
            "input_sha256": file_sha256(input_path),
            "output": str(dest),
            "output_sha256": file_sha256(dest),
            "rows_in": len(rows),
            "rows_out": len(prepared),
            "skipped": skipped,
            "finish_reasons": finish,
            "generated_tokens": generated_tokens,
            "generate_seconds": round(elapsed, 1),
            "generated_tokens_per_second": round(generated_tokens / max(elapsed, 1e-6), 1),
        }
        print(json.dumps({input_path.stem: summary["splits"][input_path.stem]}), flush=True)

    summary["render_parity"] = {
        "checked": render_checked,
        "failures": len(render_failures),
        "rule": "LLM.chat prompt_token_ids == chat_template(tokenize=False)+encode(add_special_tokens=False)",
    }
    (args.output_dir / "generation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["render_parity"]), flush=True)


if __name__ == "__main__":
    main()

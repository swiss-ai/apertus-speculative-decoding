"""Compare mix text, local chat-template rendering, and optional serving tokenize.

Numeric hidden-state comparison still needs a GPU job. This probe is the
tokenizer/BOS gate on at least 32 conversations.

Mix ``text`` may omit the BOS that ``chat_template.jinja`` emits. That is
recorded and does not block training when the training renderer is the
serving chat template with ``add_special_tokens=False``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apertus_eagle.contract import REPO_ROOT
from apertus_eagle.parity import mix_omits_leading_bos, serving_bos_policy
from apertus_eagle.parse_rendered import parse_rendered_text
from apertus_eagle.prepare_data import DEFAULT_PARQUET, lookup_mix_text
from apertus_eagle.renderer import BOS_ID, prepare_serving_messages

DEFAULT_TARGET = Path(
    "/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-70B"
)


def _load_tokenizer(model_path: Path, chat_template_path: Path | None) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if chat_template_path is not None and chat_template_path.is_file():
        tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    return tokenizer


def _as_id_list(token_ids: Any) -> list[int]:
    from apertus_eagle.renderer import coerce_token_ids

    return coerce_token_ids(token_ids)


def _render_ids(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[list[int], bool]:
    from apertus_eagle.renderer import chat_template_token_ids

    serving_messages, enable_thinking = prepare_serving_messages(messages)
    ids = chat_template_token_ids(
        tokenizer,
        serving_messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return ids, enable_thinking


def compare_one(tokenizer: Any, text: str) -> dict[str, Any]:
    messages = parse_rendered_text(text)
    rendered_ids, enable_thinking = _render_ids(tokenizer, messages)
    stored_ids = _as_id_list(tokenizer.encode(text, add_special_tokens=False))
    bos = serving_bos_policy(rendered_ids)
    bos_only = mix_omits_leading_bos(rendered_ids, stored_ids)
    ids_equal = rendered_ids == stored_ids
    return {
        "n_messages": len(messages),
        "roles": [message["role"] for message in messages],
        "rendered_n_tokens": len(rendered_ids),
        "stored_n_tokens": len(stored_ids),
        "ids_equal": ids_equal,
        "mix_omits_leading_bos": bos_only,
        "mix_starts_with_s": text.startswith("<s>"),
        "rendered_starts_with_bos": bos["starts_with_bos"],
        "rendered_double_bos": bos["double_bos"],
        "stored_starts_with_bos": bool(stored_ids) and stored_ids[0] == BOS_ID,
        "enable_thinking": enable_thinking,
        "dropped_developer": any(m["role"] == "developer" for m in messages),
        "rendered_prefix_ids": rendered_ids[:8],
        "stored_prefix_ids": stored_ids[:8],
        "prefix_match_len": _prefix_len(rendered_ids, stored_ids),
        "training_ok": (not bos["double_bos"]) and bos["starts_with_bos"],
    }


def _prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


def reconstruct_mix_text(row: dict[str, Any]) -> str | None:
    """Rebuild the mix control-span string without a leading BOS.

    Sampled split JSONL stores messages plus ``original_assistant`` and drops
    the parquet ``text``. The mix renderer does not emit ``<s>``; the serving
    chat template does.
    """
    messages = list(row.get("messages") or [])
    assistant = row.get("original_assistant")
    if assistant is not None:
        messages = [*messages, {"role": "assistant", "content": assistant}]
    if not messages:
        return None
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = message.get("content")
        if not role or content is None:
            return None
        parts.append(f"<|{role}_start|>{content}<|{role}_end|>")
    return "".join(parts)


def _text_from_row(payload: dict[str, Any], mix_text: dict[str, str]) -> str | None:
    text = payload.get("text")
    if text:
        return str(text)
    row_id = str(payload.get("id") or "")
    if row_id and row_id in mix_text:
        return mix_text[row_id]
    return reconstruct_mix_text(payload)


def probe_file(
    tokenizer: Any,
    path: Path,
    *,
    limit: int = 32,
    mix_text: dict[str, str] | None = None,
) -> dict[str, Any]:
    mix_text = mix_text or {}
    rows = []
    skipped_no_text = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        text = _text_from_row(payload, mix_text)
        if not text:
            skipped_no_text += 1
            continue
        report = compare_one(tokenizer, text)
        report["id"] = payload.get("id")
        rows.append(report)
        if len(rows) >= limit:
            break
    equal = sum(1 for row in rows if row["ids_equal"])
    bos_only = sum(1 for row in rows if row["mix_omits_leading_bos"])
    no_double_bos = all(not row["rendered_double_bos"] for row in rows)
    rendered_always_bos = all(row["rendered_starts_with_bos"] for row in rows) if rows else False
    training_ok = all(row["training_ok"] for row in rows) if rows else False
    # Training unblocks when the serving renderer is BOS-correct. Mix identity
    # is reported separately; a BOS-only mix gap is expected.
    passed = bool(rows) and training_ok and no_double_bos and rendered_always_bos
    return {
        "n": len(rows),
        "skipped_no_text": skipped_no_text,
        "ids_equal": equal,
        "ids_mismatch": len(rows) - equal,
        "mix_omits_leading_bos": bos_only,
        "double_bos": sum(1 for row in rows if row["rendered_double_bos"]),
        "rendered_always_bos": rendered_always_bos,
        "stored_always_bos": (
            all(row["stored_starts_with_bos"] for row in rows) if rows else False
        ),
        "training_ok": training_ok,
        "pass": passed,
        "pass_rule": (
            "serving renderer starts with BOS and has no double BOS. Mix text may "
            "omit BOS and already include developer/system spans the template injects; "
            "that does not block training."
        ),
        "examples": rows[:8],
    }


def _collect_ids(path: Path, *, limit: int) -> set[str]:
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not payload.get("text") and payload.get("id"):
            ids.add(str(payload["id"]))
        if len(ids) >= limit:
            break
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="apertus-eagle-probe-renderer")
    parser.add_argument("--model", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--chat-template",
        type=Path,
        default=Path(__file__).resolve().parent / "chat_template.jinja",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="JSONL with a 'text' field (raw mix rows) or sampled splits with id",
    )
    parser.add_argument("--parquet-dir", type=Path)
    parser.add_argument(
        "--mix-text-json",
        type=Path,
        help="id→mix text sidecar written by login python + pyarrow",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/70b/eagle/preflight/renderer-parity.json",
    )
    args = parser.parse_args(argv)
    tokenizer = _load_tokenizer(args.model, args.chat_template)
    mix_text: dict[str, str] = {}
    mix_source = "reconstruct_mix_text"
    if args.parquet_dir and not args.input:
        from apertus_eagle.prepare_data import iter_parquet_rows

        tmp = args.output.with_name("renderer-probe-sample.jsonl")
        count = 0
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as handle:
            for row in iter_parquet_rows(args.parquet_dir):
                handle.write(json.dumps({"id": row["id"], "text": row["text"]}) + "\n")
                count += 1
                if count >= args.limit:
                    break
        report = probe_file(tokenizer, tmp, limit=args.limit)
        report["sample"] = str(tmp)
        mix_source = "parquet_sample"
    elif args.input:
        missing_ids = _collect_ids(args.input, limit=args.limit)
        if args.mix_text_json and args.mix_text_json.is_file():
            loaded = json.loads(args.mix_text_json.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                mix_text = {str(key): str(value) for key, value in loaded.items()}
                mix_source = "mix_text_json"
        parquet_dir = args.parquet_dir or DEFAULT_PARQUET
        if missing_ids and not mix_text and parquet_dir.is_dir():
            try:
                mix_text = lookup_mix_text(parquet_dir, missing_ids)
                mix_source = "parquet_lookup"
            except ImportError as exc:
                mix_text = {}
                mix_source = "reconstruct_mix_text_no_pyarrow"
                print(f"parquet lookup skipped ({exc}); using reconstructed mix text")
        report = probe_file(tokenizer, args.input, limit=args.limit, mix_text=mix_text)
        report["mix_text_lookup"] = {
            "requested": len(missing_ids),
            "found": len(mix_text),
            "source": mix_source,
            "parquet_dir": str(parquet_dir) if missing_ids else None,
            "mix_text_json": str(args.mix_text_json) if args.mix_text_json else None,
        }
    else:
        raise SystemExit("pass --input or --parquet-dir")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "examples"}, indent=2))
    raise SystemExit(0 if report.get("pass") else 1)


if __name__ == "__main__":
    main()

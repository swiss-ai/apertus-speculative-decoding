#!/usr/bin/env python3
"""Pre-registered epoch-1 decision. This never launches another epoch."""
import argparse
import json
import re
import statistics
from pathlib import Path


def parse_curve(text):
    rows = []
    for match in re.finditer(
            r"train/loss=([^\s,]+).*?train/eal=([^\s,]+).*?global_step=(\d+)",
            text or "", re.S):
        try:
            loss, eal, step = float(match.group(1)), float(match.group(2)), int(match.group(3))
        except ValueError:
            continue
        if loss == loss and eal == eal:
            rows.append((step, loss, eal))
    return rows


def _window(rows, center, width=50):
    chosen = [eal for step, _loss, eal in rows if abs(step - center) <= width // 2]
    if not chosen:
        return None
    return statistics.mean(chosen)


def doubling_gain(rows):
    """Gain in analytical accepted length from step 2038 to step 4077."""
    early, late = _window(rows, 2038), _window(rows, 4077)
    if early is None or late is None or early == 0:
        return None
    return (late - early) / abs(early)


def _flat(series):
    if len(series) < 2:
        return True
    return max(series) - min(series) <= 0.02 * max(abs(series[0]), 1e-6)


def decide(gain, domains, opb_acceptance):
    """Apply the pre-registered rules. ``launch_epoch_2`` is always false."""
    gaps = []
    if opb_acceptance:
        for name, series in sorted((domains or {}).items()):
            if not series:
                continue
            if series[-1] < 0.75 * opb_acceptance and _flat(series):
                gaps.append(name)
    token_limited = gain is not None and gain >= 0.03
    flat_curve = gain is not None and gain < 0.03
    if gain is None and not domains:
        action = "incomplete"
        reason = "The epoch curve and coverage probes are not both available."
    elif flat_curve and domains is not None and not gaps:
        action = "stop"
        reason = "In-distribution accepted length is flat and no domain is below 75% of open-perfectblend."
    elif gaps or token_limited:
        action = "review-more-data"
        reason = (
            "A coverage gap or a last-doubling gain of at least 3% would justify more unique tokens. "
            "Epoch 2 on open-perfectblend plus the expansion, with a short learning-rate re-warmup, "
            "would be a separate reviewed campaign."
        )
    else:
        action = "stop"
        reason = "The measured curve does not clear the pre-registered bar for more training."
    return {
        "action": action,
        "launch_epoch_2": False,
        "doubling_gain": gain,
        "coverage_gaps": gaps,
        "token_limited": bool(token_limited),
        "reason": reason,
    }


def write_report(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Epoch-1 data decision",
        "",
        payload["reason"],
        "",
        "- Action: `%s`" % payload["action"],
        "- Launch epoch 2 automatically: no",
        "- Last-doubling accepted-length gain: %s" % payload["doubling_gain"],
        "- Coverage gaps: %s" % (", ".join(payload["coverage_gaps"]) or "none recorded"),
        "",
        "Rules were fixed before the measurements: a domain below 75% of the open-perfectblend",
        "acceptance and flat across milestones needs that domain's data; a gain of at least 3%",
        "over the last doubling means more tokens are worthwhile; a flat curve with a small",
        "coverage gap means stop. Nothing in this report changes the learning rate, batch,",
        "data, or optimizer of the current campaign.",
        "",
    ]
    path.write_text("\n".join(lines))
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = parse_curve(args.log.read_text(errors="replace")) if args.log else []
    domains, opb = {}, None
    if args.probes and args.probes.is_file():
        payload = json.loads(args.probes.read_text())
        domains = payload.get("domains") or {}
        opb = payload.get("opb_acceptance")
    report = decide(doubling_gain(rows), domains, opb)
    report["curve_rows"] = len(rows)
    write_report(args.output, report)
    print("EPOCH_DECISION " + json.dumps({k: report[k] for k in ("action", "launch_epoch_2", "doubling_gain")}))


if __name__ == "__main__":
    main()

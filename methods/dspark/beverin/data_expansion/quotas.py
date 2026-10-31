"""Quotas for the expansion corpus. Pure Python, no dataset access."""

TOOL = (
    "Toucan-1.5M",
    "glaiveai/glaive-function-calling-v2",
    "EnvScaler-SFT-Traj-9K",
    "OpenSeeker-v1-Data",
    "jupyter-agent/jupyter-agent-dataset",
)
MULTILINGUAL_ALL = (
    "romansh-sft",
    "jvamvas/apertus-pretrain-romansh-backtranslated-sft",
    "swiss-german-dialects",
    "africa-sft",
)
MULTILINGUAL_FILL = (
    "euroblocks-25-12-multilingual",
    "EuroBlocks-SFT-Synthetic-1124",
    "wikipedia",
)
QUOTAS = {
    "multilingual": 300000,
    "chat": 200000,
    "tool": None,
    "instruction": None,
    "medical": 60000,
    "science": 60000,
    "safety": None,
    "math": 50000,
    "code": 50000,
    "summarization": 30000,
}
PRIORITY = (
    "tool", "multilingual", "instruction", "chat", "summarization",
    "medical", "science", "safety", "math", "code",
)
PROBE_PER_DOMAIN = 1000
PROMPT_TOKEN_LIMIT = 4096
SUMMARY_TOKEN_LIMIT = 3500


def classify(source):
    name = source or ""
    lower = name.lower()
    if (name in TOOL or "toucan" in lower or "glaive" in lower or "envscaler" in lower
            or "openseeker" in lower or "jupyter-agent" in lower):
        return "tool"
    if (name in MULTILINGUAL_ALL or name in MULTILINGUAL_FILL or "romansh" in lower
            or "swiss-german" in lower or name == "africa-sft" or "euroblock" in lower
            or name == "wikipedia"):
        return "multilingual"
    if (name.startswith("dolci-if") or name.startswith("if-") or "multiturnif" in lower):
        return "instruction"
    if "smoltalk" in lower:
        return "chat"
    if "sciriff" in lower:
        return "science"
    if "medmcqa" in lower or "medical" in lower or "miriad" in lower:
        return "medical"
    if "safety" in lower or "harmless" in lower or "constitutional" in lower:
        return "safety"
    if "code" in lower:
        return "code"
    if "math" in lower or name in ("gsm8k", "HARP"):
        return "math"
    if "finepdf" in lower:
        return "summarization"
    return "other"


def _fill(sources, quota):
    chosen = {}
    remain = quota
    for name in sorted(sources):
        if remain <= 0:
            break
        take = min(int(sources[name]), remain)
        if take:
            chosen[name] = take
            remain -= take
    return chosen


def _fill_multilingual(sources, quota):
    chosen = {}
    for name, count in sources.items():
        if name in MULTILINGUAL_ALL or "romansh" in name.lower() or "swiss-german" in name.lower() or name == "africa-sft":
            chosen[name] = int(count)
    remain = max(0, quota - sum(chosen.values()))
    for name in MULTILINGUAL_FILL:
        if remain <= 0 or name not in sources:
            continue
        take = min(int(sources[name]), remain)
        if take:
            chosen[name] = take
            remain -= take
    return chosen


def allocate(counts):
    grouped = {}
    for source, count in counts.items():
        grouped.setdefault(classify(source), {})[source] = int(count)
    plan = {}
    for domain, sources in grouped.items():
        quota = QUOTAS.get(domain)
        if domain == "multilingual":
            plan[domain] = _fill_multilingual(sources, quota)
        elif quota is None:
            plan[domain] = dict(sources)
        else:
            plan[domain] = _fill(sources, quota)
    return plan


def priority_names():
    return list(PRIORITY)

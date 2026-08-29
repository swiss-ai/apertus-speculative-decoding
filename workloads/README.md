# Workloads

`smoke.jsonl` only checks the harness and greedy-output equivalence. It is too small and its
"long" examples are too short for research conclusions.

The canonical corpus uses one JSON object per line:

```json
{"id":"unique-id","workload":"code","messages":[{"role":"user","content":"..."}],"max_tokens":512}
```

The measured corpus should contain disjoint, versioned strata for open chat, code generation,
and long-context summarization. Record source name, source revision, license, selection code,
Apertus-tokenizer input-token counts, and the corpus SHA-256. Do not commit restricted source
text; commit a manifest and deterministic preparation code instead.

Recommended shape targets are:

| Workload | Input tokens | Output cap | Main serving phase |
|---|---:|---:|---|
| open chat | 128–2,048 | 256 | mixed |
| code | 128–4,096 | 512 | decode-heavy |
| long-context summarization | 16,384–65,536 | 256–512 | prefill + decode |

Use natural EOS for the primary production-like comparison. Add a separate fixed-output control
with `ignore_eos=true`; do not silently mix the two designs.

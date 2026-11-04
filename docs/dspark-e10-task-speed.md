# DSpark e10 k=3: per-task speedups, in distribution and out of it

Status 2026-11-04: harness built and tested; controlled pipeline smoke submitted.
No measurement yet. Branch `experiment/dspark-e10-task-speed`.

The epoch-10 DSpark head (thinking off, k=3) serves always-on on Kubernetes as
`SwissAI-Research/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3`. This campaign
measures what that deployment delivers per task, separately for tasks inside
and outside the drafter's training distribution.

What existed:
- The 2026-10-30 controlled test covered only 3 workloads at a 256-token cap
  (`results/8b/speed/id-ood-20261030-c`). Results at C=1/8/128:
  - ID OPB: 2.19x / 2.08x / 1.14x.
  - OOD chat: 1.74x / 1.50x / 0.92x.
  - OOD summarization: 1.43x / 1.30x / 1.00x.
- The workload-coverage branch measured acceptance on 21 slices but had no
  plain arm, so it has no speedups.
- The live pod has only a 10-request smoke test (mean acceptance length 2.41).

## Design

**Suite** (`experiments/task-speed/suite.yaml`, frozen before measuring).
- 28 slices from existing workloads, 32 prompts each, selected by sha256 order
  of `task-speed-20261104:<id>`.
- Each slice keeps its source's output cap and stops at natural EOS. Requests
  are greedy, with thinking off.
- Each slice is classed by its share of the drafter's training corpus
  (`results/8b/coverage/corpus-share.json`):

| Class | Slices | Why |
|---|---|---|
| ID | opb (held-out OPB), math-gsm8k, probe-math, probe-humaneval, multi-turn | The corpus's own prompts and dominant types: math 43.8%, multi-turn 25.0%, Python in 13.3% code |
| Near-ID | code-multi, json-schema, tool-call, lang-eng, arena-hard-clean, arena-hard-contaminated | Formats present at a low share; chat benchmark partly in the corpus sources |
| OOD task | chat, summarization (eng/deu/fra/ita documents), long-context (8.5k–23.6k tokens), translate-to-en, translate-from-en | Task types absent or rare: no long-document summarization, no prompts of 8k+ tokens, 0.66% translation |
| OOD language | deu, fra, ita, spa, roh, zho, jpn, hin, ara, swa, yor, amh | 0–0.054% of rows each; English is 99.55% |

**Load grid.** C=1, 8, 32 and 128, closed loop.
- Requests per cell are whole passes over the slice: 32, 32, 64 and 256 (C=128 runs 8 passes).
- Every cell has a warmup of C requests.
- Each deployment first gets a pre-pass of one prompt per slice at C=1 and at C=32 (C=128 live). This compiles the drafter's Triton kernels before measurement; on 2026-10-30, a JIT compile inside a measured cell cost one request 8.5 s.

**Speedup** is DSpark output tokens/s divided by plain output tokens/s on the same cell.
- It is the geometric mean over deployment pairs, with the exact bootstrap over pairs.
- Verdicts are as in the 2026-10-30 campaign:
  - speedup: interval above 1.05;
  - slowdown: interval below 0.95;
  - within ±5%: interval inside 0.95–1.05;
  - otherwise inconclusive.
- Class results are geometric means over their slices.
- A linear fit of speedup against acceptance length per C gives the break-even acceptance at each load.

### Track A: controlled (Clariden)

`experiments/task-speed/submit-controlled.sh` runs `run-controlled.py` on one 4-GPU node.
- It uses the pilot container that the live Deployment was frozen from: the same image, the 4 source overlays, the forced-thinking-off template and the same checkpoints.
- Engine flags are the live ones: TP1, bf16, 32k context, 0.8 memory, 16,384 batched tokens, 256 sequences, prefix caching off.
- Each GPU runs one pair: a fresh plain deployment and a fresh DSpark deployment, in AB order on odd pairs and BA on even pairs. The 4 pairs run concurrently.
- After timing, the runtime is re-hashed and an artifact manifest is written.
- Cells left inconclusive get extension pairs on the same corpus (`TASK_SPEED_SELECT` from the report's `select.json`).

### Track B: deployed (Kubernetes pod, through the gateway)

The plain baseline for the live pod is the same Deployment toggled to plain mode (`serving/kubernetes/prepare.py --variant plain`).
- vLLM fixes speculation at engine start, and a k=0 path in the DSpark engine would keep its about 27% smaller KV cache.
- The plain render differs only in having no speculative config, its own model ID (`...-8B-plain-matched`) and a variant annotation. A test enforces this.
- Both modes pin node `nid006597`.
- Windows are D1, P1 on day 1 and P2, D2 on day 2, each via a Git toggle (`experiments/task-speed/toggle.md`).

`run-live.py` sends the same corpus and grid through the authenticated gateway.
- It reads counters inside the pod with `kubectl exec`.
- It rejects and reruns any cell with other users' traffic: the engine must be idle at the start, and its finished-request counter must advance by exactly the requests sent.
- A `--direct` port-forward pass on 3 slices measures the gateway overhead.

The client and kubectl run together on Clariden debug nodes. The prompt corpus includes restricted text, so it stays on the cluster. Windows resume across jobs.

## Differences from the approved plan

- **Prompts:** 32 per slice instead of 64, so each window is about 80 min instead of over 2 h. Load cells still cover whole passes and 256 requests at C=128.
- **Slice count:** 28, not 29. The probe split into math and HumanEval was double-counted.
- **Track A pairs:** 4 concurrent pairs, one per GPU, instead of 3 sequential ones. Pairs on different GPUs also capture GPU-to-GPU spread.
- **Correctness:** dropped from Track A. Load-test cells store only output hashes, not text, so there is nothing to compare greedy outputs on. If it is wanted, `apertus-bench capture` and `correctness-gate` can run on the same corpus as a separate step.
- **Drafter hash check in plain mode:** kept, not skipped. Plain mode verifies the same assets.

## Results

Pending.

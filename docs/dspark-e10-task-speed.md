# DSpark e10 k=3: per-task speedups, in distribution and out of it

Status 2026-11-05: complete. Live (deployed) campaign: 4 windows, 224 valid paired
cells. Controlled Clariden run: job 3850624, 4 pairs, 112 cells, none rejected;
extension job 3867076 for the 2 inconclusive cells (done). Branch `experiment/dspark-e10-task-speed`.

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

## Results: the deployed pod (live, through the gateway)

Report: `results/8b/task-speed/report-live-20261105/REPORT.md` (tables per slice,
per-request decode speedup, latency, acceptance fit); raw tree on Clariden
(`live-20261104-raw.tar.gz`, sha256 in the run directory, file hashes in
`artifact-manifest.json`). Windows D1 (DSpark), P1 (plain), P2 (plain, fresh pod),
D2 (DSpark, fresh pod) on GH200 nid006597, 2026-11-04/05; two deployment pairs per
cell (screening level, so intervals are narrow but rest on two pairs).

Output-token throughput speedup, geometric mean over slices (range over slices):

| Class | C=1 | C=8 | C=32 | C=128 | Acceptance length |
|---|---|---|---|---|---|
| ID | 2.59x (2.20–2.81) | 2.45x (2.14–2.65) | 2.12x (1.80–2.32) | 1.40x (1.24–1.46) | 3.50 |
| Near-ID | 2.18x (1.94–2.48) | 1.84x (1.23–2.27) | 1.61x (1.02–1.99) | 1.20x (1.02–1.34) | 3.12 |
| OOD task | 1.57x (1.30–2.22) | 1.44x (1.26–2.10) | 1.07x (0.74–1.25) | 1.02x (0.87–1.19) | 2.25 |
| OOD language | 1.32x (1.08–1.61) | 1.28x (0.94–1.64) | 1.22x (0.98–1.74) | 0.87x (0.70–1.39) | 1.76 |

- **In distribution, DSpark always wins.** Math, code, OPB and multi-turn run
  2.2–2.8x faster at C=1 and still 1.24–1.46x at C=128. Time per output token drops
  from 5.8 ms to 2.0 ms; time to first token is unchanged (about 30 ms).
- **Acceptance length predicts the speedup.** At C=1, speedup ≈ 0.12 + 0.68·L
  (R² 0.96). Break-even is L ≈ 1.3 up to C=32 but L ≈ 2.1 at C=128, so under heavy
  load only tasks that accept more than about 2 tokens per step gain.
- **Out of distribution, the gain shrinks and turns into a loss under load.** Chat
  1.55x at C=1 but 0.92x at C=128; translation from English 0.87x; summarization
  flat (1.03x); long context keeps 1.14x. Eleven of 28 slices are significantly
  slower at C=128: chat, translate-from-en and 9 of 12 non-English languages
  (0.70–0.89x). Arabic and Swahili (L ≈ 1.45) barely gain even at C=1 (1.08x, 1.13x).
- **Live acceptance equals the offline measurements** (GSM8K 3.75 vs 3.74,
  JSON 3.73 vs 3.73, German 1.61 vs 1.61), so the serving path loses nothing.

Caveats:
- Cells marked `?` (json-schema, tool-call and translate-to-en at C=32,
  lang-eng/swa/yor at C=8) are short (a few seconds) and dominated by single
  gateway stream stalls; the per-request decode speedup (TPOT ratio) in the
  report is unaffected and shows the expected gain there.
- The gateway dropped 39 of 256 streams in one long-context C=128 attempt and a few
  elsewhere (`incomplete chunked read`); such cells were remeasured.
- Someone else sent requests to the DSpark ID during D1 (every 30–60 s); cells with
  other traffic were rerun, and from the last D1 job on, cells with at most 2% foreign
  tokens were accepted (recorded per cell in `guard.json`).

## Results: controlled (Clariden, same GPU) and transfer to the deployment

Report: `results/8b/task-speed/report-20261105/REPORT.md` (controlled and live
side by side, plus live ÷ controlled per cell). Run
`controlled-20261104T133618Z` (job 3850624, node nid006248, 4 GPUs, 4 concurrent
AB/BA pairs, 84 min); same corpus (sha256 7d8e60ef…), runtime re-hash passed.

| Class | C=1 | C=8 | C=32 | C=128 | Acceptance length |
|---|---|---|---|---|---|
| ID | 2.61x (2.20–2.86) | 2.57x (2.19–2.83) | 2.15x (1.81–2.41) | 1.38x (1.23–1.45) | 3.50 |
| Near-ID | 2.30x (2.00–2.63) | 2.16x (1.79–2.58) | 1.82x (1.47–2.12) | 1.25x (1.13–1.40) | 3.12 |
| OOD task | 1.58x (1.31–2.24) | 1.44x (1.25–2.14) | 1.25x (1.07–1.76) | 0.99x (0.82–1.13) | 2.25 |
| OOD language | 1.32x (1.09–1.63) | 1.36x (1.09–1.87) | 1.24x (1.00–1.88) | 0.85x (0.67–1.36) | 1.76 |

- **The deployment delivers the controlled speedup.** Live ÷ controlled is
  0.93–1.07 in nearly every cell. The exceptions are the short live cells
  already flagged `?` above (json-schema C=8/32, lang-eng and lang-yor C=8,
  translate-to-en C=32), where single gateway stream stalls dominate; their TPOT
  ratios match the controlled ones. The gateway adds about 10 ms TTFT and no
  decode cost.
- **The same picture, with tighter intervals.** At C=128, 12 of 28 slices are
  significantly slower in the controlled run: chat (0.91x), translate-from-en
  (0.82x) and 10 of 12 non-English languages (0.67–0.92x); only Yoruba (1.36x)
  and Romansh stay at or above break-even. The C=1 fit is
  speedup ≈ 0.04 + 0.72·L (R² 0.98); break-even L ≈ 1.3 up to C=32 and 2.2 at C=128.
- 110 of 112 controlled cells reach a verdict with 4 pairs. lang-roh C=128 and
  lang-ara C=32 got 4 extension pairs (`controlled-ext-20261105`, job 3867076,
  8 arms ok). With 8 pairs they sit on the band edge and stay `?`:
  lang-roh C=128 0.95x [0.93, 0.97] (about 5% slower), lang-ara C=32 1.08x
  [1.05, 1.11] (about 8% faster). No further extension; neither changes a
  class result or conclusion.

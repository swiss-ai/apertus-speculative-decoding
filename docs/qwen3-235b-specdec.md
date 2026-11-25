# Qwen3-235B-A22B: speculative decoding on one GH200 node

Action item from the 20 November huddle (Faruk, Yu, Rob). The CSCS guest at the
6 November huddle (David Isztl) reported that speculative decoding did not pay off
in their serving: acceptance was below 50% after the first draft token and the drafter
added memory overhead, so they disabled it. We first assumed the report was about
GLM-5.3-Flash (`docs/glm53-flash-specdec.md` on branch
`experiment/glm53-flash-specdec`). On 20 November Yu confirmed it concerns
**Qwen3-235B-A22B**. This note measures plain against EAGLE3 speculative decoding
for that model.

**This is not a reproduction of the CSCS deployment.** Their serving configuration
for Qwen3 was not shared. The Slack search covered #swiss-ai-spec-decode, the DM
with David Isztl and #swiss-ai-serving. It turned up only the 6 November huddle
notes and a P/D disaggregation PDF, which uses vLLM 0.31. The #swiss-ai-serving
launches of `Qwen/Qwen3-235B-A22B-Instruct-2507` that it found use two nodes in
BF16 with no speculative configuration. This note therefore uses the GLM baseline
conventions instead (one node, TP4, the same image and the same bench settings).

## Setup

- **Target:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` at revision
  `e156cb4efae43fbee1a1ab073f946a1377e6b969` (Apache-2.0; official block-FP8; 221 GiB).
  The infra01 model store holds only the BF16 checkpoint (438 GB), which does not fit
  on one 4 x 96 GB node. The FP8 checkpoint was downloaded with Slurm job 4004120 to
  `/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/models/`. All 25 LFS
  files matched the Hub's SHA-256 values, and the full manifest is in each run
  directory (`target.sha256`).
- **Drafter:** `RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3` at revision
  `87cd7aa934131942f5d7277617d7d6020f2df7f1` (Apache-2.0). It is a one-layer EAGLE3
  head in speculators format, 2.4 GB in float32, with a 64k draft vocabulary.
  `model.safetensors` has SHA-256
  `d591c7a357a66ad97225aec0e350ef79ccf10f97a41aefdf0f43d0228bcc984a`. Its model card recommends k=3 with
  vLLM. It was trained against the BF16 target on Magpie and UltraChat, so drafting
  for the FP8 target is a small distribution shift.
- **Runtime:** the same CI image as the GLM sweep,
  `vllm_nvidia_alps-arm64.sqsh` (vLLM 0.30.0, torch 2.13.0+cu130), with SHA-256
  `d259b3fb40615d406febc094ae8509bc1aa588f97c4d738677d988ac997d7a90`.
- **Layout:** one node with TP4 for attention and experts and no expert parallelism,
  matching the GLM sweep. The other engine settings are `--max-model-len 16384`,
  `--max-num-seqs 128`, `--max-num-batched-tokens 16384`,
  `--gpu-memory-utilization 0.90`, prefix caching off and
  `--compilation-config.pass_config.fuse_allreduce_rms false`, the usual TP>1
  workaround on GH200. Each arm's `server.log` records its KV-cache capacity.

## Arms and workloads

| Arm | Speculation |
| --- | --- |
| `plain` | none |
| `eagle3-k3` | EAGLE3, 3 draft tokens (the card's setting) |
| `eagle3-k5` | EAGLE3, 5 draft tokens |

The workloads, concurrency levels and decoding settings are identical to the GLM
sweep:

- `probe`: 64 math and code prompts from RedHatAI/speculator_benchmarks, with an
  output cap of 384 tokens.
- `arena-hard-clean`: 371 Arena-Hard prompts, with an output cap of 1024 tokens.
- Concurrency C = 1, 8, 32 and 128. Each level measures at least max(32, 4 x C)
  requests, rounded up to whole passes over the prompts.
- Decoding is greedy and uses the model's default chat template (Instruct-2507 has no
  thinking mode).

## Decision rule

A speculative arm counts as worth using at a given C when its output throughput is at
least **10% higher** than `plain` at that C on both workloads, with no failed
requests. Losses are reported as plainly as gains.

## Procedure and provenance

1. Smoke test (`QWEN3_SMOKE=1`, debug partition): load the model, then run 8 short
   probe requests at C=1 and C=8.
2. Full arms: one job per arm on the normal partition, run one after another on one
   node at a time (`serving/qwen3/submit-arm.sh ARM`). Each arm gets a fresh
   deployment and run directory.

Each run directory under `results/qwen3-235b/` keeps the following provenance:

- the git revision and checkout diff;
- the exact `vllm serve` command and the environment file;
- the image digest;
- the pinned target and drafter revisions with full-file SHA-256 manifests;
- the workload digests;
- the runtime versions and `pip freeze`;
- the Slurm job and node;
- the server and bench logs and the GPU memory samples;
- the bench client's cells (`metadata.json`, `requests.jsonl`, `summary.json` and
  the metric snapshots).

## Result

Each arm ran as one deployment and one job on 22 November 2026, under account
d_research_serving:

| Arm | Job | Node | Run directory |
| --- | --- | --- | --- |
| smoke `eagle3-k3` | 4004146 | nid005396 | `results/qwen3-235b/smoke-20261122T171650Z-eagle3-k3` |
| `plain` | 4004250 | nid006230 | `results/qwen3-235b/sweep-20261122T173201Z-plain` |
| `eagle3-k3` | 4004251 | nid006230 | `results/qwen3-235b/sweep-20261122T173203Z-eagle3-k3` |
| `eagle3-k5` | 4004252 | nid006988 | `results/qwen3-235b/sweep-20261122T173206Z-eagle3-k5` |

All 7,869 measured requests succeeded (24 cells) and the logs show no preemptions. KV-cache use
stayed at or below 18%, so admission did not limit any arm.

The drafter cost KV capacity:

| Arm | KV cache (tokens) | vs plain |
| --- | --- | --- |
| `plain` | 587,792 | — |
| `eagle3-k3` | 537,824 | -8.5% |
| `eagle3-k5` | 537,072 | -8.6% |

The tables below give output throughput against plain, the mean acceptance length
(AL, tokens per target step) and the median TPOT. They are reproduced by
`python experiments/qwen3-235b/compare.py <plain run> <k3 run> <k5 run>`.

#### probe

| C | plain tok/s | plain TPOT p50 ms | eagle3-k3 tok/s (vs plain) | eagle3-k3 AL | eagle3-k3 TPOT p50 ms | eagle3-k5 tok/s (vs plain) | eagle3-k5 AL | eagle3-k5 TPOT p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 114 | 8.7 | 259 (+127.0%) | 3.16 | 3.7 | 288 (+152.7%) | 3.94 | 3.3 |
| 8 | 537 | 13.9 | 1152 (+114.6%) | 3.16 | 6.3 | 1258 (+134.3%) | 3.90 | 5.7 |
| 32 | 1349 | 21.2 | 2882 (+113.6%) | 3.15 | 9.6 | 3200 (+137.1%) | 3.90 | 8.4 |
| 128 | 3091 | 37.4 | 3416 (+10.5%) | 3.15 | 36.7 | 2733 (-11.6%) | 3.92 | 43.8 |

#### arena-hard-clean

| C | plain tok/s | plain TPOT p50 ms | eagle3-k3 tok/s (vs plain) | eagle3-k3 AL | eagle3-k3 TPOT p50 ms | eagle3-k5 tok/s (vs plain) | eagle3-k5 AL | eagle3-k5 TPOT p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 115 | 8.7 | 205 (+78.9%) | 2.40 | 4.9 | 205 (+79.0%) | 2.66 | 4.9 |
| 8 | 546 | 14.5 | 925 (+69.3%) | 2.40 | 8.6 | 917 (+67.9%) | 2.66 | 8.6 |
| 32 | 1355 | 22.9 | 2322 (+71.4%) | 2.40 | 13.1 | 2344 (+73.0%) | 2.67 | 13.1 |
| 128 | 3482 | 34.5 | 4177 (+20.0%) | 2.41 | 29.5 | 1621 (-53.4%) | 2.66 | 78.0 |

#### Verdict (>= +10% output throughput on both workloads, no failures)

- eagle3-k3 C=1: PASS (probe +127.0%, arena-hard-clean +78.9%)
- eagle3-k3 C=8: PASS (probe +114.6%, arena-hard-clean +69.3%)
- eagle3-k3 C=32: PASS (probe +113.6%, arena-hard-clean +71.4%)
- eagle3-k3 C=128: PASS (probe +10.5%, arena-hard-clean +20.0%)
- eagle3-k5 C=1: PASS (probe +152.7%, arena-hard-clean +79.0%)
- eagle3-k5 C=8: PASS (probe +134.3%, arena-hard-clean +67.9%)
- eagle3-k5 C=32: PASS (probe +137.1%, arena-hard-clean +73.0%)
- eagle3-k5 C=128: fail (probe -11.6%, arena-hard-clean -53.4%)

### Reading

- **EAGLE3 k3 meets the decision rule at every concurrency**, on both workloads,
  with no failures:
  - at C=1–32, +114–127% on probe and +69–79% on arena-hard-clean;
  - at C=128, +10.5% on probe and +20% on arena-hard-clean. The probe gain only just
    clears the 10% threshold.
- **EAGLE3 k5 helps up to C=32 and loses heavily at C=128.**
  - Up to C=32 it gains +134–153% on probe, about 20 points more than k3, and the
    same as k3 on arena-hard-clean (about +70–80%). On arena-hard-clean the 4th and
    5th draft positions are rarely accepted.
  - At C=128 it is -12% on probe and **-53% on arena-hard-clean**, with TPOT p50 at
    78 ms against 35 ms for plain.
  - A likely cause, not yet verified: at C=128 a k5 decode step carries 128 x 6 =
    768 tokens, which exceeds this build's 512-token CUDA-graph capture limit
    (`max_cudagraph_capture_size` 512). A k3 step carries 128 x 4 = 512 tokens and
    still fits.
  - k5 also ran on a different node from plain and k3.
- **Acceptance per draft position** on arena-hard-clean with k3 is 0.66, 0.44 and
  0.30 (k5 continues 0.19 and 0.12). This matches the CSCS observation that
  acceptance falls below 50% after the first token. Even so, it gives an AL of 2.4 and
  a 1.7–1.8x throughput gain up to C=32 on this layout. On probe (math and code) the
  per-position rates are 0.85, 0.71 and 0.58 (AL 3.15).

### Verdict

On one GH200 node with the FP8 target at TP4, **EAGLE3 with k=3 is worth enabling for
Qwen3-235B-A22B at every concurrency we measured (C ≤ 128)**. The gain is largest at
C ≤ 32 and narrow at C=128 on the probe set. k=5 is better only for single-stream
math and code, and it must not run at high concurrency without raising the CUDA-graph
capture size or a k schedule tied to load. The low acceptance CSCS reported after the
first token does not, on its own, explain a lack of speedup: we see the same
per-position profile and still a large gain.

Caveats:

- This is not the CSCS deployment. Their Qwen3 configuration is unknown, the public
  launches use BF16 across two nodes, and the P/D work used vLLM 0.31.
- Each arm is a single deployment, with no repeats.
- Decoding is greedy and outputs are capped at 1024 tokens, so we did not test
  sampled decoding (T>0), where acceptance usually drops.
- The drafter was trained against the BF16 target and is served here against FP8.
- Concurrency stops at 128, and KV use stayed low, so long-context or memory-bound
  regimes are not covered.

Next steps toward a reproduction:

- obtain CSCS's engine flags, the sampling temperature, the k value and their
  concurrency;
- rerun k3 at T>0 and at C=256;
- test whether raising `max_cudagraph_capture_size` recovers k5 at C=128.

Compute: under 0.05 node-hours for the download (job 4004120, 2.6 min), 0.25 node-hour for the
smoke test and about 3.3 node-hours for the three arms (83 + 54 + 59 min). Model
loading took about 13 minutes per arm.

## Context: the earlier GLM-5.3-Flash arms (not rerun)

These are from branch `experiment/glm53-flash-specdec`, run on 20 November. Their
raw runs are in the cluster checkout under
`/iopsstor/scratch/cscs/faruk_zahiragic/glm53-specdec/repo/results/glm53-flash/`.

| Arm | Job | Status |
| --- | --- | --- |
| `plain` | 3988223 | probe complete; arena-hard-clean C=1/8/32 complete; the engine died at C=128 (`EngineDeadError`), so that cell is missing |
| `mtp-k1` | 3988226 | complete |
| `mtp-k3` | 3988227 | complete |
| `dflash-k3`, `dflash-k7` (vLLM) | smoke 3988221 | never served: this vLLM build's Glm5Next lacks the EAGLE3 aux-hidden-state interface DFlash needs; full arms cancelled |
| `sgl-dflash` (SGLang) | smoke 3988417 | the SGLang engine failed during startup; not served |
| expert-parallel layout | smoke 3987975 | rejected: the drafter inherits `enable_expert_parallel` |

The native MTP head gave large gains at low concurrency, measured as output
throughput against plain with `experiments/qwen3-235b/compare.py`:

- `mtp-k1`: +44%/+48%/+44% on probe and +41%/+34%/+35% on arena-hard-clean at
  C=1/8/32.
- `mtp-k3`: +62%/+69% on probe and +43%/+40% on arena-hard-clean at C=1/8.

At high concurrency it lost:

- C=32: `mtp-k3` was +6% on probe and -10% on arena-hard-clean.
- C=128 probe: `mtp-k1` was -10% and `mtp-k3` -27%.

There is no arena-hard-clean comparison at C=128 because the plain cell is missing.

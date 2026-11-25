# GLM-5.3-Flash: serving baseline and speculative decoding on one GH200 node

Action item from the 6 November huddle (Yu + Faruk): reproduce GLM-5.3 Flash
serving and test speculative decoding. CSCS's own serving configuration was not
shared, so this is a self-contained baseline. It is not a reproduction of the
CSCS Inference Service deployment.

## Setup

- **Target:** `zai-org/GLM-5.3-Flash` from the infra01 model store
  (`Glm5NextForConditionalGeneration`; 45 layers, 288 routed experts, 8 active,
  sparse-indexed MLA with KDA; FP8 block-quantised weights, about 332 GB of shards;
  one native MTP layer). Text-only serving (`--language-model-only`).
- **Drafter:** `incoai/GLM-5.3-Flash-DFlash2` (5-layer DFlash 2 block-diffusion
  drafter, block 8, 2.3 GB, CC BY-NC-ND 4.0, research use).
- **Runtime:** `vllm_nvidia_alps-arm64.sqsh` CI image, vLLM 0.30.0+gced6857af. It
  registers the target, the `glm5_next_mtp` speculative type and
  `DFlash2DraftModel`. The drafter card documents SGLang; vLLM is used here to match
  the rest of this repository's measurements.
- **Layout:** one node, TP4 for attention and experts alike,
  `--gpu-memory-utilization 0.94`, `--max-model-len 16384`, `--max-num-seqs 128`,
  prefix caching off. The FP8 weights take about 77 GiB of each GPU's 96 GB, so KV
  capacity is small; this is recorded per arm in `server.log`. Expert parallelism
  is not used: in this vLLM build the drafter inherits `enable_expert_parallel` and
  fails validation because it has no experts (smoke job 3987975), so all arms use
  the same EP-free layout. EP versus TP for the experts is a separate question.

## Arms and workloads

| Arm | Speculation |
| --- | --- |
| `plain` | none |
| `mtp-k1`, `mtp-k3` | native MTP layer, 1 or 3 draft tokens |
| `dflash-k3`, `dflash-k7` | DFlash2 drafter, 3 or 7 draft tokens (k7 = trained block) |

- `probe`: 64 math/code prompts (RedHatAI/speculator_benchmarks, output cap 384).
- `arena-hard-clean`: 371 Arena-Hard prompts (output cap 1024).
- Concurrency C = 1, 8, 32, 128 per node; at least 32 and 4 x C measured requests per
  level; greedy decoding, the model's default chat template (thinking as default).

Both workloads are model-independent prompt sets already used for Apertus. Neither
is GLM's training distribution, so acceptance here is not a claim about the drafter's
best case.

## Measurements and decision rule

Per cell: output throughput, acceptance length (vLLM spec-decode counters), TTFT and
TPOT p50/p95, failures. A speculative arm is worth using at a given C when it gives
at least **10% higher output throughput** than `plain` at that C on both workloads,
with no failed requests. Report losses (as for Apertus at high C) as plainly as gains.

## Procedure and provenance

1. Smoke per speculative method on the debug partition (`GLM53_SMOKE=1`): load, a few
   short requests.
2. Full sweep: one job per arm on the normal partition, a fresh deployment and run
   directory each (`serving/glm53/submit-arm.sh ARM`).

Each run directory under `results/glm53-flash/` keeps the git revision and checkout
diff, the exact `vllm serve` command, the environment file, the image digest, target
config/index digests and shard sizes, drafter digests, runtime versions and
`pip freeze`, the Slurm job and node, the server and bench logs, GPU memory samples
and the bench client's cells (`metadata.json`, `requests.jsonl`, `summary.json`,
metric snapshots).

## Run record

The November 20 Clariden campaign completed the plain smoke (3988405) and the
full MTP k1 and k3 sweeps (3988226 and 3988227). The full plain sweep stopped
after seven partial cells. The vLLM DFlash attempts (3987975 and 3988221) and
the SGLang DFlash smoke (3988417) failed. The missing full plain comparison
leaves the 10% decision rule unresolved.

Clariden retains the raw cells and deployment provenance in
`/iopsstor/scratch/cscs/faruk_zahiragic/glm53-specdec/repo/results/glm53-flash/`.
The two full MTP sweeps are `sweep-20261120T124839Z-mtp-k1` and
`sweep-20261120T124840Z-mtp-k3`. The plain records are
`smoke-20261120T131320Z-plain` and `sweep-20261120T124838Z-plain`. The failed
SGLang launch record is `smoke-20261120T131453Z-sgl-dflash`.

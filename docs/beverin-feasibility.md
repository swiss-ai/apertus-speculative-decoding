# DSpark 70B on Beverin (MI300A): feasibility check, 2026-10-24

Question: can the DSpark 70B training (online hidden-state extraction from
Apertus-v1.5-70B + speculators trainer, `methods/dspark/train/`) run on CSCS
Beverin's `mi300` partition instead of Clariden's saturated `normal` partition?

**Verdict: feasible with work.** On one MI300A node the 70B served
correctly in a public ROCm vLLM image. The extraction server produced hidden
states that pass `check_hidden_states.py` at Clariden's level (0.980 and 0.978
top-1 agreement vs 0.983). The unchanged Clariden `train-node.sh` also trained
30 DSpark steps there (FSDP, Muon, flex attention on ROCm torch), and RCCL
runs over Slingshot between nodes. Three things block a production run:

1. **Memory.** On MI300A the four GPUs and the host share one ~501 GB pool per
   node. The one-node smoke used 483 GB while training. Its epoch-end
   checkpoint save, a full state dict gathered on rank 0's host, then
   stalled at 499 GB. The planned 16-node layout should fit (estimate below),
   but checkpoint saves and resumes there are unverified.
2. **Multi-node training is unverified.** Only the RCCL micro-benchmark ran
   across two nodes. Muon's sharded all-to-all at 32 ranks, which is what hung
   on Clariden, has not run on Beverin.
3. **Software differs from Clariden's.** The model runs through a small
   text-only plugin (`methods/dspark/beverin/apertus15_text/`) on stock vLLM,
   not through the Swiss AI fork. The image is AMD's or upstream's, not ours.
   Its Python and torch differ from Clariden's, which has py3.12, torch 2.11
   and vLLM 0.23.1rc1 (fork).

Estimated effort for the full port: **2–4 working days.** That covers a 2-node
and then a 16-node pilot with checkpoint save and resume, memory tuning, an
image pinned and recorded like the Clariden ones, and a decision on the
vLLM 0.23 or 0.31 image. Each 1-node job spends ~15–17 min loading the 70B
before useful work, and multi-node pilots will wait on that too.

## What was run (all `-A infra01 -p mi300`, one node unless stated)

| Job | What | Result |
|---|---|---|
| 693709 | bare compute-node probe (srun) | 4× MI300A gfx942, amdgpu driver 6.12.12, 501 GB RAM, `/tmp` and `/dev/shm` are RAM tmpfs (352 GB); Docker Hub, Hugging Face, GitHub, PyPI reachable |
| 693712 | `enroot import docker://rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0` | 655 s; 40.1 GB sqsh, sha256 `75f5553d…68df`; torch 2.11.0+rocm7.14.1 sees 4 GPUs × 128 GiB; vLLM 0.23.1.dev1+g9ddef7117 |
| 693716 | serve smoke | cancelled: `curl` in the image does not load (`libldap_r-2.4.so.2`), health loop never succeeded |
| 693719 | serve smoke TP2 + extraction | **serve OK** (below); extraction refused: old connector rejects chunked prefill |
| 693726 | extraction + check | 1st row OK, 2nd row truncated (86 of 134 positions): old connector skips prefix-cache hits |
| 693727 | training smoke | cancelled (same prefix-cache bug) |
| 693728 | 2-node RCCL | killed by SIGPIPE from my `head` (script bug) |
| 693729 | **2-node RCCL** | aws-ofi-nccl 1.20 + libfabric 2.6, provider cxi, 4 NICs; results below |
| 693748 | **extraction + check, vLLM 0.23 image** | **HS_CHECK_OK, top-1 0.980 (579/591)** |
| 693749 | training smoke | failed at start: device-count mismatch (`ROCR_VISIBLE_DEVICES` from Slurm) |
| 693751 | import `vllm/vllm-openai-rocm:v0.31.0` | 268 s; 32.5 GB sqsh, sha256 `b58797bb…816f`; vLLM 0.31.0+rocm723, torch 2.13.0, py3.12 |
| 693753 | extraction, vLLM 0.31 image | refused at load: index names tower files the plugin excluded (fixed in plugin) |
| 693756 | GPU-visibility test (srun) | `CUDA_VISIBLE_DEVICES=2,3` with Slurm's `ROCR_VISIBLE_DEVICES=0,1,2,3` → torch counts 4, creates 2 |
| 693757 | **training smoke, vLLM 0.23 image** | **30 steps trained**; epoch-end save stalled at 499/501 GB, cancelled |
| 693758 | **extraction + check, vLLM 0.31 image, Clariden settings** (chunked prefill and prefix caching on) | **HS_CHECK_OK, top-1 0.978 (578/591)** |
| 693772 | training smoke, smaller (SEQ 2048, 128 anchors, EXTRACT_MEM 0.62), run twice for save + resume | see "Save and resume" |

Logs and run directories: `/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/`
(`logs/`, `runs/extract-<job>/` with the launch_vllm.py provenance
`vllm_command.txt`, `vllm.patch`, `checkpoint_sha256.txt`, `hs-check.log`;
`runs/train-smoke-<job>/logs/`). The repository copy that ran is in `repo/`;
its `REVISION` file names the commit of this branch.

## Environment (step 1)

- **Container engine** works as on Clariden: `srun --environment=<edf.toml>`,
  enroot + squashfuse. Images are imported on a compute node with
  `enroot import docker://…` (`import-image.sbatch`); compute nodes reach
  Docker Hub, Hugging Face, GitHub and PyPI, unlike Clariden's (no GitHub).
- **GPUs in the container:** no device hook is needed. `ENROOT_RESTRICT_DEV`
  is unset, so `/dev/kfd` and `/dev/dri` pass through. ROCm 7.14 user space
  (rocm/vllm) and ROCm 7.2.3 (vllm-openai-rocm) both run on the host's amdgpu
  6.12.12 driver.
- **RCCL over Slingshot:** EDF annotation `com.hooks.aws_ofi_nccl.enabled =
  "true"`. Beverin's hook defaults to the netstack artifact
  `gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0`
  (`/etc/enroot/hooks.d/90-aws-ofi-nccl.sh`); `variant = "rocm6"` is only
  used with `netstack.source = "host"`. The hook injects the
  `FI_CXI_RDZV_*=0` rendezvous settings that Clariden's runs set by hand.
  It also sets GH200-flavoured `NCCL_*` variables (`NCCL_P2P_LEVEL=NVL`,
  `NCCL_NET_GDR_C2C=1`), which RCCL tolerated in the test.
- **CSCS's vLLM-on-AMD tutorial** uses `vllm/vllm-openai-rocm` (calls
  AMD's `rocm/vllm` deprecated), `HWLOC_COMPONENTS=-gl` against a hang at RCCL
  init, `FI_PROVIDER=cxi`, and unsets `ROCR_VISIBLE_DEVICES`. Both EDFs here
  now carry `HWLOC_COMPONENTS`. The runs above predate that except 693758
  and 693772.
- No ROCm PyTorch/vLLM uenv exists on Beverin (only prgenv-gnu, linalg, ICON,
  QE, Forge); the shared `infra01/container-images` has one ROCm image
  (`ci/sglang_0.5.10_rocm.sqsh`). Containers are the way.
- Docs used: <https://docs.cscs.ch/clusters/beverin/>,
  <https://docs.cscs.ch/software/container-engine/resource-hook/> (hook
  annotations; `rocm5`/`rocm6` aws-ofi variants),
  <https://docs.cscs.ch/tutorials/ml/vllm-sglang-amdgpus/> (images, EDF, env).

RCCL, 2 nodes × 4 GPUs (job 693729), bus bandwidth:

| Collective | 8 MiB | 64 MiB | 512 MiB |
|---|---|---|---|
| all-reduce | 40.5 GB/s | 104.8 GB/s | 87.5 GB/s |
| all-to-all | 25.3 GB/s | 39.3 GB/s | 40.5 GB/s |
| all-gather (output size 64 / 512 / 4,096 MiB) | — | 83.2 GB/s | 84.2 / 91.2 GB/s |

The all-to-all result was checked for correctness.

## Serving Apertus-v1.5-70B on ROCm (step 2)

Upstream vLLM has `ApertusForCausalLM` (xIELU, QK-norm) but not
`Apertus1p5ForConditionalGeneration`. The Swiss AI fork's `apertus_mm.py`
needs the fork's transformers (`Apertus1p5VisionTokenizerModel`). Extraction
feeds token ids only, so `methods/dspark/beverin/apertus15_text/` is a vLLM
general plugin, found through its entry-point dist-info on `PYTHONPATH`. It
registers:

- config `apertus1p5` → `ApertusConfig` with `text_config` flattened (the
  fork's `configs/apertus1p5.py`);
- architecture → upstream `ApertusForCausalLM` + the fork's pruned head
  (`output_vocab_size` 131,072 rows, logits padded with −inf to 266,752),
  weights `model.language_model.*` → `model.*`, tower weights dropped.

The decoder is vLLM's own `ApertusModel`, with the Python xIELU (the CUDA
xIELU kernel is absent, as it is on Clariden). launch_vllm.py and the trainer
read the target config with AutoConfig, so they run through
`apertus15_run.py`, which registers the config first.

Job 693719 (vLLM 0.23 image, TP2 on GPUs 0,1, `--gpu-memory-utilization 0.85`,
CUDA graphs on):

- weights: 684 s (11.4 min) from capstor; 66.85 GiB per GPU. Server up after
  920 s (Clariden: ~6 min load + ~4 min compile/graphs).
- KV cache 36.75 GiB/GPU = 240,848 tokens; compile + graphs 110 s.
- Greedy: "What is the capital of Switzerland?" → "The capital of Switzerland
  is Bern."; a correct, documented Fibonacci function; a coherent completion
  on Swiss history.
- Single-stream decode: 512 tokens in 13.5 s = **37.9 tok/s** (TP2).

TP4 was not tried.

## Hidden-state extraction (step 3)

`extract-check.sh` runs the Clariden per-node server:
`launch_vllm.py train` (speculators 7046ac6 + our five patches, a byte-for-byte
copy of the Clariden checkout), layers `1 20 39 58 77`, file backend to
node-local `/tmp/hidden_states`, TP2 on GPUs 0,1, `--enforce-eager`,
`--provenance-dir`, max-model-len 8192. It then runs `check_hidden_states.py`
on GPU 2 against `apertus-dspark70b/pilot/tp4-seqs512.jsonl` (16 rows, as on
Clariden).

| | vLLM 0.23 image (693748) | vLLM 0.31 image (693758) | Clariden fork (reference) |
|---|---|---|---|
| shape (T, 6, 8192) bf16, ids, finite | 16/16 rows | 16/16 rows | yes |
| top-1 agreement, answer tokens | **0.980** (579/591) | **0.978** (578/591) | 0.983 |
| server flags beyond Clariden's | `--no-enable-chunked-prefill --no-enable-prefix-caching` | none | — |
| server up | 1,000 s | ~1,000 s | ~600 s |
| 16 rows checked in | 260 s | 210 s | — |

Per-layer mean norms agree between the two images to <0.5 %. The Clariden
number is from `docs/dspark-70b-plan.md`; I assume the same 16 rows (same
script defaults and file).

The vLLM 0.23 image's `ExampleHiddenStatesConnector` predates the fork's. It
needs chunked prefill off (config validation), and it saves only positions
computed in the step, so prefix-cache hits gave truncated states (693726).
`check_hidden_states.py` now reports such a shape mismatch instead of
crashing. vLLM 0.31's connector has neither problem and runs with Clariden's
settings, so it is the better base.

## Training smoke (step 4)

`train-smoke.sbatch` runs the Clariden `train-node.sh` inside the vLLM 0.23
image: TP2 extraction on GPUs 0,1 and two FSDP trainer ranks on GPUs 2,3, with
Yu's recipe (DSpark 5 layers, block 8, Markov 256, confidence head, CE 0.1 +
TV 0.9, Muon, `--fsdp-shard`) on the prepared 70B corpus. Changes to the
shared scripts are opt-in (`PY_PRELUDE`, `EXTRACT_FLAGS`) or fallbacks
(Python health probe, amd-smi), with Clariden defaults unchanged.

Job 693757 (SEQ 4096, 256 anchors per rank, i.e. half the per-rank batch of
the planned 16-node run; 2 ranks, so 2-way sharding):

- 30 steps, loss 0.87–0.94 over steps 22–28, accept_len ~1.03 (fresh drafter),
  data fetch 0.4 % of the step.
- per step: fwd 0.68 s, bwd 0.56 s, **optimizer (Muon) 3.37 s**, total 4.6 s.
  Clariden 16-node pilots, full per-rank batch at 32 ranks: fwd 0.56–0.72 s,
  bwd 0.81–1.06 s, opt 2.2–2.6 s, 4.0 s. Per token the forward here is
  ~2× Clariden's, but both include FSDP gathers at very different rank
  counts. Muon at 2 ranks says little about Muon at 32, so the per-step time
  at 16 nodes has to be measured.
- memory: extraction GPUs 2 × 102 GiB (`EXTRACT_MEM` 0.75), trainer GPUs
  2 × 91.5 GiB, node `used` 473–483 of 501 GB while training. The epoch-end
  save (`get_model_state_dict(full_state_dict, cpu_offload)` on rank 0) then
  pushed the node to 499 GB with 1 GB available and stalled; cancelled.

### Save and resume (job 693772)

PENDING

## Memory budget for the planned 16-node layout (estimate, not measured)

MI300A has no separate host memory: `free` shows ~501 GB, and GPU
allocations, `/tmp` (hidden-state files) and page cache all come from it.

| Per node | 1-node smoke (measured) | 16-node layout (estimate) |
|---|---|---|
| extraction, 2 GPUs | 2 × 102 GiB at 0.75 | 2 × ~80 GiB at ~0.6 (67 GiB weights + KV and state buffers) |
| trainer ranks, 2 GPUs | 2 × 91.5 GiB (2-way FSDP, half batch) | 2 × ~45 GiB (Clariden 16-node pilot: 44.8 GiB at 32-way FSDP, full batch) |
| host processes | ~64 GB | ~64 GB + hidden-state files in `/tmp` |
| node 0 during a save | ran out | + full model and optimizer state on rank 0's host (tens of GB; Clariden checkpoint 25 GB) |
| total | 483–499 GB | ~330 GB + save peak |

So the 16-node layout probably fits, and the 4-node layout (trainer GPUs
82–97 GB on Clariden) probably does not. A 2-node pilot with a forced early
checkpoint and a resume should confirm this before any chained run.

## What the full port needs

1. **Image.** Prefer `vllm/vllm-openai-rocm:v0.31.0` (same connector as the
   fork, Clariden's server flags; CSCS's recommended image) and run the
   training smoke on it; the smokes above trained on the 0.23 image only.
   Record its digest as for the Clariden images (`serving/images/`), or build
   our own on top of it with the plugin and side deps baked in. If the exact
   fork model code is wanted, `dspark-upstream-adaptive`'s
   `vllm-apertus-mm-v0.31.patch` targets the same vLLM version, but needs the
   Swiss AI transformers fork.
2. **Memory.** `EXTRACT_MEM` ~0.6 and a 2-node pilot with an early checkpoint
   and a resume; watch `logs/mem-<job>.log`.
3. **Multi-node.** `train-smoke.sbatch` is single-node; a Beverin
   `submit-train.sh` path (EDF, `PY_PRELUDE`, `EXTRACT_FLAGS`, unset
   `ROCR_VISIBLE_DEVICES`, 24 h limit instead of 12 h) is about half a day.
   Then pick all-to-all or all-gather Muon from a 16-node pilot as on Clariden.
4. **Load time.** 11 min of weight loading per job from capstor. Fewer,
   24-hour chain links help; copying the target to iopsstor is worth a test.
   Node-local `/tmp` would cost HBM.
5. **Smaller items.** `curl` is broken in the 0.23 image (Python probe used).
   tvm-ffi builds a small extension into `~/.cache/tvm-ffi` (home). The
   container + Python import preamble before the server starts took 1.5–11
   min, varying with iopsstor.

## Files

- `methods/dspark/beverin/`: `apertus15_text/` (plugin) + dist-info,
  `apertus15_run.py`, `env-vllm023.toml`, `env-vllm031.toml`, `env.sh`,
  `import-image.sbatch`, `smoke.sbatch` (+ `serve-smoke.sh`,
  `extract-check.sh`, `generate_smoke.py`), `train-smoke.sbatch`,
  `rccl-test.sbatch` + `rccl_bench.py`.
- `methods/dspark/train/`: ROCm fallbacks and opt-in hooks
  (`common.sh`, `train-node.sh`, `rank-wrapper.sh`). `train-node.sh` now passes
  `--provenance-dir` (AGENTS.md). `check_hidden_states.py` reports shape
  mismatches.

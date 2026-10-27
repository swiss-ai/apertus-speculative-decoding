# DSpark 70B on Beverin (MI300A): feasibility check, 2026-10-24

Question: can the DSpark 70B training (online hidden-state extraction from
Apertus-v1.5-70B + speculators trainer, `methods/dspark/train/`) run on CSCS
Beverin's `mi300` partition instead of Clariden's saturated `normal` partition?

**Current update (2026-10-27):** the rocBLAS workaround is validated on
16 nodes at approximately 41 training hours/epoch, with save/resume passing;
a 32-node comparison is running. The one-node production run continues.
See the scaling follow-up below. The initial feasibility summary and slow
16-node results that follow are historical.

**Update 2026-10-25: the 16-node pilot trained and saved but ran at ~200 s per
step against Clariden's 4.1 s, so the epoch run was not launched (see "16-node
pilot").** Original verdict of the one-node check: **feasible with work.** On one MI300A node the 70B served
correctly in a public ROCm vLLM image. The extraction server produced hidden
states that pass `check_hidden_states.py` at Clariden's level (0.980 and 0.978
top-1 agreement vs 0.983). The unchanged Clariden `train-node.sh` also trained
30 DSpark steps there (FSDP, Muon, flex attention on ROCm torch), and RCCL
runs over Slingshot between nodes. Three things block a production run:

1. **Memory.** On MI300A the four GPUs and the host share one ~501 GB pool per
   node. The one-node smokes used 430–483 GB while training. Rank 0 gathers
   the full state dict on its host for an epoch-end checkpoint save. One save
   succeeded (25 GB checkpoint, and a resume from it worked); two others
   stalled or were OOM-killed. The planned 16-node layout should fit (estimate
   below), but saves there are unverified.
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
| 693772 | training smoke, smaller (SEQ 2048, 128 anchors, `EXTRACT_MEM` 0.62) | 8 steps, **checkpoint saved** (25.5 GB, ~1 min), then cancelled in the full validation pass |
| 693792 | resume of 693772's run | **resumed** from the checkpoint (epoch 2, steps 8–11); the next epoch-end save was **OOM-killed** (Slurm: 1 oom_kill event) |

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

### Save and resume (jobs 693772, 693792)

Same image and recipe, run smaller: SEQ 2048, 128 anchors, `EXTRACT_MEM` 0.62,
`MAX_STEPS` 8 and then 12.

- 693772: 8 steps (step ~4.5 s: fwd 0.65–0.75 s, bwd 0.35 s, Muon 3.3–4.0 s).
  The epoch-end save wrote `model.safetensors` 12.95 GB and
  `optimizer_state_dict.pt` 13.16 GB in ~1 min, the same 25 GB as Clariden's.
  Node `used` peaked at ≥462 GB (60 s sampling). After `MAX_STEPS` the trainer
  starts a full validation pass over the 175k held-out rows; I cancelled
  it there.
- 693792, same `RUN_ID`: "Found checkpoint … Resuming training on epoch 1",
  then steps 8–11 (the first step took 108 s, warm-up). The next epoch-end save
  was killed by the kernel OOM killer: rank 0 got SIGKILL, Slurm reported an
  oom_kill event, and the memory log stalled for 2.5 min.

Even at a quarter of the planned per-rank batch, the two trainer GPUs held
88–95 GiB each. That is mostly the 2-way-sharded fp32 state (~75 GB
unsharded) and the allocator cache, which on MI300A is node memory too. The
extraction GPUs held 85–89 GiB at `EXTRACT_MEM` 0.62. Saves on one node
therefore succeed or fail by a few GB.

## 16-node pilot (2026-10-25): runs, saves, but ~50× too slow

Settings as Clariden's 16-node run (`opb-thinkoff-16n`): same data, 32 ranks
(TP2 extraction on GPUs 0,1 and 2 trainer ranks on GPUs 2,3 per node), SEQ
8192, 512 anchors, LR 6e-4, 10-epoch schedule, all-gather Muon
(`DSPARK_DTENSOR_ALLGATHER=1`), layers `1 20 39 58 77`, speculators 7046ac6 +
all five patches (rebuilt from `patches/`, identical to the Clariden copy).
Image `vllm-openai-rocm-v0.31.0.sqsh` + `apertus15_text`, stock server flags,
`EXTRACT_MEM` 0.62. Submitted by `methods/dspark/beverin/submit-train.sh`
(run `opb-thinkoff-16n-beverin-pilot`, `CKPT_FREQ` 0.025, 2 h 45 min, deadline
stop 40 min before the end).

| | Beverin pilot 693916 | Clariden 16-node (all-gather) |
|---|---|---|
| all 16 extraction servers up | 27 min after start (concurrent loads from capstor; first node 10 min) | — |
| step time, steady (steps 1–29) | **191–214 s** | 4.09 s |
| forward / backward / optimizer | 0.9 s / 54–58 s / 136–151 s | 0.56 s / 0.81 s / 2.6 s |
| steps in 1 h 47 min of training | 30 | — |
| one epoch (~8,300 steps) | **~19 days** | ~9.5 h |
| node memory while training | 360–372 GB of 501 | — |
| deadline-stop save at step 30 | **OK**, ~1 min, 25.5 GB; peak 424 GB on node 0, ≤390 GB elsewhere (10 s sampling) | — |
| hang / stall | none (rc 0, no STALL) | — |

Trainer GPUs held 50–55 GiB and extraction GPUs ~85 GiB, close to the
memory estimate below. Memory is not the problem at 16 nodes.

The time goes to inter-node communication inside training, not compute:

- 1 node, 2 ranks (693757): backward 0.56 s, Muon 3.4 s.
- 2 nodes, 4 ranks, same settings (693946, all-gather Muon): backward 50 s,
  Muon 130 s, 181 s per step. 693947 (all-to-all Muon) also took 75 s backward
  and 132 s Muon on its first step. The slowdown does not grow with node count
  (2 → 16) and does not depend on the Muon mode.
- Microbenchmarks of the same collectives in the same image and layout (2
  ranks per node on GPUs 2,3, bf16 and fp32) run at full speed. On 4 nodes
  (693943, 693944) all-reduce, all-gather and reduce-scatter reach ~68 GB/s
  and all-to-all ~20 GB/s. `FI_MR_CACHE_MONITOR=disabled` made no
  difference.
- With a fresh allocation per call (new buffer addresses), 512 MiB
  reduce-scatter and all-gather drop to 6–9 GB/s (693957–693960; includes the
  allocation itself). `OFI_NCCL_DISABLE_DMABUF=1`, `FI_MR_CACHE_MONITOR=disabled`
  and `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` did not help. This
  points at buffer registration with the network on MI300A, but it does not
  explain a 100× gap, so the cause is still open.
- SIGUSR1 stack dumps (speculators-stack-dump.patch) did not appear in the
  trainer log when sent by hand, so the stacks are unknown.

Not done: a resume within the 16-node allocation. At ~200 s per step it would
not change the verdict, and the one-node resume (693792) worked. The epoch
run (`opb-thinkoff-16n-beverin`) was **not launched**.

Next steps, if Beverin is still wanted:
1. Profile two nodes (torch profiler or `NCCL_DEBUG=INFO
   NCCL_DEBUG_SUBSYS=NET,REG,COLL`, plus the aws-ofi MR-cache hit counts) to
   see which collective is slow, and whether it is registration, a host
   bounce, or proxy progress.
2. Try FSDP2 without fp32 gradient reduction (bf16 reduce-scatter), RCCL
   `NCCL_PROTO`/channel settings, and the netstack `host` source with the
   `rocm6` plugin (`com.hooks.netstack.source = "host"`) instead of the
   rocm7 artifact.
3. Ask CSCS support: RCCL inside PyTorch FSDP across MI300A nodes, collectives
   ~100× slower than rccl-tests-style loops.

## Handoff continuation (2026-10-25)

User confirmed **LR 1.5e-4** for the one-node, 16,384-token step. No Clariden
commands were run. Its jobs and checkouts were left untouched.

### Outcome: production run started; rocBLAS fixes the slow training path

The production pilot **694469** completed normally in 37 minutes, saving at
step **20**, resuming exactly from 20, and saving again at **40**. Its
38 steady samples (excluding the first step of each attempt) have median
**12.45 s/step**. Backward is **0.674 s** and Muon/AdamW optimizer work
**2.89 s**, compared with ~56 s and ~222 s in the one-node hipBLASLt run.
Target-feature production now limits much of the step; median fetch is
6.285 s with one worker and prefetch 1. These phase medians are not additive.

The ten-second memory log's peak during the first save was **382 GiB** with
at least **104 GiB available**, and during the resumed save **403 GiB** with
at least **83 GiB available**. Across the entire pilot the sampled peak was
**407 GiB**, minimum available **77 GiB**. These are the container's
`free -g` fields, not a claim about unobserved sub-ten-second peaks.

The ten-epoch run is **`opb-thinkoff-1n-beverin`**, LR **1.5e-4**, with seven
24-hour continuations **694472–694478**. They were held until save/resume and
memory checks passed, then released at 15:40 UTC on 2026-10-25. Job **694472**
started on `nid002988`; its successors depend on the prior link finishing.
Each link writes a progress/resume audit. This initial chain covers seven
days, not the entire ten-epoch schedule. Further continuations use the same
run ID and must depend on 694478 while this chain remains queued/running.
The measured 141,472-step epoch at 12.45 s/step is approximately **20.4 days**
before validation and restart overhead, rather than the initial eight-day
estimate. The memory-safe worker/prefetch settings are retained.

The matching two-node rocBLAS pilot **694470** also completed and saved.
Its five steady samples have median **14.8 s/step**, versus **183 s** in the
otherwise matching hipBLASLt baseline: backward **0.674 s**, optimizer
**2.99 s**, and forward **9.27 s**. The CPU profiler serializes/times GPU
work, and hidden-state production can be charged to different phase
boundaries; the full-step median is the useful comparison. Node 0 and node 1
each peaked at **364 GiB**, with **120 GiB** minimum available. **The large
slowdown is removed; no training run above two nodes was launched.**

The workaround is trainer-only:
`BEVERIN_BLAS_BACKEND=cublas` calls
`torch.backends.cuda.preferred_blas_library("cublas")` on ROCm, selecting
rocBLAS instead of the image's default hipBLASLt. Extraction keeps the same
v0.31 server and model path, including working chunked prefill/prefix
caching. This changes the GEMM backend; it leaves the drafter recipe,
precision policy, optimizer and dataset unchanged. Both the environment
setting and runtime hook source are in the provenance.

### Memory and one-node save/resume pilots

The full-context pilot retains SEQ 8192, 512 anchors, the ten-epoch schedule,
TP2 extraction on GPUs 0,1 and two FSDP trainer ranks on GPUs 2,3. Host
memory mitigations: `WORKERS=1`, `PREFETCH=1` (previously 4/2), one API
process and one renderer worker (rather than the launcher's automatic 9/2),
and collection of Python garbage plus release of unused HIP allocator blocks
before a checkpoint save. Model, loss, optimizer, context and batch are
unchanged. A Beverin-only step-boundary hook forces an early mid-epoch save,
then restarts torchrun **within the allocation with the extraction server
kept up**. It preserves the ten-epoch LR schedule and avoids `MAX_STEPS`,
which would enter a full validation pass.

`EXTRACT_MEM=0.56` was too low: vLLM failed with "No available memory for the
cache blocks" before training. Job 694384 failed this way; diagnostic jobs
694385 and 694389 inherited the same setting and also failed before training.
Queued jobs 694390–694393 were held and cancelled before allocation, then
replaced. **0.62 is retained** for all replacement jobs. The initial launcher
returned allocations on extraction startup failure; the updated Beverin
wrapper retries startup up to three times in the allocation. Trainer retries
continue to preserve the extraction servers as before.

| Job | Run / image | Result |
|---|---|---|
| 694384 | `opb-thinkoff-1n-beverin-pilot`, v0.31, extraction memory 0.56 | no KV-cache space; no training |
| 694425 | `opb-thinkoff-1n-beverin-pilot-v2`, v0.31 | ~279 s/step; saved at 10 and logged resume; reached 90-minute limit |
| 694435 | `opb-thinkoff-1n-beverin-pilot-v023`, v0.23 | saved at 20, resumed to 40; 11.35 s/step combined steady median; peak 430 GiB, minimum available 71 GiB |
| 694469 | `opb-thinkoff-1n-beverin`, v0.31 + rocBLAS | save/resume passed; 12.45 s/step; production checkpoint at 40 |

**The previous one-node / multi-node comparison was confounded by the
image.** The 4.6 s reference used vLLM 0.23 / torch 2.11, whereas the slow
multi-node pilot used vLLM 0.31 / torch 2.13. The current v0.31 one-node
measurement has backward ~56 s and optimizer ~222 s. This substantially
changes the diagnosis: crossing the node boundary is not necessary for the
slowdown. Memory during that run is around 399 GiB, so it is not at the
physical limit.

### Two-node profiling and isolated fixes

All replacement diagnostic runs use SEQ 8192, 512 anchors per rank, the same
0.62 extraction memory and 1/1 workers/prefetch, six training steps, and
separate run/checkpoint directories. They have 90-minute limits and
`--nice=10000`; the one-node pilot was allocated first. The matrix launcher
supports bounded lanes (default two). Six lanes were used with the observed
spare capacity; every training trial still uses only two nodes. The MR test
replaced the failed host-netstack trial's slot, with its original baseline
completion dependency removed and recorded in `logs/schedule-change-694432.txt`.
No larger training allocation was launched.

Steady timings below exclude the first compiling step; each median has five
samples. The baseline additionally captured three CPU torch-profiler steps.
All tests record `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,REG,COLL`.
Overrides are applied **after** the enroot hook, which otherwise overwrites
several of these settings.

| Job | Change from baseline | Median step / backward / optimizer (s) |
|---|---|---|
| 694426 | v0.31 baseline + CPU profiler | **183 / 51.8 / 128** |
| 694427 | FSDP `MixedPrecisionPolicy(reduce_dtype=torch.bfloat16)` | **193 / 55.3 / 136** |
| 694428 | `NCCL_PROTO=Simple` | **178 / 51.5 / 123** |
| 694429 | `NCCL_MIN_NCHANNELS=16` | **179 / 51.9 / 127** |
| 694430 | `NCCL_NCHANNELS_PER_NET_PEER=8` | **181 / 50.3 / 130** |
| 694431 | host netstack, rocm6 plugin | fails before serving: injected `libselinux.so.1` requires GLIBC_2.38; image libc lacks it |
| 694432 | `FI_MR_CACHE_MONITOR=disabled`, applied inside container | **175 / 48.3 / 126** |
| 694436 | v0.23 / torch 2.11, otherwise matching settings | **16.1 / 0.822 / 3.16**; saved at 6 |
| 694440 | isolated 1- and 2-node sharded-matmul comparison, both images | completed; see below |
| 694470 | v0.31 + trainer rocBLAS, two profiled steps | **14.8 / 0.674 / 2.99**; saved at 6 |

The baseline CPU profiler attributes 51.55% of self CPU time to `aten::addmm`
and 46.50% to `aten::mm`: together 533.6 s across three profiled steps. The
corresponding average CPU total durations are ~248 and 250 ms per call.
In contrast, `_c10d_functional::wait_tensor` totals 71.9 ms over 1,881 calls,
and `FSDP::post_backward_reduce (layers.4)` totals 327 ms over three calls.
These are CPU timings, **not direct GPU network-kernel timings**: they point
toward the sharded-matmul path but do not by themselves locate the native
cause. Full Chrome traces for all four ranks and the tables are preserved.

A small exploratory GEMM probe in v0.31 found ordinary 1024-square GPU
matmuls fast with either BLAS preference. A same-node DTensor probe found
replicated matmuls fast and sharded matmuls slow; because it shared an
allocation with training, job 694440 repeats this on isolated GPUs with
Python profiling. These probes do not contribute to the training medians.

Job **694440** reproduced this on isolated GPUs. Each rank has a 1024×1024
local BF16 tensor; global sharded size scales with world size. Warmed medians
omit iteration 0 and aggregate all ranks:

| Image / backend | One-node sharded matmul | Two-node sharded matmul |
|---|---|---|
| v0.31 default hipBLASLt | **478 ms** | **516 ms** |
| v0.23 default | **1.28 ms** | **4.81 ms** |
| v0.31 forced rocBLAS | **0.402 ms** | validated by full training in 694470 |

Plain matmuls were ~0.05 ms and replicated DTensor matmuls ~0.1 ms in both
images. Python profiling attributes the slow sharded call to the native ATen
operation. Switching BLAS preference in the same image removes the delay.
This isolates a hipBLASLt-dependent slow path; its native cause and the role
of operand shape/stride or gathered-buffer properties remain for CSCS/AMD
support. It is not established as a PyTorch-only regression, since the images
also have different ROCm versions.

The requested minimum of 16 channels in 694429 left the actual collective
channel count at 36, as in the baseline. The per-peer trial 694430 did change
that effective setting from four to eight. All runs used the AWS Libfabric
plugin with provider cxi; no host/TCP fallback was identified.

**Correction to the earlier MR-cache conclusions:** the raw logs for jobs
693944 and 693959 still show `FI_MR_CACHE_MONITOR=userfaultfd`, despite the
intended disabled setting. The container hook overrode it, so those jobs
were not valid disabled-monitor controls. Job 694432 is the actual test;
it still does not restore useful training speed.

### Provenance and chain checks

Source snapshots are new directories under `beverin-feasibility/`:
`repo-d31c7d63/` (initial jobs), `repo-6e2d7693/` (cancelled matrix),
`repo-c977af19/` (replacement matrix), and `repo-786ebde3/` (older-image
pilots and save timestamps), and `repo-ada15200/` (rocBLAS pilots and the
production chain). The old `repo/` and completed raw results remain
intact. The new `speculators-7046ac6-provenance/` retains its `.git` directory
at `7046ac63fac4eeb218dc2e92a385c7d1ec8c4e00` plus all five original patches,
so the built-in writers record its SHA and diff instead of "unknown".

Each job records launch script, EDF, revisions, Speculators patch, Slurm
job/node identities, image hash and resolved non-secret network environment
under `runs/<run>/deployments/<job>/`. vLLM provenance stays under
`runs/<run>/logs/vllm-provenance-<job>-node<N>/`; resolved training config
and provenance stay with the checkpoints and are copied per deployment.
Ten-second memory logs use `free -g`, so the inherited `*_gb` columns are
actually GiB. Each new chain link audits checkpoint files, saved-step
advancement and the expected logged resume, writing `job-audit.json`.

- v0.31 sqsh SHA-256:
  `b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f`.
- v0.23 sqsh SHA-256:
  `75f5553dbe4b4a8bc431bfd3fe78b38ebf9724d09dfc0a2a4e7c85b0f84868df`.
- Regenerated corpus SHA-256:
  `91b9f8f8dd28b790357919ec884c8dbfa9579d67200611a525b915dd174bd7fd`.
- SHA-256 of the prepared-dataset content-hash manifest (all 75 Arrow shards,
  JSON metadata, token frequencies and held-out IDs):
  `dbf6b72a266bf6e808b893a2fdc8e9ae344d7128092880afd6e934380a617ee4`.
- Canonical raw logs, profiles, configurations and provenance:
  [`results/70b/dspark/beverin-20261025/`](../results/70b/dspark/beverin-20261025/).
- [CSCS support draft](beverin-support-ticket.md), not sent.

## Scaling follow-up (2026-10-27): 16 nodes validated; 32-node comparison running

The user authorized a 16-node save/resume pilot and a conditional 32-node
comparison, keeping the one-node production run intact. No full multi-node
production chain has been submitted. No Clariden jobs were queried or changed.

**16-node job 697549 completed** (09:34:49–11:19:03 UTC, 1:44:14). It used
32 trainer ranks, nominal batch 262,144 tokens, LR 6e-4, SEQ 8192 / 512
anchors, `repo-ada15200/`, v0.31 + trainer rocBLAS, and the existing safe
worker/extraction settings. CPU profiling was off. It saved at 100,
resumed from the exact epoch/local/global cursor 0/100/100, advanced with
continuous logged LR, saved normally at 128 and saved/exited at 200.
All 15 verification checks passed. All nodes confirmed rocBLAS with no CPU
profiling; reported data-recovery errors remained zero and losses finite.

| Reference | Steady samples | Mean / median step | Exact steps/epoch | Projected training time/epoch | Training node-hours/epoch |
|---|---:|---:|---:|---:|---:|
| 1 node, production snapshot 694473 | 200 | 14.490 / 14.8 s | 141,472 | 569.43 h (23.73 d) | 569.4 |
| 2 nodes, debug 694470 | 5 | 14.36 / 14.8 s | 67,783 | 270.38 h (11.27 d) | 540.8 |
| 16 nodes, 697549 | 197 | **18.036 / 18.2 s** | **8,179** | **40.98 h (1.71 d)** | **655.6** |
| 32 nodes, 697813 | running | pending measurement | pending measurement | pending measurement | pending measurement |

The two-node reference includes two CPU-profiled steps. The one-node window
is steps 10,530–10,729, captured 09:53:50 UTC. The earlier one-node pilot's
38 steady samples were faster (12.45 s median); timing varies with the
window. Against the fresh production reference, 16 nodes give 13.90x epoch
speedup, 86.9% parallel efficiency and 15.1% more training node-hours.
The step-count benefit includes better packing, so counts are taken from
logs rather than divided by node count.

The 16-node mean projects **17.07 training days for ten epochs**. The three
save durations were 90.58, 69.33 and 61.97 s; their mean implies roughly
**1.31 additional hours per epoch** at the configured checkpoint frequency.
Target startup took about 29 minutes, with the first compiling step taking
102 s. The first resumed step took 57.3 s. Validation and queue waits remain
unmeasured. Steady timings omit the first step of each attempt and the
post-checkpoint step whose fetch timer includes save latency; save costs
are reported separately.

Whole-job 10-second memory samples peak at **340 GiB used**, minimum
available **144 GiB**. The first-save window peaks at 330 GiB, minimum
available 156 GiB. This supports the conditional 32-node comparison.
**697813** was submitted at **13:17:22 UTC**, uses 64 trainer ranks,
524,288 nominal tokens/step and LR **8.48528137423857e-4**, with the same
100+100-step save/resume plan and source/recipe/settings. Both pilots have
three-hour limits, 20-minute deadline-save margins and `--nice=10000`.
The LR values follow the square-root batch heuristic; short pilots do not
establish training-quality equivalence or multi-day stability.

The initial monitoring stream stopped before 697549 completed. A fresh
check at 13:15 UTC found the successful completion; full raw logs/audits
cover the entire run. Independent status snapshots now monitor 697813.
Production 694473 remained running when 697813 was submitted, and all five
successors were intact.

Raw logs, launchers, image/data/target digests, runtime sources, configurations,
checkpoint metadata, memory samples and reproducible verification scripts:
[`results/70b/dspark/beverin-scaling-20261027/`](../results/70b/dspark/beverin-scaling-20261027/).
Large model/optimizer files remain at the recorded cluster run paths.

## Memory budget for the planned 16-node layout (estimate, not measured)

MI300A has no separate host memory: `free` shows ~501 GB, and GPU
allocations, `/tmp` (hidden-state files) and page cache all come from it.

| Per node | 1-node smoke (measured) | 16-node layout (estimate) |
|---|---|---|
| extraction, 2 GPUs | 2 × 102 GiB at 0.75 | 2 × ~80 GiB at ~0.6 (67 GiB weights + KV and state buffers) |
| trainer ranks, 2 GPUs | 2 × 91.5 GiB (2-way FSDP, half batch) | 2 × ~45 GiB (Clariden 16-node pilot: 44.8 GiB at 32-way FSDP, full batch) |
| host processes | ~64 GB | ~64 GB + hidden-state files in `/tmp` |
| node 0 during a save | 1 of 3 saves survived | + full model and optimizer state on rank 0's host (fp32 gather, then 25 GB bf16) |
| total | 483–499 GB | ~330 GB + save peak |

So the 16-node layout probably fits, and the 4-node layout (trainer GPUs
82–97 GB on Clariden) probably does not. A 2-node pilot cannot settle this,
because its 4-way sharding still holds ~19 GB of state per rank. Run the
16-node pilot with an early forced checkpoint, a resume, and a 10 s memory
log on node 0. Cheap mitigations if it is tight: a lower `EXTRACT_MEM`,
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` to keep the allocator
cache from holding node memory, or sharded (DCP) checkpoints instead of
the full state dict on rank 0.

## What the full port needs

1. **Image.** Prefer `vllm/vllm-openai-rocm:v0.31.0` (same connector as the
   fork, Clariden's server flags; CSCS's recommended image) and run the
   training smoke on it; the smokes above trained on the 0.23 image only.
   Record its digest as for the Clariden images (`serving/images/`), or build
   our own on top of it with the plugin and side deps baked in. If the exact
   fork model code is wanted, `dspark-upstream-adaptive`'s
   `vllm-apertus-mm-v0.31.patch` targets the same vLLM version, but needs the
   Swiss AI transformers fork.
2. **Memory.** `EXTRACT_MEM` ~0.6. Verify saves at 16 nodes, and add a
   finer memory log (see the budget section). If needed, sharded
   checkpoints (a speculators patch, ~half a day).
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

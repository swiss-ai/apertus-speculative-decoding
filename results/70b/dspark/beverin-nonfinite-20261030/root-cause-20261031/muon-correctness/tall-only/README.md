# Tall Muon oracle: one paired native failure, job 710518

The original image aborted on both GPU ranks; the ALPS image passed on the
same node and physical GPUs 2/3. This was one observed paired test, with no
model, corpus, optimizer training or learning rate. It does not reproduce
the historical training NaN or identify a specific failing kernel.

Job 710518 used nid002488 for 255 seconds on 31 October 2026,
16:33:41–16:37:56 UTC. It had one attempt, requeue disabled, a 20-minute limit,
and sequential original/ALPS cells capped at 360 seconds each. The immutable
source revision was `0030d3123e3bf7b8fa4f6c41e3615ab57b45203c`.

| Image | SquashFS SHA256 |
|---|---|
| `vllm-openai-rocm-v0.31.0.sqsh` | `b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f` |
| `alps-vllm-rocm-7d95232cbc75.sqsh` | `416216c9de21d110fbb7cb12cef90923af51e04c955d7d7dffa9786f125a4928` |

Both complete image hashes were asserted before execution. The ALPS OCI
identity is `sha256:7d95232cbc7511be5672d272f7a567ea52a3c5fbd5310ce0903956f0f37b5cc7`.

The unchanged probe calls the installed private
`torch.optim._muon._zeropower_via_newtonschulz` on independent BF16 DTensor
and dense GPU clones. Its canonical function-source SHA256 is
`99e2359a95e1a914345207e49f072a658866333f4458b7b45faafa35c628f7c4`.
It uses five iterations, coefficients `(3.4445, -4.7750, 2.0315)`, epsilon
`1e-7`, forced rocBLAS (`preferred_blas_library('cublas')`), and the production
all-gather/chunk redistribution workaround. Fixtures cover zero, tiny random,
random, skewed random, uniform rank-one and signed rank-one gradients.

The explicit setting `MUON_ORACLE_SHAPES='80x16 43008x8192'` exercises the
production gate/up geometry and the initial transpose: row sharding becomes
column sharding, with partial Gram reductions and nonunit-beta addmm. The
previous wide control, job 710498, passed both images at `16x80` and
`8192x40960`. With two ranks, the tall local shape is `21504x8192`; the
historical 64-rank local shape was `672x8192`.

The exact launch argv, resolved environment, source manifest, image paths,
source revision and Slurm state are in [submission/submission.json](submission/submission.json)
and [results-710518/raw/](results-710518/raw/). The launch used `sbatch
--no-requeue` with the shape setting exported explicitly; both rank argv
and the final batch summary verify the exact tall shapes.

The original cell passed all six small cases per rank, then both processes
aborted during the first large zero-gradient case. ROCm reported
`HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`, code `0x2a`; torchrun reported
SIGABRT on both ranks. No numerical assertion or large-case comparison was
completed. The ALPS cell passed all 12 cases per rank with finite outputs;
its maximum raw-update relative L2 difference was 3.774%, within the stated
5% BF16 tolerance. The large zero case matched the dense reference exactly.

Both cells used PCI bus IDs `202000`/`302000`. The original used P2P/IPC;
ALPS used SHM with P2P disabled. ROCm, RCCL and other native libraries also
differ, so this comparison does not isolate one component. The oracle adds
input reconstruction and final-update `full_tensor()` gathers; those are
not guaranteed to occur in production Muon. The last visible 672 MiB
collective therefore cannot localize the fault to the Newton–Schulz kernel.

Four core files remain intact under:

```
/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-tall-correctness-20261031T163340Z/deployments/710518/
  core_nid002488_173742
  core_nid002488_173742.gpu
  core_nid002488_173743
  core_nid002488_173743.gpu
```

The cores were not copied or fully scanned/hashed on the login node. Bounded
metadata reads totaled **355,356 bytes**, below the 4 MiB limit. Each GPU
note identifies queue 1 as COMPUTE_AQL with exception mask `0x8`, the illegal
instruction bit; the other queues have no exception. Active dispatch position
and kernel identity were not established. Offset-16 descriptor-derived entry
addresses in the raw observations are unvalidated and rejected as findings.
See [core-metadata/README.md](core-metadata/README.md).

All 43 small raw artifacts, 621,779 bytes, are retained with remote hashes
and a failure-aware verifier in [results-710518/](results-710518/). No further
job or experiment was launched after this test. Nothing was sent externally.

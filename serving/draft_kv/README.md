# Experimental DSpark drafter KV overlay

This opt-in backend was evaluated with the trained Apertus 1.5 8B DSpark
thinking-off head. The [validation screen](../../docs/drafter-kv-sparsity.md)
failed the serving-performance and acceptance gates. Native 2,048-token drafter
caching remains the recommended configuration for that head.

`integration.py` adapts the pinned vLLM allocator and drafter lifecycle;
`attention.py` supplies sparse attention and page-selection kernels;
`graphs.py` manages bounded query CUDA graphs. `policy.py` is the CPU reference,
and `tracing.py` contains GPU-independent trace scheduling. `check_gpu.py` is the
installed-runtime CUDA regression suite and requires the recorded serving image.

`prepare-overlay.sh` extracts source from the image and invokes `install.py`
after the normal Apertus compatibility patches have been applied. The installer
writes transformed files and the complete helper package, records their hashes,
and emits container mounts plus the exact applied patch. The generated patch and
`serving/patches/vllm-apertus-drafter-kv.patch` require these helper mounts; they
are not standalone `SWEEP_PATCHES` entries. `pinned-source.json` records the
upstream source revision and audit hashes.

The shared `serving/node-sweep.sh` launcher enables the overlay only when
`SWEEP_DRAFT_KV=1`. `SWEEP_DRAFT_KV_PREFLIGHT=1` additionally runs the installed
GPU checks before serving and requires that overlay flag. Sparse policies are
selected in the arm's environment through `VLLM_APERTUS_DRAFT_KV_POLICY`,
`VLLM_APERTUS_DRAFT_KV_BUDGET` and `VLLM_APERTUS_DRAFT_KV_RECENT`; defaults retain
native behavior. Each deployment needs a fresh output and overlay directory.

Campaign generation, input hashing and submission entrypoints live in
[`experiments/drafter-kv-8b/`](../../experiments/drafter-kv-8b/README.md), while
shared analysis and controller code live in `src/apertus_bench/`. Completed raw
artifacts and executed source snapshots remain immutable under `results/`.

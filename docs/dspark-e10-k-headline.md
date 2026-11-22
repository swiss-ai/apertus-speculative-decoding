# DSpark e10 in production: headline speedups at k = 3, 5, 7

Status 2026-11-22: running (job 4004240, one node, d_research_serving).
Branch `experiment/dspark-e10-k-headline`.

Action item B from the 2026-11-20 sync (transcript 45:38–46:00): the production
deployment was only measured at its deployed draft length k=3; the leads slide
deck and the Hugging Face release need its speedup at k=3, 5 and 7.

## What is measured

**Production** = Apertus-v1.5-8B with the DSpark e10 head (thinking off) as served
on the CSCS serving platform: TP1 bf16, max-model-len 32768, gpu-memory 0.8,
max-num-batched-tokens 16384, max-num-seqs 256, prefix caching off, the forced
thinking-off chat template, the `vllm_apertus_1.5_release` image (sha256
`f5fc017f…`) with the four DSpark source overlays. Only the draft length changes
between the DSpark arms. The live pod was not touched: the k=3 campaign showed
live and controlled speedups within ±7% (`docs/dspark-e10-task-speed.md`), so
controlled same-GPU runs stand in for it.

**Design.** `experiments/task-speed/submit-controlled.sh` with `TASK_SPEED_KS=3,5,7`:
- one node, 4 GPUs; each GPU runs a block of four fresh deployments, plain, k3, k5, k7,
  one after the other; the order is rotated by GPU (Latin square), so each arm runs in
  each position once;
- the 28-slice task-speed suite unchanged (`experiments/task-speed/suite.yaml`):
  ID, near-ID, OOD task and OOD language classes, 32 prompts per slice, greedy,
  natural EOS at each source's output cap, C = 1/8/32/128 with whole passes;
- speedup = DSpark output tok/s ÷ plain output tok/s of the same block, paired and
  validated exactly as in the k=3 campaign; groups are geometric means over their
  slices, with the exact bootstrap over the 4 blocks.

**Decision rule (frozen before results).**
- Headline per load: the k with the highest all-slices geomean speedup; called a
  tie when the runner-up is within 3%.
- Recommend changing the deployed k from 3 only if another k is ≥5% faster on the
  all-slices geomean at C=1, 8 and 32, with its interval clear of k3's, and not
  more than 5% slower at C=128.
- Sanity: the k3 arm must reproduce the 2026-11-05 controlled k3 class speedups
  within ±7%, or the run is not used.

## Restoring the production stack after the scratch purge

The November scratch purge had emptied the pilot directory that the harness
loads (`dev-dspark-12h-20261029-a`), the earlier cluster checkout and the
coverage prompt file. Restored without changing any byte that is measured:
- `experiments/k-headline/restore-pilot.py` rebuilds the pilot from the pilot
  records committed with the k=3 controlled run and the vLLM overlays and chat
  template of the live Deployment's ConfigMap
  (`breithorn-poc/rob-poc@20b21600:swissai-research-platform/models/vllm-apertus-dspark-e10/configmap.yaml`).
  All 4 overlays and the template match the pilot's recorded sha256;
  `environment.toml` differs only in the pilot path and the compile-cache path.
- `coverage.jsonl` was rebuilt with `apertus_bench.coverage_workloads`
  (`origin/workload-coverage` ba1533b0) from the pinned Hugging Face revisions
  in its manifest (job 4004231); it is byte-identical to the frozen file
  (sha256 `ddb921fe…`). The other four sources are the unchanged home-directory
  files, hash-checked by the submitter.
- The cluster checkout is a shallow sparse clone of the pushed branch at
  `/iopsstor/scratch/cscs/faruk_zahiragic/k-headline/code`.

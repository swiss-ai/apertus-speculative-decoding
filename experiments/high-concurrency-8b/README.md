# High-concurrency sweeps, Apertus 1.5 8B (final drafters)

Canvas item "High-concurrency inference: check the load-test setup on one GPU,
multiple GPUs on one node, and disaggregation, and tune top-k". Write-up:
docs/high-concurrency-8b.md. Submit with `serving/node-sweep.sh SPEC`
(SWEEP_PARTITION=normal, SWEEP_TIME per spec header).

Drafters: DSpark OPB thinking-off e10 and thinking-on e10 (Yu, final), EAGLE
3.1 e10 thinking-off (ours, final). Matched mode: thinking-off drafters on
thinking-off prompts, the thinking-on drafter on thinking-on prompts.

Workloads (cluster only; manifests in workloads/8b/):
- opb-heldout.jsonl: 128 prompts of Yu's thinking-off held-out split.
- arena-hard.jsonl: Arena-Hard v0.1, 371 clean + 129 contaminated prompts,
  two cells pooled with `sweep_report --pool`.
- opb-heldout-think.jsonl, arena-hard-think.jsonl: the thinking-on
  counterparts (thinking-on held-out split; answers capped at 2,048 tokens).

Concurrency is per node. `max_num_seqs` is 256 per GPU of an engine (TP2
512, TP4 1024), so every layout can hold the same requests per node.

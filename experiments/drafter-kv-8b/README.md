# Drafter KV sparsity campaign: Apertus 1.5 8B

The [completed study](../../docs/drafter-kv-sparsity.md) stopped after a negative
DSpark thinking-off screen. No additional runs are planned. Results, exact launch
commands, failed attempts and provenance are in
[`results/8b/drafter-kv/kv-sparsity-20261022-a/`](../../results/8b/drafter-kv/kv-sparsity-20261022-a/README.md).

This directory holds the one-off campaign entrypoints, following the layout of
the other stage-specific experiment directories:

- `design.yaml`: fixed design and gates; it is not a submit-ready sweep spec.
- `preflight.sbatch` and `preflight-on-node.sh`: allocated-node input hashing,
  source audit and GPU checks. They preserve their outputs with the allocation.
- `preflight.py`: compute-node source/package/GPU audit.
- `prepare-preflight.py`: generate immutable installed-runtime validation specs
  from an already-hashed input inventory.
- `submit.sh CAMPAIGN ALLOCATION GENERATED_SPEC.yaml`: submit one explicitly
  reserved allocation, using the shared `serving/node-sweep.sh` launcher.

Controller and evidence logic stay in the benchmark package. From the repo root,
CLI help is available without allocating a GPU:

```bash
PYTHONPATH=src python3 -m apertus_bench.drafter_kv_experiment --help
PYTHONPATH=src python3 -m apertus_bench.drafter_kv_evidence --help
PYTHONPATH=src python3 experiments/drafter-kv-8b/prepare-preflight.py --help
python3 experiments/drafter-kv-8b/preflight.py --help
```

The controller reserves all four GPUs of an allocation against the fixed budget,
requires recorded parent gates before later phases, and never advances phases
automatically. Explicitly superseded failed attempts retain their actual cost.
The optional monitor aborts a campaign on failure; a monitor halt is permanent.
Manual replacement reservations are for attempts in a campaign that has not
been halted. Do not use the completed campaign directory for new deployments.

The runtime and overlay are documented in
[`serving/draft_kv/README.md`](../../serving/draft_kv/README.md). Historical result
files keep their original paths and source hashes; use their recorded revision
for exact replay rather than rewriting old provenance after this layout cleanup.

# Bounded recovery of unfinished layouts (2026-10-27)

The original Claude coordinator history is retained in the task handoff. This
recovery handles only missing artifacts, failed TP4 k7 arms and the original
P/D/TP2DP2 final comparisons. See `docs/high-concurrency-8b.md` for the source
failures, candidate selection and limits on performance claims.

`prepare_nixl.py --root FRESH_PREFIX` downloads exactly NIXL 1.5.0 and the
NIXL CUDA13 CPython3.12 aarch64 1.5.0 wheel, verifies PyPI's advertised hashes,
and writes `manifest.json`. It does not modify the old shared prefix and does
not establish native runtime compatibility. `check_dependencies.py` is the
allocated-node native import arm; real P/D startup is checked by subsequent
original P/D arms.

Run the durable coordinator on Clariden with the repository venv and source:

```bash
PYTHONPATH=src ~/venvs/sml-apertus/bin/python -m apertus_bench.layout_recovery controller \
  --root results/8b/layout-recovery-20261027/controller-v1 \
  --site /iopsstor/scratch/cscs/faruk_zahiragic/pydeps/nixl-layout-recovery-20261027-v1/site
```

Use an absolute root in the actual launch. Run under a remote persistent process
with an exclusive controller lock, and preserve its command, PID, log and launch
receipt. It submits at most one bounded Slurm allocation at a time. Every run
has a fresh output/overlay directory. It freezes source identity and generated
specs, stops on failure, and can resume an existing live job without resubmitting.
Inspect an existing failed job before editing state or creating an explicit new
recovery attempt; it never retries failures automatically.

The durable registry is `ROOT/state.json`. Downstream allocation IDs are
`stages[*].job_id`; each entry contains `stage`, `output`, `spec`, spec hash,
Slurm status and an infrastructure/completeness gate. The controller writes
`analysis-STAGE-JOB.json` only after successful measurements. Terminal Slurm
states are not scientific completion. The final state is
`measurements_complete_pending_scientific_review`; scientific promotion is a
separate evidence review, requiring substantial repeatable benefit and the
original protocol gates. Raw artifacts remain at each stage's `output` path.

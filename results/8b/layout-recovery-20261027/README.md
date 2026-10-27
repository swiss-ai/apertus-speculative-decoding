# Layout recovery evidence, 2026-10-27

Recovered four original terminal runs without rerunning any successful arm:
EAGLE k5 TP4 (3737762), the partial EAGLE k7 TP4 (3738048), thinking-on
DP4/TP4 (3721142) and thinking-on P/D (3721143). Original remote raw files
remain untouched. New local compressed representations are byte-for-byte
lossless; `source-collection.json` preserves all original SHA256 values and
stored hashes. The 4.20GB transfer archive was verified locally, extracted
and removed as an intermediate; it is not a source measurement artifact.

`historical-analysis.json` covers 569 cells across eleven original layout runs,
including already-collected DP4/TP4 runs. All569 retained corpus/deployment/engine
provenance and passed observable metric-counter continuity; raw request usage
was present for every successful request and token sums matched every summary.
Two cells from the known unhealthy3737433 plain run fail99% success (C512 and
C1024). Quality checks do not validate its abnormally slow C128 performance,
which is separately excluded as an unhealthy allocation. Earlier cells from a
partial failed arm cannot stand in for missing strata. No independently repeated
deployment confidence interval or per-layout calibrated correctness conclusion
is implied.

`diagnosis.json` gives the original CUDA allocation failures, dependency error,
and matched memory-budget remedy. `nixl-manifest.json` pins both wheel hashes
and every installed file in the campaign-owned prefix, while `nixl-prepare.log`
records installation. File repair has succeeded; native imports and actual
P/D startup still require the new allocated-node stages.

The fresh campaign registry is on Clariden at
`results/8b/layout-recovery-20261027/controller-v1/state.json`. It exposes
top-level `job_ids`, with `stages[*].job_id` and stage output/spec/gates as the
durable detailed records. The parent's local receipt points to its absolute
path. Slurm completion is distinct from scientific completion; the controller
finishes at `measurements_complete_pending_scientific_review`.

See `docs/high-concurrency-8b.md` for the current qualified verdict and
`experiments/high-concurrency-8b/recovery/README.md` for continuation. Only
substantial repeatable benefits merit implementation language.

The running remote source is pinned to
`efad15a10609e615c5c8635be3c243d1ec5b9da7`, with all75 relevant tracked
source hashes verified against the local checkout before submission. Do not
sync newer highconc result commits or integrate another branch into this remote
checkout while the controller is active. A HEAD or source change deliberately
stops subsequent submissions. Local artifact commits are independent of the
running remote snapshot. Launch receipt, command and dependency identities
are preserved here and at the remote campaign root.

The coordinator venv lacked `prometheus_client`; the initial source-identity
import check failed before any allocation was submitted. This was fixed by
installing exactly `prometheus-client==0.22.1` into the fresh campaign-owned
`pydeps/layout-controller-20261027-v1/site` prefix. The shared venv was not
modified. The pip report/log and source-identity environment manifest retain
this dependency and the existing PyYAML6.0.3 identity.

Initial recovery allocation: preflight **3769354**, launched by durable remote
controller PID69249 on clariden1. Its detailed initial state is retained in
`controller-first-observation.json`. This is an infrastructure preflight, not
a final speedup verdict. The controller will register all further job IDs at
the same durable state path. The local parent monitor reads that registry.
The final pre-launch source check is explicitly suffixed `efad15a1`; earlier
unsuffixed source-check artifacts record a superseded import check.

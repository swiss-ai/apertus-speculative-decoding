# Final-checkpoint 8B serving-layout campaign

Status on 2026-10-27: original finished arms are recovered; affected TP4 and P/D
arms need fresh measurements. The campaign is **not scientifically complete**.
The bounded recovery controller stops on failed infrastructure/request/health
gates and records measurement completion separately from scientific review.

## What the completed runs establish

These are deployment-level screening measurements, generally one launch per
setting. Request counts are not independent deployment repeats. A gain below
10% is not a reason to implement another serving configuration; even a larger
screening gain needs matched confirmation and operational checks. The existing
protocol's calibrated correctness controls have not been run for every layout,
so throughput results do not establish output equivalence.

- DP4 is the current strong high-load candidate. At concurrency 1024,
  thinking-off DSpark k3 produced 29,598 tok/s on held-out OPB and
  37,830–38,105 tok/s on Arena-Hard, versus 25,236 and 29,055–30,632 tok/s
  for its same-job plain baseline. EAGLE k3 produced 29,843 and
  37,388–37,772 tok/s, versus 25,279 and 28,768–30,731 tok/s. These
  are substantial descriptive gains, not yet a repeated deployment estimate.
- Recovered EAGLE k5 TP4 (3737762) is load-sensitive: at concurrency 1024,
  21,819 tok/s OPB and 24,520–25,419 tok/s Arena-Hard, versus
  12,452 and 13,620–13,962 tok/s plain (1.75–1.87x). At concurrency
  128 it gives 13,471 tok/s OPB versus 14,118 plain, and only
  11,263–12,229 tok/s Arena-Hard versus 17,390–18,559 plain. The
  34–35% Arena regressions rule out enabling this setting at every load.
  These TP4 absolute high-load rates are also below DP4's rates, measured
  in different allocations; a large ratio against plain TP4 is not a reason
  to prefer TP4 over DP4.
- Thinking-on DP4 DSpark k3 produced 45,841 tok/s OPB and
  51,517–51,939 tok/s Arena-Hard at concurrency 1024, versus 31,980
  and 34,158–34,860 tok/s plain. k7 was slower there (43,163 and
  45,622–46,231 tok/s). The original TP4 k7 arm crashed; it is not a
  completed high-load comparison.
- No new P/D verdict exists: original thinking-on plain DP4 completed, but
  both P/D arms failed before they could serve. A dependency failure is not
  evidence of a disaggregation performance loss or gain.

Numbers above are read from immutable `summary.json` cells beneath
`results/8b/sweeps/`. Arena clean and contaminated strata are shown as ranges,
not merged with an unweighted mean. Their labels and raw rows remain separate.
Derived recovery analysis retains source-file hashes and quality checks.

## Recovered failures and the repair

1. EAGLE k7 TP4 (3738048) ran out of CUDA memory: a 5.52 GiB allocation
   could not fit in 4.23 GiB free; PyTorch reported 9.16 GiB reserved but
   unallocated. The engine-dead messages were downstream symptoms.
2. Thinking-on DSpark k7 TP4 (3721142) likewise failed a 7.40 GiB
   allocation with 7.19 GiB free and 8.23 GiB reserved but unallocated.
3. DSpark-off k7 TP4 (3737433) never measured its speculative arm. Its plain
   baseline slowed to 2,017 tok/s at C128 and 839 tok/s at C1024, with
   only 74.9% successful C1024 requests, before the job timed out. Its
   ratios cannot be used. The old intended retry was never submitted.
4. Thinking-on P/D (3721143) used an incomplete NIXL 1.5.0 installation:
   `nixl_meta_utils.py`, `nixl/__init__.py` and `nixl/_api.py` were absent.
   The connector therefore reported NIXL unavailable. The repair installs
   the original exact NIXL and CUDA13 aarch64 CPython3.12 wheel versions
   in a fresh campaign-owned prefix, with wheel and installed-file hashes;
   it does not alter the shared installation.

TP4 k7 retries use `gpu_memory_utilization=0.7` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on **both** the plain
and speculative arms. The original budget was 0.8. That reserves about
9.5 GiB more headroom per 95 GiB GPU and reduces available KV capacity;
actual advertised KV-cache tokens, preemptions and latency must be compared.
This is a deployment tradeoff, not an identical-capacity rerun. Context length
32768, sequence capacity 1024, and batched tokens 16384 are retained.
A short allocated-node C1024 stress/import preflight precedes the full retries;
its short-output timing is not a final workload performance result.

## Bounded remaining comparisons and k selection

Recover all original DP4/TP4 finished arms rather than rerunning them. After
the affected TP4 arms, run only the original `tp2dp2.yaml`, `think-pd.yaml`
and `pd.yaml` workloads, models and algorithms. Retain the original TP2DP2
k3/k5 finalists: DP4 supports k3 near saturation, while k5 remains competitive
at lower loads and TP4 changes the crossover. There is no consistent reason
to add k7 to another grid. P/D retains the original k3 setting for its
high-load comparison. Small k3/k5 differences select candidates for this
bounded study; they do not create implementation recommendations.

Each original failed attempt remains intact. `source-collection.json` in
`results/8b/layout-recovery-20261027/` records all 1,420 newly recovered
files, original hashes and lossless stored representations. Fresh runs retain
normal node-sweep cell metadata, requests, metrics, environment, container,
patches, drafters and Slurm provenance. New run directories never overwrite
an original cell. A final review must show absolute throughput/latency and
KV capacity, rejected cells, losing strata, repeat uncertainty and operational
cost before recommending deployment.

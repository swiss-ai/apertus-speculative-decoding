# Sampling recovery, 2026-10-27

Read [the verdict](../../../../docs/sampling-final-8b.md). This recovery closes two
already successful sampling sweeps; it submits no jobs and changes no serving code.

Canonical original records:

- [topk-off-20261022T122308Z](../../sweeps/topk-off-20261022T122308Z/): job 3721144,
  180 cells, 94,050 successful requests, 50,304,357 output tokens.
- [topk-think-20261022T122311Z](../../sweeps/topk-think-20261022T122311Z/): job 3721145,
  144 cells, 75,240 successful requests, 98,833,025 output tokens.

The original trees contain `metadata.json`, `summary.json`, `requests.jsonl.gz`,
before/after Prometheus snapshots, load snapshots, client and server logs, GPU
memory samples, specs, launch argv/status, job/node receipts, Git revision, patch
list, checkout diff, resolved environment and container digest. JSONL records
contain timing/event gaps, token counts and output hashes, with `output: null`;
source prompt text and model weights are not included. Large text files use
lossless deterministic gzip; decompressing restores the exact original bytes.
No original metadata, cell summaries, historical identities or comments were edited.

[collection-manifest.json](source-recovery/collection-manifest.json) records every
source file's remote path, byte count and SHA256, plus separately labelled
recovery-time patch/overlay/target references and launch-revision committed source.
The 2,213 original run files total 1,560,953,586 bytes. Including the initial
recovery references gives 2,245 files / 1,561,563,137 bytes. The
[collection verification](collection/collection-verification.json) records the
transport archive hash and the 335 lossless gzip mappings; the redundant transport
archive was removed after byte verification.

Source checkout:
`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-spec-decoding-highconc`.
Recovery collection read-only HEAD was `b40e8571ec6f9417075a6658eaf3df32443122b0`.
This is a recovery reference, not the launch revision. Both original launch
receipts say `a0597294cbe829aa8d6a1e577c462571df8bf572`; both original image receipts
say `f5fc017ff794fa8d80b73dd9a788028a1872dbfc2e21c0de043fb7738c11a0b5`
for `vllm_apertus_1.5_release-arm64.sqsh`.

Recorded checkpoint identities (`checkpoint_sha256` is the launcher's combined
config-and-weights manifest digest, not just the safetensors digest):

| Head | Recorded manifest SHA256 |
|---|---|
| DSpark final off, 2026-10-04 epoch10 | `6b11d196606b1fad792247a2bcd9fe0255ea8bed2bb5463fd88a09092a394315` |
| DSpark final on, 2026-10-10 epoch10 | `1391857cf538f831693f31853b11ef42ac8c4248f7325f3ee251377aad6c4572` |
| EAGLE3.1 `e31-opb-thinkoff-stage1-se` | `d825649bc24cb6f028da2dd96aeff06180779dcdd79702682333af9c076c400a` |

[Checkpoint support](source-recovery/checkpoint-support/recovery-manifest.json)
contains recovery-time copies of the small head configs, EAGLE export report,
and DSpark training command/config/patch/validation files with their source paths
and hashes. Weights were neither copied nor rehashed during recovery. EAGLE's
export report agrees with the originally recorded manifest identity. These copies
supplement the original run, without pretending they were co-located at launch.
The target contract records revision `a411d838600baf0e3635a3daf66fb7c55fc97bb6`, target
weight manifest `0a6c3ca1c1e8cf220c21971b3b96b0e8799827f7bb00ac056dfb3657a533b08a`.
EAGLE cells retain target model/revision/TP; plain and DSpark do not have all of
those fields, so their missing original fields remain missing.

The four [sanitized prompt schedules](source-recovery/prompt-schedules/) retain
only id, workload, output cap and explicit chat-template options. Every full
source file digest matches its original metadata digest. Arena's clean subset
has 371 of 500 source prompts; OPB has 128. Off OPB caps vary 16–1,024 and thinking
OPB 56–2,048. Arena caps are 1,024 off / 2,048 on. At C256 the client repeats
OPB eight times (1,024 requests) and Arena three times (1,113 requests). These are
repeated workload rows within a deployment, not repeated independent deployments.
No restricted prompt text is redistributed.

There are two important unrepaired protocol discrepancies. The original harness
Git metadata is stale because its lookup uses the container's older working
directory; the launch-revision `node_sweep.py` prepends its own source to the client
`PYTHONPATH`. Original comments claim no seed, but all metadata has seed base 1 and
the retained launch client/runner send `1 + request_index` in the HTTP payload.
There is no wire capture. The report distinguishes the source/metadata evidence
from the comments and does not generalize to unseeded sampler throughput.

Reproduce the analysis locally from the repo root, choosing fresh output paths:

```bash
PYTHONPATH=src python results/8b/sampling/recovery-20261027/analyze.py \
  --output results/8b/sampling/recovery-20261027/analysis-v3
python results/8b/sampling/recovery-20261027/views.py \
  --analysis results/8b/sampling/recovery-20261027/analysis-v3 \
  --output results/8b/sampling/recovery-20261027/verdict-v2
```

`analyze.py` verifies source hashes, recomputes counts/tokens and conventional
latency quantiles from every raw request, checks event-gap counts/means, verifies
TPOT arithmetic and cap/truncation consistency, and recomputes speculative counter
deltas. Pairing requires exactly matching generation settings, corpus hash,
prompt IDs, request index/prompt schedule and engine settings. Output tables expose
TPOT and multi-token stream gaps separately, as well as output length/caps,
truncation, GPU-seconds/M, acceptance and preemptions. Original event-gap quantiles
are retained from the summaries; their millions of event samples are not re-sorted.

[analysis-v1](analysis-v1/) was the first completed validation;
[analysis-v2](analysis-v2/) adds explicit arm identity checks and source provenance,
with identical numerical cell/comparison values. Both are retained. Final report
tables use `analysis-v2` and [verdict-v1](verdict-v1/). Every output refuses to
overwrite an existing output directory. `views.py` additionally compares top-k
restrictions only at T1/top-p1 with the same arm/mode/workload/load; these comparisons
change the sampling distribution and carry no quality claim.

`collect_remote.py` is the read-only original-artifact collector. It streams an
uncompressed tar over SSH; local gzip avoids compression work on the login node.
Collection did not sync or mutate the remote checkout. The sampling jobs were not
rerun. `SHA256SUMS` covers the recovered and analysis files as stored locally;
the original-byte hashes in the manifests remain authoritative after decompression.

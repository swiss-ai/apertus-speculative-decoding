# DSpark e10 k=3: what the controlled test establishes

All 6,336 measured requests succeeded across 90 cells and three paired fresh
deployment repeats. Both configurations ran sequentially on the same physical
GH200 GPU, through loopback, with identical target, image, overlays, template,
generation settings and prompt order. No new Slurm allocation or training was
submitted. The existing Model Launch serving worker remained on GPU 0; the test
used idle GPU 1 of job 3791982, node nid007257.

The primary result uses natural EOS with a 256-token cap. Throughput speedup is
DSpark output tokens/s divided by baseline output tokens/s. Brackets are 95%
percentile bootstrap intervals over three deployment pairs on this one node.

| Workload | C=1 | C=8 | C=128 burst |
| --- | --- | --- | --- |
| Held-out OpenPerfectBlend (ID) | 2.19x [1.85, 2.49] | 2.08x [1.98, 2.15] | 1.14x [1.12, 1.17] |
| OOD chat | 1.74x [1.73, 1.74] | 1.50x [1.49, 1.50] | 0.92x [0.92, 0.93] |
| OOD long-document summarization | 1.43x [1.42, 1.45] | 1.30x [1.29, 1.31] | 1.00x [0.99, 1.01] |

This establishes a gain at low concurrency in all three tested strata. Under the
tested high-concurrency bursts, OOD chat loses about 7.7% throughput, and OOD
summarization stays within the declared +/-5% practical-equivalence band. ID
still benefits at that load. Enabling the drafter for all workloads and loads is
not supported by these results.

The separate exact-256-token check confirms low-load gains: ID 2.47x, OOD chat
1.84x and OOD summarization 1.51x. At C=128 it measures 1.18x, 0.95x and 0.98x,
respectively. The OOD chat interval there straddles the declared 0.95 cutoff, so
the report calls a greater-than-5% regression inconclusive, while its interval
is entirely below 1. The fixed-length and natural-EOS cells are kept separate.

## The retained ID latency spike

The third natural-EOS ID C=1 cell contains one request with 8,486 ms TTFT and
9,130 ms total latency (`opb-opb-0036187`, request index 10). The engine logs
Triton JIT compilation of `_prepare_dflash_inputs_kernel` during that cell,
consistent with this spike. Warmup did not cover every shape. That request is
retained in the headline and raw data; no favorable rerun replaced it.

The three ID natural-EOS throughput ratios are 2.50x, 2.45x and 1.71x, giving
the headline 2.19x and its wider interval. Median ID total response time remains
about 1.50 s baseline versus 0.55 s DSpark in every repeat. The three fixed-length
ID C=1 ratios are 2.48x, 2.45x and 2.47x. Future serving warmup should exercise
the actual prompt shapes before advertising stable tail latency.

## Memory and scope

The engine reports 469,840 cache tokens for baseline and 342,789 with DSpark:
about 27% less token capacity at the same 80% GPU memory budget. The result
includes the drafter's memory and engine overhead; acceptance alone would not
capture this cost. Mean natural-EOS draft-token acceptance is approximately 79%
on ID, 44% on OOD chat and 32% on OOD summarization.

Each stratum has 32 deterministically selected prompts. ID comes from the
existing DSpark OpenPerfectBlend validation workload. OOD labels denote shifts
relative to that mixture; they do not guarantee every individual topic was
absent from training. C=128 sends a single burst of 128 requests, covering four
corpus passes, rather than measuring sustained production capacity. Three
repeats on one node do not quantify variation across nodes, prompt populations
or gateways. No losslessness or answer-quality conclusion follows from this
performance experiment. The original illustrative replies and quality review
remain in the separate usage-demo experiment.

## Artifacts

[REPORT.md](REPORT.md) contains all 15 conditions and every deployment-pair ratio.
[paired-results.json](paired-results.json) retains absolute throughput, latency,
acceptance, output lengths and load metrics. Each raw cell retains metadata,
request measurements and metric snapshots. The actual restricted corpus stays
on Clariden; its manifest, selection IDs and digest are preserved here.

[runtime-recheck.json](runtime-recheck.json) confirms the image, target assets
and head hashes after measurement. The collected archive's 655 original files
passed the artifact manifest verification. This review and README are derived
documents added after collection; original raw bytes and their manifest remain
unchanged. The run's source identity is
`f586455fbad788f94e60ef5ba74892f112439fd0`; later report/finalizer sources are also
included. Launch commands and logs are under [launch/](launch/).

## Serving discovery remains a separate deployment issue

The existing worker used Model Launch's dev OpenTela wrapper. Its configured
bootstrap peer ID is stale. The live server advertises a different identity, and
the handshake rejects the mismatch before model discovery. Bounded probes
reproduced failure with the stale ID and successful connection plus relay
reservation with the advertised ID. The pilot now exposes Model Launch's native
`--opentela-bootstrap-addr` override and preserves it in launch provenance.

That handshake fix is verified; the original worker was not relaunched during
this campaign and remains a dev deployment. Actual gateway serving must still
be verified after a launch using the corrected bootstrap. See the
[execution and discovery notes](../../../../experiments/serving-pilot/SPEED-ID-OOD.md).

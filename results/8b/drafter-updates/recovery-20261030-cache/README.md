# T5 bounded EAGLE adaptation: final verdict

**Keep workload-specific adaptation in mind for persistent multilingual traffic and low-load summarization. Do not implement chat adaptation from these results. No blanket replacement of the general drafter.**

The original 19 final exported heads completed 384 steps each. The preregistered development-only rule selected one 8k/no-replay head per domain. All three selected heads completed both same-node GPU-swapped serving rounds. No training, sampling, correctness or promotion gate was relaxed. This is Apertus 1.5 8B, thinking off, target TP1, fixed T=0.7/top-p=0.8/sampling-top-k=20, natural EOS within the original completion caps.

The clearest competitive matched configuration is draft k3/C1 for both useful domains:

| Domain | Throughput, tok/s, base→adapted | Median decode, ms/token | p95 request, seconds | Implementation verdict |
| --- | --- | --- | --- | --- |
| Multilingual | 267→312–313 (+17% both swaps) | 3.41→2.78–2.79 | 1.24→1.11–1.15 | Substantial specialist benefit |
| Summarization | 221→251 (+14% both swaps) | 4.13→3.55 | 1.58–1.59→1.414 | Useful at C1; C64/C256 fail the full benefit rule |
| Chat | All six settings gain only about 3–8% throughput | About 0–6% change; not consistently enough | No qualifying repeated latency/throughput combination | Do not implement from this experiment |

The fixed rule requires at least 10% throughput **and** 10% lower median decode latency in **both** swapped repeats, successful requests/warmups and calibrated correctness. Multilingual meets that matched-k rule in all six primary k3/k7 × C1/64/256 cells. Under load, the competitive k3 choice gains roughly 11–15% throughput, but can add 7–14 ms median TTFT; retain that conditional tradeoff. Summarization qualifies only at C1; its high-load gains are insufficient. Chat qualifies nowhere, despite its positive development-acceptance result.

Larger matched-k percentages need context. Multilingual k7/C1 reaches 325–329 tok/s, a 26–30% gain over its weaker k7 baseline, with 2.38–2.41 ms/token. Against the stronger unadapted k3 setting, its gain is about 22%, with roughly 4 ms extra TTFT and slightly higher p95 request time. At C64/256, unadapted k3 already outperforms adapted k7. For summarization, adapted k3 has higher throughput and lower median decode latency than adapted k7 even at C1. Recommendations therefore use competitive operating points rather than every individually improved setting. The [supplementary multilingual configuration review](multilingual-configuration-review.json) preserves the cross-k comparisons; these use separate GPU pairs and supplement the original matched-k tests.

OPB forgetting remains material to the decision: development accepted length falls 4.92% for multilingual, 2.48% for summarization and 1.93% for chat. Measured OPB/tail/TTFT regressions also occur in the serving grid. A domain-specific deployment can benefit substantially; these results do not justify replacing the default head for a general traffic mix.

**Absolute compute and amortization.**

| Domain | Selected training-loop wall, 4 GPUs | Training-only lower bound | All recovery traces + all curves of this domain | Conservative amortization, matched k3/C1 |
| --- | ---: | ---: | ---: | ---: |
| mult | 29.7 min | 1.98 GPUh | 15.77 GPUh | 104.4M output tokens |
| summ | 40.8 min | 2.72 GPUh | 23.93 GPUh | 159.5M output tokens |
| chat | 19.2 min | 1.28 GPUh | 13.26 GPUh | No qualifying benefit |

The full EAGLE recovery, including all curves, failed recovery attempts, trace generation and all serving confirmations, allocated **55.47 GPUh**. Domain setup in the table charges all recovery traces plus every curve of that domain; it excludes other domains and the serving-measurement campaign. At matched k3/C1, multilingual saves 544–546 GPU-seconds per million output tokens and summarization 540–547. Selected training alone amortizes after about 13M/18M tokens, before capture/startup/export; charging the entire recovery instead gives 367M/370M tokens. These are experimental recovery/search costs and lower bounds, **not a measured recurring production-update cost**. Accounting covers the 11 recovery jobs; original pre-recovery smoke/live-swap jobs are excluded. Production traffic volume, update cadence, engineering cost and monetary tariffs were not measured.

**Every primary serving cell.** A/B are the two physical-GPU-swapped deployment repeats. Complete absolute metrics, acceptance, cost and regressions for all 84 conditions are in [verdict.json](verdict.json) and the [168-comparison CSV](serving-absolute-cells.csv).

| Head | Draft k | C | tok/s base A/B→adapt A/B | TPOT ms base A/B→adapt A/B | TTFT p50 delta A/B, ms | Request p95 base A/B→adapt A/B, s | Original matched-k rule |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| mult | 3 | 1 | 267.0/267.5→312.4/313.1 | 3.415/3.411→2.784/2.791 | +0.6/+0.3 | 1.243/1.243→1.108/1.149 | yes |
| mult | 3 | 64 | 6864.5/6959.7→7717.7/7708.1 | 7.540/7.507→6.520/6.566 | +8.1/+7.4 | 2.843/2.821→2.633/2.657 | conditional |
| mult | 3 | 256 | 8644.4/8827.1→9935.4/9754.2 | 25.325/24.726→20.959/21.354 | +2.2/+14.3 | 9.811/9.580→8.709/8.993 | conditional |
| mult | 7 | 1 | 254.1/258.8→329.4/325.3 | 3.546/3.485→2.375/2.408 | +1.3/+0.4 | 1.533/1.411→1.280/1.296 | yes |
| mult | 7 | 64 | 4781.7/4936.8→6093.3/5854.7 | 10.833/10.446→7.722/8.057 | +1.2/-9.8 | 4.587/4.459→3.826/3.937 | yes |
| mult | 7 | 256 | 5921.3/6042.6→7442.1/7270.9 | 36.331/35.609→26.312/26.729 | +9.0/+22.0 | 15.800/15.311→13.573/13.830 | conditional |
| summ | 3 | 1 | 221.4/221.1→251.4/251.5 | 4.138/4.123→3.550/3.548 | +3.0/-2.5 | 1.582/1.591→1.414/1.414 | yes |
| summ | 3 | 64 | 1800.9/1802.9→1896.2/1876.6 | 31.958/31.396→31.026/30.774 | +12.8/-19.6 | 12.097/12.123→11.871/11.570 | no |
| summ | 3 | 256 | 1919.8/1905.0→1976.2/1991.1 | 75.728/75.847→72.310/72.099 | -215.5/-394.0 | 43.324/43.457→41.936/41.588 | no |
| summ | 7 | 1 | 195.9/195.5→239.3/239.5 | 4.597/4.603→3.732/3.731 | +2.1/-2.9 | 1.804/1.807→1.381/1.383 | conditional |
| summ | 7 | 64 | 1550.5/1522.5→1662.0/1710.7 | 36.292/37.160→34.490/33.847 | +32.1/-34.2 | 13.821/14.332→13.604/13.386 | no |
| summ | 7 | 256 | 1665.1/1636.3→1745.5/1786.3 | 85.082/86.897→81.165/79.160 | -527.6/-1150.9 | 50.302/51.232→46.856/45.897 | no |
| chat | 3 | 1 | 253.6/254.7→266.6/265.9 | 3.312/3.294→3.249/3.298 | +4.1/-2.5 | 1.339/1.335→1.244/1.245 | no |
| chat | 3 | 64 | 4296.8/4281.0→4525.1/4536.0 | 11.386/11.565→11.329/11.198 | +5.7/+6.5 | 4.791/4.757→4.526/4.533 | no |
| chat | 3 | 256 | 5077.5/5052.8→5233.0/5228.3 | 41.382/41.398→41.339/41.077 | +16.8/+11.6 | 17.720/17.797→16.460/16.520 | no |
| chat | 7 | 1 | 235.5/236.1→253.3/252.7 | 3.517/3.499→3.320/3.334 | -1.3/-1.6 | 1.575/1.571→1.476/1.479 | no |
| chat | 7 | 64 | 3300.0/3338.4→3575.9/3508.9 | 14.456/14.351→13.862/14.215 | +0.5/+6.3 | 6.685/6.580→6.151/6.263 | no |
| chat | 7 | 256 | 3908.2/3959.8→4175.7/4114.1 | 52.409/51.455→49.784/50.562 | -3.4/+8.0 | 24.104/23.802→21.545/21.971 | no |

**All 19 training curves.** All original scientific gates remain recorded; the original held-out-improvement gate uses the best evaluation. The 500-trace chat arms finish below their starting acceptance despite an earlier improvement. Only final heads were eligible for the fixed serving selection. Replay generally reduces OPB forgetting but was not separately serving-confirmed here; these alternatives receive no implementation recommendation.

| Original arm | Final development accepted-length gain | OPB change | Reported training wall, min |
| --- | ---: | ---: | ---: |
| adapt-chat-n2k-r0 | +0.1087 | -3.35% | 19.28 |
| adapt-chat-n2k-r20 | +0.1305 | -2.50% | 17.85 |
| adapt-chat-n500-r0 | -0.0137 | -4.41% | 18.73 |
| adapt-chat-n500-r20 | -0.0106 | -3.79% | 18.08 |
| adapt-chat-n8k-r0 | +0.2231 | -1.93% | 19.22 |
| adapt-chat-n8k-r20 | +0.2134 | -1.30% | 18.29 |
| adapt-mult-n2k-r0 | +0.5708 | -6.05% | 17.49 |
| adapt-mult-n2k-r20 | +0.5904 | -3.55% | 20.24 |
| adapt-mult-n500-r0 | +0.3244 | -5.95% | 23.08 |
| adapt-mult-n500-r20 | +0.3175 | -4.72% | 31.21 |
| adapt-mult-n8k-r0 | +0.7568 | -4.92% | 29.71 |
| adapt-mult-n8k-r20 | +0.7232 | -2.35% | 35.18 |
| adapt-summ-n2k-r0 | +0.3806 | -3.73% | 38.40 |
| adapt-summ-n2k-r0-lr1e4 | +0.4278 | -8.19% | 38.93 |
| adapt-summ-n2k-r20 | +0.3841 | -2.85% | 38.04 |
| adapt-summ-n500-r0 | +0.2070 | -4.03% | 42.42 |
| adapt-summ-n500-r20 | +0.2091 | -3.27% | 38.11 |
| adapt-summ-n8k-r0 | +0.5169 | -2.48% | 40.82 |
| adapt-summ-n8k-r20 | +0.4979 | -1.68% | 38.61 |

![Development curves; simulation only](development-curves.png)

[All development points](development-curves-derived.json) retain the exact curves. Simulation was used only for development selection; it is not serving acceptance or speed.

**Evidence and reproducibility.** All 336 raw cells, 153,600 measured requests and 56 calibrated correctness gates passed; failed warmups=0. All member hashes, target/checkpoint/source identities, corpus digests and complete prompt passes were independently audited. The controls do not prove losslessness or full-answer quality. These are seeded fixed-concurrency benchmarks on stand-in domain traces, with the original completion caps; they do not establish production capacity or a universal update policy.

- [Training capsule](training-curves-complete-20261031b.tgz): 208 files, configs/provenance/train summaries/full histories/evals/export reports/raw logs; [manifest](collection-manifest.json).
- [Multilingual raw capsule](confirmation-3814370-capsule/capsule-manifest.json): 1,785 files; [audit](confirmation-3814370-capsule/integrity-audit.json); [paired summary](multilingual-confirmation-summary.json).
- [Summarization raw capsule](confirmation-3814755-capsule/capsule-manifest.json): 410 files, including exact scheduling override; [audit](confirmation-3814755-capsule/integrity-audit.json); [paired summary](summarization-confirmation-summary.json).
- [Chat raw capsule](confirmation-3815183-capsule/capsule-manifest.json): 409 files; [audit](confirmation-3815183-capsule/integrity-audit.json); [paired summary](chat-confirmation-summary.json).
- [Final 29-file closure/provenance capsule](final-eagle-recovery-20261031.tgz), SHA256 d68c7818f14b33807cda0fe6b1dac0a03becf90e98a9171731c5cb45eab4f81f; [manifest](final-artifact-manifest.json), [final controller receipt](campaign-final.json), [Slurm allocations](allocation-accounting.txt), [as-run source manifest](serving-source-manifest.json), [unchanged policy](preregistration.json).
- [Preserved 3814329 startup failure](failed-confirmation-3814329.json) and raw capsule; earlier trace/data/quota failures and the original four-step smoke scientific failure remain unchanged in their associated recovery trees.

Raw parts are at most 25 MiB each; reconstruct with the command in each capsule manifest, then verify with `python3 verify-capsule.py confirmation-JOB-capsule`. Caches, weights and restricted source corpora remain on the cluster. Training source 589ddd593f62255f9217c1b40e5502808cc7bb94; serving/controller source 7f0a41dcc0a3c5d433b83dac80075ee0516a694e. Scientific policy SHA256 64c9844f6cd526e6c693f85a5f75cc0d5d7d3f605112b14c3ae1c70a969ef44c. Remote complete root: `/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/drafter-updates/recovery-20261030-cache`.

**Scope completed:** original bounded EAGLE recovery, including all 19 curves and three serving confirmations. Existing same-architecture live swap was already complete and was not rerun. The substantial EAGLE result triggers only the original conditional DSpark best-setting dependency: verify compatible data/port/provenance before that repeat. No DSpark adaptation run or result, routing implementation or deployment is claimed. Broad canvas #10/A3 remains open.

[Final Slack summary](https://swissai-initiative.slack.com/archives/C0C2QMYCQ5B/p1793467082269699) and [updated action canvas](https://swissai-initiative.slack.com/docs/T0627JS49GT/F0C6431UBTP). The [publication receipt](slack-publication.json) retains exact text and links, verification that unrelated canvas sections were preserved, and the unchanged open #10/A3 state.

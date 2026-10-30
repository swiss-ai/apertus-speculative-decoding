# Shallow-drafter validation diagnosis, 30 October 2026

Training **3790212 completed all 6,200 steps** in 2 h 25 min. Its held-out
trainer EAL is **3.51992526**, versus **4.63015594** for the completed five-layer
control. The one-layer held-out `accept_len_epoch` is 3.35215246. These metrics
are not serving acceptance or speed. Both checkpoints are preserved; neither
training run is repeated.

Validation **3790216** passed training/recipe/source/target/checkpoint preflight
and completed every serving deployment and capture. It then failed the original
calibrated greedy correctness gate. All eight trained/plain comparisons fail
minimum prefix fraction and maximum normalized edit distance:

| first comparison | measured prefix / required minimum | measured edit / allowed maximum | token difference / allowance |
|---|---|---|---|
| one-layer k7 | 0.18323 / 0.34742 | 0.51405 / 0.06429 | 0 / 2 |
| five-layer k7 | 0.18323 / 0.34792 | 0.50627 / 0.46545 | 2 / 4 |

The independently restarted plain controls match all six prompts exactly and
pass their calibrated self-checks. Target, tokenizer, template and target weight
hashes match the contract; supplied engine flags and request sampling match
apart from the intended speculative configuration. No payload or model-identity
defect has been established. The current result is a **correctness block**, not
a launcher failure or a measured throughput loss. Benchmark jobs remain gated.

One confirmed harness gap remains: both validation repeats assigned plain/L1/L5
to GPUs 0/1/2. They measured restart variation without counterbalancing physical
GPUs. A single bounded diagnostic repairs that protocol gap and adds existing
DSpark e10 k7 to identify a common serving-path problem. It does not assume that
GPU placement caused the failure. New placement is:

| arm | repeat A GPU | repeat B GPU |
|---|---:|---:|
| plain | 0 | 3 |
| existing DSpark e10 k7 | 1 | 2 |
| trained one-layer k7 | 2 | 1 |
| trained five-layer k7 | 3 | 0 |

The diagnostic retains the same six held-out prompts, 384-token capture cap,
original image digest, engine flags, checkpoints and margins (prefix/edit 0.05,
completion tokens 2). Each trained model is calibrated only against plain and
its own identical configuration; e10 controls are never pooled into another
model's envelope. Every original failed comparison remains preserved. A
repeated correctness failure stops the chain, without relaxing limits or
starting throughput work. No eager-mode, precision or training changes are made.

The launcher error trap now preserves the actual correctness failure instead
of replacing it with a generic host error. Completed training has aged out of
Slurm's 300-second live-job retention: the continuation explicitly checks its
COMPLETED/0:0 accounting row, step 6200, one-layer config, held-out validation and
passed identity report before submission; full hashes are rechecked on compute.
Only the new validation and, conditionally, the original a/b/strict-review
stages are submitted. The original G3 and strict fixed-k >=10% implementation
screens remain unchanged. 95 focused tests, Ruff and shell syntax checks pass.

`training-and-failed-validation.tar.gz` preserves 82 compact copied artifacts
(compressed training/extraction logs, complete metadata, failure reports,
provenance and validation logs); 12 raw captures remain on the cluster with
source hashes indexed. All 82 copies were verified, including decompression.
Archive SHA-256:
`b1655fd4339f2110f63d18b7654beb311c44dea9707caf9d626c888ae6fe0770`.
The trained one-layer weight SHA-256 is
`9d6f8f8df6dbc95716c39aa5a6fb13e6872604aa08bd942e6c4422315cc01cba`.
The preserved image digest is
`f5fc017ff794fa8d80b73dd9a788028a1872dbfc2e21c0de043fb7738c11a0b5`.

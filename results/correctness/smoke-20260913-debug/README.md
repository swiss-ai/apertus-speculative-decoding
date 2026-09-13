# Greedy correctness, 2026-09-13, `debug` partition

Sequential greedy captures for the baseline, both `draft-n3-tp4` deployment repeats and the
`ngram-n3` deployment of `../../smoke-screening-20260913-debug/`, plus two generations of comparison
over them: the original exact-match comparison, and the re-analysis under the calibrated criterion
that replaced it
(`docs/protocol.md`, section "Greedy correctness gate"). Nothing here was overwritten; the
exact-match artifacts are exactly as first recorded.

## Captures

| Capture | Deployment | Job | Node |
|---|---|---|---|
| `baseline.json` | `baseline` | 3391425 | nid007645 |
| `draft-n3-tp4.json` | `draft-n3-tp4` repeat 1 | 3391426 | nid006633 |
| `draft-n3-tp4-repeat2.json` | `draft-n3-tp4` repeat 2 | 3392153 | nid006687 |
| `ngram-n3.json` | `ngram-n3` | 3392370 | nid006687 |

Six smoke prompts, temperature 0.0, top_p 1.0, per-request seed `1 + request_index`, natural EOS,
`max_model_len=131072`, one request at a time. Every request succeeded in all four captures, so all
six prompts are comparable in every pair. Each capture keeps the full output text, so any comparison
criterion can be recomputed from these files without re-running the deployments.

## Exact-match comparisons, as first recorded

`apertus-bench compare-captures` at harness revision `270413f2`, schema version 1:

| Comparison | Artifact | Exact matches |
|---|---|---:|
| baseline vs. draft repeat 1 | `comparison.json` | 0 of 6 |
| baseline vs. draft repeat 2 | `comparison-repeat2.json` | 3 of 6 |
| draft repeat 1 vs. draft repeat 2 | `comparison-draft-repeat1-vs-repeat2.json` | 0 of 6 |
| baseline vs. n-gram | `comparison-ngram-n3.json` | 2 of 6 |

The third row is the one that matters. Those two deployments ran the *same* configuration, so they
must agree if exact agreement is achievable at all; they agree on nothing. Exact greedy equality is
therefore not a property of this stack across independently launched deployments, and the 0 of 6 in
the first row is not evidence that speculation is lossy.

## Calibrated re-analysis

`apertus-bench compare-captures` (schema version 2, same exact-match numbers plus divergence
statistics) and `apertus-bench correctness-gate`:

| Artifact | Pair | Role |
|---|---|---|
| `divergence-baseline-vs-draft-repeat1.json` | baseline vs. draft repeat 1 | compared pair |
| `divergence-baseline-vs-draft-repeat2.json` | baseline vs. draft repeat 2 | compared pair |
| `divergence-draft-repeat1-vs-repeat2.json` | draft repeat 1 vs. repeat 2 | calibration control |
| `divergence-baseline-vs-ngram.json` | baseline vs. n-gram | compared pair, uncalibrated |
| `gate-baseline-vs-draft-repeat1.json` | baseline vs. draft repeat 1 | verdict |
| `gate-baseline-vs-draft-repeat2.json` | baseline vs. draft repeat 2 | verdict |

Divergence per pair over the six prompts. Prefix is the longest common prefix; edit distance is
character-level, normalized by the longer output; token difference is `|Δ completion_tokens|`.

| Pair | Exact | Prefix chars, worst | Prefix fraction, worst | Norm. edit distance, mean / worst | Token diff, worst |
|---|---:|---:|---:|---:|---:|
| baseline vs. draft repeat 1 | 0/6 | 297 | 0.184 | 0.081 / 0.227 | 3 |
| baseline vs. draft repeat 2 | 3/6 | 151 | 0.125 | 0.104 / 0.262 | 0 |
| **control**: draft r1 vs. r2 | 0/6 | 151 | 0.128 | 0.147 / 0.256 | 3 |
| baseline vs. n-gram, no control | 2/6 | 151 | 0.127 | 0.117 / 0.308 | 2 |

Both gates return `within_control_envelope`. Neither comparison of the baseline against a
speculative deployment diverges more than the two identically configured speculative deployments
diverge from each other — on the mean normalized edit distance the control is the *worst* pair of the
three (0.147 against 0.081 and 0.104). Divergent pairs share 151–564 characters of prefix and then
continue differently but fluently, with completion-token counts within 3.

The last row is measured but deliberately left unjudged. `ngram-n3` was launched once, so there is
no second deployment of that configuration to calibrate it against, and the control above belongs to
a different configuration. Borrowing it would be an abuse worth spelling out, because the answer
would flip on one statistic: the n-gram pair's worst normalized edit distance, 0.308, sits just
outside the draft control's 0.256 plus the 0.05 margin. A six-prompt control from another
configuration is far too thin to carry that call, which is exactly why the gate requires a
same-configuration control instead of the nearest available one. The n-gram arm therefore has a
recorded divergence profile and no losslessness verdict.

So this run shows **no divergence beyond deployment-level numerical nondeterminism** for the drafter
arm, which is consistent with speculative decoding being lossless here. It is not a proof: the gate
can only say that the effect is not separable from the control, and the control is a single
cross-deployment pair of six prompts drawn from the speculative arm, so baseline-side variation is
not measured. The project has one baseline deployment, so a baseline-vs-baseline control does not
exist yet; the next baseline launch should add one, and every deployment should capture twice so a
within-deployment control exists too.

## Reproducing

From the repository root, against the committed captures:

```bash
apertus-bench correctness-gate \
  --treatment results/correctness/smoke-20260913-debug/baseline.json \
              results/correctness/smoke-20260913-debug/draft-n3-tp4.json \
  --control results/correctness/smoke-20260913-debug/draft-n3-tp4.json \
            results/correctness/smoke-20260913-debug/draft-n3-tp4-repeat2.json \
  --output /tmp/gate.json
```

`apertus-bench compare-captures BASELINE CANDIDATE --output ...` records one pair's divergence
without judging it, which is how the n-gram row above was produced.

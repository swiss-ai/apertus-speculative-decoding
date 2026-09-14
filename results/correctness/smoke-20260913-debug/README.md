# Greedy correctness, 2026-09-13, `debug` partition

Sequential greedy captures for the baseline and both deployment repeats of each speculative arm
(`draft-n3-tp4` and `ngram-n3`) of `../../smoke-screening-20260913-debug/`, plus two generations of comparison
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
| `ngram-n3.json` | `ngram-n3` repeat 1 | 3392370 | nid006687 |
| `ngram-n3-repeat2.json` | `ngram-n3` repeat 2 | 3403354 | nid007500 |

Six smoke prompts, temperature 0.0, top_p 1.0, per-request seed `1 + request_index`, natural EOS,
`max_model_len=131072`, one request at a time. Every request succeeded in all five captures, so all
six prompts are comparable in every pair. Each capture keeps the full output text, so any comparison
criterion can be recomputed from these files without re-running the deployments.

## Exact-match comparisons, as first recorded

`apertus-bench compare-captures` at harness revision `270413f2`, schema version 1:

| Comparison | Artifact | Exact matches |
|---|---|---:|
| baseline vs. draft repeat 1 | `comparison.json` | 0 of 6 |
| baseline vs. draft repeat 2 | `comparison-repeat2.json` | 3 of 6 |
| draft repeat 1 vs. draft repeat 2 | `comparison-draft-repeat1-vs-repeat2.json` | 0 of 6 |
| baseline vs. n-gram repeat 1 | `comparison-ngram-n3.json` | 2 of 6 |
| baseline vs. n-gram repeat 2 | `comparison-ngram-repeat2.json` | 1 of 6 |
| n-gram repeat 1 vs. repeat 2 | `comparison-ngram-repeat1-vs-repeat2.json` | 1 of 6 |

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
| `divergence-baseline-vs-ngram.json` | baseline vs. n-gram repeat 1 | compared pair, uncalibrated |
| `comparison-ngram-repeat2.json` | baseline vs. n-gram repeat 2 | compared pair |
| `comparison-ngram-repeat1-vs-repeat2.json` | n-gram repeat 1 vs. repeat 2 | calibration control |
| `gate-baseline-vs-draft-repeat1.json` | baseline vs. draft repeat 1 | verdict |
| `gate-baseline-vs-draft-repeat2.json` | baseline vs. draft repeat 2 | verdict |
| `gate-baseline-vs-ngram-repeat1.json` | baseline vs. n-gram repeat 1 | verdict |
| `gate-baseline-vs-ngram-repeat2.json` | baseline vs. n-gram repeat 2 | verdict |

Divergence per pair over the six prompts. Prefix is the longest common prefix; edit distance is
character-level, normalized by the longer output; token difference is `|Δ completion_tokens|`.

| Pair | Exact | Prefix chars, worst | Prefix fraction, worst | Norm. edit distance, mean / worst | Token diff, worst |
|---|---:|---:|---:|---:|---:|
| baseline vs. draft repeat 1 | 0/6 | 297 | 0.184 | 0.081 / 0.227 | 3 |
| baseline vs. draft repeat 2 | 3/6 | 151 | 0.125 | 0.104 / 0.262 | 0 |
| **control**: draft r1 vs. r2 | 0/6 | 151 | 0.128 | 0.147 / 0.256 | 3 |
| baseline vs. n-gram repeat 1 | 2/6 | 151 | 0.127 | 0.117 / 0.308 | 2 |
| baseline vs. n-gram repeat 2 | 1/6 | 47 | 0.041 | 0.213 / 0.392 | 40 |
| **control**: n-gram r1 vs. r2 | 1/6 | 47 | 0.041 | 0.163 / 0.264 | 38 |

Both draft gates return `within_control_envelope`. Neither comparison of the baseline against a
draft deployment diverges more than the two identically configured draft deployments diverge from
each other — on the mean normalized edit distance that control is the *worst* pair of the three
(0.147 against 0.081 and 0.104). Divergent pairs share 151–564 characters of prefix and then
continue differently but fluently, with completion-token counts within 3.

**The n-gram gates split, and that is a statement about the gate rather than about n-gram.** The
second n-gram deployment supplies the same-configuration control that arm previously lacked, and it
is tighter than the draft control on every statistic, so it sets the allowance: worst normalized edit
distance 0.264 + 0.05 = 0.314, worst token difference 38 + 2 = 40.
`gate-baseline-vs-ngram-repeat1.json` returns `within_control_envelope` at 0.308, and
`gate-baseline-vs-ngram-repeat2.json` returns `exceeds_control_envelope` at 0.392 on that one check
of four — passing the other three, with its token difference exactly on the allowance. Two
deployments of one configuration returning opposite verdicts against an identical control means the
instrument's resolution is coarser than the effect it is asked to detect: the allowance is a
worst-of-six order statistic from a single control pair, so it is itself noisy, and adding the
draft-repeat pair as a second control does not loosen it because the gate takes the tightest
envelope. **Do not read "the gate passes for n-gram" out of these files.**

So this run shows **no divergence beyond deployment-level numerical nondeterminism** for the drafter
arm, which is consistent with speculative decoding being lossless here, and for n-gram it shows that
the divergence between arms is of the same character and roughly the same size as the divergence
between two deployments of one arm. Neither is a proof: the gate can only say whether the effect is
separable from the control, both controls are cross-deployment pairs of six prompts drawn from a
speculative arm, so baseline-side variation is not measured. The project has one baseline deployment,
so a baseline-vs-baseline control does not exist yet; the next baseline launch should add one, and
every deployment should capture twice so a within-deployment control exists too. Six prompts cannot
adjudicate losslessness at this resolution regardless.

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

Substituting `ngram-n3.json` / `ngram-n3-repeat2.json` for both the treatment candidate and the
control reproduces the split verdict above. `apertus-bench compare-captures BASELINE CANDIDATE
--output ...` records one pair's divergence without judging it.

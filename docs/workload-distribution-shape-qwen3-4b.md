# Workload distribution shape on Qwen3-4B (Red Hat DSpark drafter)

Repeat of the Apertus 1.5 8B study ([`workload-distribution-shape.md`](workload-distribution-shape.md),
job 4004306) on a second target, as Yu suggested on 2026-11-22: acceptance depends
on (1) target entropy, (2) target-drafter alignment and (3) greedy vs sampled
drafting, and Qwen3-4B "shows a clearer trend of the entropy". The question is
whether the Apertus verdict (alignment, not entropy, explains the chat and
non-English gaps) holds on a target and a drafter we did not train.

Status: **pre-registered, not yet run** (this section is written before any
measurement; results will be added below it without editing it).

## Setup (frozen before running)

- Target: `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`,
  its own chat template with `enable_thinking=False`, bfloat16, TP=1.
- Drafter: `RedHatAI/Qwen3-4B-speculator.dspark`, revision
  `bdffaa5f011da5a1a58ad841a1f6ced3e87f2f23` (Apache-2.0; speculators
  `DSparkDraftModel`, block size 7 with sampling from the anchor, vanilla Markov
  head rank 256, aux layers [1, 9, 17, 25, 33], mask token 151669, 5 causal
  sliding-window layers of width 2048, full 151,936 vocabulary). No AngelSlim or
  z-lab heads. No EAGLE head exists for this target, so the study has one drafter.
- Prompts: the same 28 slices x 32 prompts (896) as the Apertus run, selected by
  the frozen ids of [`results/8b/workload-shape/inputs/suite.yaml`](../results/8b/workload-shape/inputs/suite.yaml)
  (corpus digest `7d8e60ef...`), the same slice labels and ID/OOD classes. The
  classes describe the Apertus drafter's training corpus, not Red Hat's; they are
  kept so the slices line up, and the data-driven (k-means) split is reported too.
- Trajectories: plain vLLM greedy continuations of the target, C=1, 512-token cap,
  thinking off. Same pipeline (`apertus_tree.shape`, `shape_aggregate`, `analyze.py`).
- Sampled drafting (Yu's third factor): replayed from the same teacher-forced
  distributions, per-depth acceptance probability 1 - TV(p, q) chained along the
  trajectory, at T=1.

## Pre-registered decision rule

Computed on the replayed DSpark k=3 acceptance length across the 28 slices, with
the decomposition against the same ID reference slices as the Apertus run (opb,
math-gsm8k, probe-math, probe-humaneval, multi-turn).

- **Entropy hypothesis holds** if the mean-target-entropy R² is >= 0.8 across
  slices AND the entropy-mix (shape) term explains >= 50% of the chat gap and of
  the pooled non-English (12 `lang-*` slices other than lang-eng) gap.
- **Alignment generalises** if the TV R² exceeds the entropy R² by >= 0.3 AND the
  alignment term explains >= 70% of chat's gap.
- **Otherwise: mixed**; report the per-slice decomposition.

Validation gate before any conclusion: serve Qwen3-4B + this DSpark head in vLLM
at k=3 (and k=7, its default), C=1, greedy, on the same 896 prompts. The offline
replay must match the served acceptance per slice with Pearson r >= 0.98 and mean
absolute difference <= 0.1 tokens per step. If the gate fails, the replay is
debugged, not interpreted. If no available vLLM can serve the head, the results
are marked unvalidated and the logit test against the speculators reference is
the only check.

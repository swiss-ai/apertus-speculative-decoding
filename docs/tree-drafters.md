# Tree-based speculation with several drafters (TreeGraft) for Apertus 1.5 8B

Branch `tree-drafters`. Yu's suggestion (Slack, 17 Oct): follow TreeGraft
(arXiv 2608.26112), which builds one draft tree from several drafters.

## Summary

_Pending: filled in once the simulation (step B) has run._

## A. What TreeGraft does

Read from paper v3 (14 Oct 2026) and the released code
(`github.com/fjm9933/TreeGraft` at `e4d1376`, cloned and read on the cluster
only).

**Drafters.** A training-free n-gram "small drafter" plus a pretrained small
LM of the target's family as the "middle drafter" (Llama 3.2 1B/3B or 3.1 8B
for Llama 3.3 70B and 3.1 8B; Qwen3 0.6B-8B for Qwen3 32B/8B; 10 pairs).
There is no trained head of the EAGLE/DSpark kind; the paper names combining
TreeGraft with trained drafters as future work.

**Tree building and grafting.** Up to D = 5 drafting steps, frontier W = 10,
m = 10 children per expanded node, verification budget B_ver = 63 draft
nodes. Every step ranks nodes by path probability:

- the n-gram drafter extends the frontier from matches of the context suffix
  (n = 3 down to 1, then a frequency fallback), scoring a child by the
  target's cached prefill log-probability at the matched source position;
- when the middle drafter is called, it runs one tree-attention forward over
  the whole shared tree, overwrites every node's path score with its own
  log-probabilities, re-selects the frontier among all nodes of the last few
  depths (lookback 5, so it can graft at earlier, previously unexpanded nodes)
  and attaches its top-10 children to each. Existing branches are never
  removed ("non-destructive").
- Before verification the tree is pruned to B_ver by path score, keeping
  ancestor chains. The target verifies it with tree attention (EAGLE-style
  mask and posterior), lossless.

**Scheduler.** Per drafting step, call the middle drafter or not. Offline, a
value system (four MLP heads, 23 inputs: runtime context such as prompt
length, measured T_tgt and T_mid, the call history, six tree signals such as
depth, leaf ratio, frontier score and overlap after reselection) predicts
accepted length and latency for call/skip followed by every future suffix;
the per-state margin "best throughput if called minus best if skipped" is
distilled into a small MLP (released as a weights table) that calls the
middle drafter when the margin is positive, at most B0 = 5 (default
floor(T_tgt / T_mid)) calls per step. Fitting data: 960 trajectories, about
160 A100 GPU-hours.

**Costs and assumptions.** One request at a time (the code indexes
`input_ids[0]`; the appendix's "batch size 11" contradicts the checklist's
batch 1), Hugging Face models with their own KV-cache wrappers, 2 A100s,
temperature 0, 256 new tokens. Measured per call: target verification of the
63-node tree 69 ms (Llama 70B: 122 ms), middle-drafter tree forward 29-55 ms,
n-gram 0.9 ms, scheduler 0.3 ms. So a middle-drafter call costs 40-70% of a
target forward, and the scheduler's job is to amortise it. Reported speedups
over autoregressive decoding: TreeGraft 1.60x average, n-gram only 1.32x,
middle drafter only 1.39x, so +15.1% over the better endpoint (max +26.6%,
46 of 60 settings won). No accepted lengths are reported for the main
results, and nothing about batching or load.

**What transfers to us.**

- The setting is far from ours. Our drafters are trained heads that already
  accept 4.7-4.8 tokens per step at k=7 in distribution (TreeGraft's
  ablation trees accept about 2.2), drafting costs 2-4 ms rather than 30-50,
  and we serve many requests per GPU, where every verified token costs
  throughput (about 28 us/token above 1k batch tokens).
- Tree verification with a 63-node budget is the low-load regime only: at C=64
  a 64-node tree per request is 4,160 verified tokens per step, about 7x the
  576-token CUDA-graph limit.
- The transferable ideas are (1) grafting candidates of several drafters into
  one trie and ranking them by the stronger drafter's probability; (2) a cheap
  rule for calling the expensive drafter only when the cheap one is unsure.
  Our candidates: DSpark e10 (one parallel block pass; its vanilla Markov head
  makes the children of any node exact for the cost of a rank-256 bias, so a
  DSpark tree needs no further drafter passes), EAGLE e10 (one pass per depth,
  EAGLE-2 dynamic tree) and n-gram prompt lookup (free).
- The learned scheduler does not transfer: it is fitted to HF timings at batch
  1 and to the n-gram + small-LM pairs. We test simple threshold rules instead.

## B. Offline simulation

_Pending._

## Runs

| Job | What | Output (cluster) |
| --- | --- | --- |
| 3722023 | validation (failed in 43 s: no target contract selected; fixed in 3f3f734) | `spec-exp-data/tree-drafters/runs/validate-20261022T1440Z` |
| 3722336 | full run, cancelled before start (same failure) | `spec-exp-data/tree-drafters/runs/full-20261022T1530Z` |
| 3730086 | validation: greedy captures (128 opb-heldout + 128 chat), candidates for 4 + 4 sequences x 120 anchors | `spec-exp-data/tree-drafters/runs/validate-20261023T1400Z` |
| 3730087 | full run: captures, candidates for all 256 sequences, replay | `spec-exp-data/tree-drafters/runs/full-20261023T1400Z` |

# Tree-based speculation with several drafters (TreeGraft) for Apertus 1.5 8B

Yu's suggestion (Slack, 17 Oct): follow TreeGraft
(arXiv 2608.26112), which builds one draft tree from several drafters.

## Summary

**Verdict: do not implement multi-drafter serving from this screen.** The
completed offline replay shows low-load headroom from a single DSpark tree,
but adding n-gram candidates changes the best projected throughput by less
than 0.1%; running both neural drafters is 10–27% slower than the best
single-drafter endpoint. At C>=64 no substantial tree/grafting throughput
gain remains. These are decode-only cost projections, not serving speedups.

Recovery completed 2026-10-27. Jobs 3737472/3737473 had finished on 25/26
October and were collected afterwards. Their original high-load
projections are invalid: the cost model extrapolated a downward timing-table
segment into negative verification time. The corrected, versioned analysis
and all nonrestricted raw evidence are in
[`results/8b/tree-drafters/screen-20261024/`](../results/8b/tree-drafters/screen-20261024/README.md).

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

**Inputs and validity.** One greedy target capture of 128 held-out
Open-PerfectBlend and 128 chat prompts, with up to 512 output tokens, supplies
86,844 candidate anchors. The replay follows each policy's own accepted-prefix
jumps. DSpark and EAGLE chains at k=7 emit 4.818/4.802 tokens per step on OPB
and 2.625/2.556 on chat (target bonus included). Per-shard HF/vLLM next-token
agreement is 98.99–99.44%; this numerical mismatch and the single capture per
workload limit interpretation. No tree-serving correctness benchmark was run.

**Cost repair.** Raw timings end at 1,984 tokens / 65.996 ms and 2,048 /
65.161 ms. Extending that last negative slope produced -18.339 ms verification
for C=256 with a 32-node tree, and a spurious 13.37x OPB speedup. The original
report is retained unchanged; its throughput projections are superseded.
The repaired model interpolates a cumulative-maximum latency envelope and
extrapolates using a nonnegative least-squares slope over the upper half of
the measured range (31.52 us/token). Every out-of-range prediction is flagged.
All reported best chain/single-tree/grafted settings below stay within the
1–2,048-token measured range; their choices also survive halving or multiplying
the extrapolation slope by 1.5. Tree-attention kernels, metadata, prefill and
serving overhead are not measured by this model.

Projected decode throughput relative to plain, choosing the best budget for
each endpoint (full C=1/8/64/128/256 tables and budgets are in
[`comparisons.csv`](../results/8b/tree-drafters/screen-20261024/analysis-v2/comparisons.csv)):

| workload | C | best chain | best single tree/chain | best grafted policy |
| --- | ---: | ---: | ---: | ---: |
| OPB | 1 | 3.52x | 4.38x | 4.38x |
| OPB | 64 | 1.75x | 1.79x | 1.77x |
| OPB | 256 | 1.32x | 1.33x | 1.25x |
| chat | 1 | 1.92x | 2.49x | 2.49x |
| chat | 64 | 1.28x | 1.30x | 1.26x |
| chat | 256 | 1.06x | 1.06x | 1.00x |

At C=1 the best single DSpark tree uses 32 nodes on OPB and 48 on chat:
6.109/3.551 tokens per step versus 4.818/2.625 for the best chains. This is
24%/30% projected throughput headroom before unmeasured integration costs.
The best graft simply adds n-gram candidates to that same tree and barely
changes its result. Adding EAGLE to DSpark gives more accepted tokens, but
its extra passes make the best two-neural-drafter setting 27%/20% slower
than the single tree at C=1 and about 10–12% slower at C>=64.

**Original G4 acceptance gate.** At a four-node verify budget, grafting emits
3.945 versus 3.792 tokens/step on OPB (+4.0%) and 2.660 versus 2.450 on chat
(+8.5%): below the 10% affordable-budget threshold. At the low-load 16-node
budget it reaches 6.061 versus 4.818 (+25.8%) and 3.497 versus 2.625 (+33.2%),
passing the original 15% chain comparison. However, the best single trees
already give 5.856/3.268 there. Thus the formal low-load gate passes largely
because it compares trees with chains; it does not establish useful
multi-drafter value. There is no replicated confidence interval or measured
end-to-end gain to justify implementing multiple drafters.

## C. Design-only consequence of the low-load gate

The bounded next candidate would be **one DSpark tree at low load**, falling
back to the existing short-chain path under load. A prototype would need
tree attention in model runner v2, ancestor/position metadata, prefix
verification with the target bonus, tree-aware KV commit/rollback, and CUDA
graph capture for the selected budgets. Correctness must cover rejection,
EOS/caps, preemption and continuous batching against same-configuration
controls before timing it. Neither the simulator nor the image's recorded
source search establishes that these facilities exist.

Adding EAGLE would require a second resident parameter/KV state and a joint
proposal scheduler; their actual memory and runtime costs are unmeasured.
There is no evidence here that this extra complexity pays. A rough engineering
estimate is 2–4 engineer-weeks for a research single-tree prototype plus
serving validation, with kernel integration the main uncertainty; this is a
planning estimate, not measured effort. No serving implementation or further
GPU run was launched. Keep the single-tree low-load result as research
headroom; stop the multi-drafter implementation direction for this setup.

## Runs

| Job | What | Output (cluster) |
| --- | --- | --- |
| 3722023 | validation (failed in 43 s: no target contract selected; fixed in 3f3f734) | `spec-exp-data/tree-drafters/runs/validate-20261022T1440Z` |
| 3722336 | full run, cancelled before start (same failure) | `spec-exp-data/tree-drafters/runs/full-20261022T1530Z` |
| 3730086 | validation: greedy captures done (128 opb-heldout: 49,557 tokens; 128 chat: 37,543); candidates failed (teacher load needs accelerate; fixed in c04554d) | `spec-exp-data/tree-drafters/runs/validate-20261023T1400Z` |
| 3730087 | full run, failed the same way | `spec-exp-data/tree-drafters/runs/full-20261023T1400Z` |
| 3737472 | **Completed 25 Oct**, 221 s: 4 + 4 sequences x 120 anchors, replay; original projected costs superseded | `spec-exp-data/tree-drafters/runs/validate-20261024T1000Z` |
| 3737473 | **Completed 26 Oct**, 411 s: all 256 sequences, 86,844 anchors; original projected costs superseded | `spec-exp-data/tree-drafters/runs/full-20261024T1000Z` |
| local reanalysis, 27 Oct | Cost fix `806d3be8`; 17 focused simulator regressions pass; acceptance aggregates unchanged; no new GPU work | `results/8b/tree-drafters/screen-20261024/analysis-v2/` |

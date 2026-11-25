# Tree drafters review: PCTree and TreeSpark for Apertus 1.5 8B

Action item F from the 2026-11-20 huddle (Yu, Rob, Faruk): Yu suggested
tree-based drafters to raise acceptance length at low concurrency, noting
that they cost compute at high concurrency. This is a reading and
feasibility review only. No GPU job was run.

## Summary

**Recommendation: worth a bounded follow-up, but only as a low-load mode.
Do not retrain (TreeSpark) yet.**

- PCTree is an inference-time change to DSpark. It needs no retraining, and
  our earlier offline screen already simulated almost the same algorithm on
  our DSpark head. That screen projected +24% (OPB) and +30% (chat) decode
  throughput at C=1, about +11% at C=8, and roughly nothing at C>=64.
  PCTree's own batch-1 results show the same pattern: chat gains are larger
  than math gains.
- TreeSpark retrains the drafter with tree-aware losses and wider attention
  heads. Over the same untrained tree (the paper's own "DSpark + Tree"
  control), it adds only about 6% acceptance and 4-5% throughput. That is
  not worth a training run until tree serving exists.
- **The blocker is serving, not modelling.** vLLM removed its tree-attention
  backend on 2026-05-09 (PR #42121, "no plans to support it in the short to
  medium term"). Neither v0.30 nor v0.31 has it. The only route to upstream
  support is an open, unreviewed plugin PR (#64844, +6.2k lines, merge
  conflicts). That PR leaves the tree attention kernel to the plugin.
- Our projections assumed tree construction is almost free (0.01 ms). PCTree
  measured 3.5-5.2 ms in HF eager mode against a 3.5 ms drafter backbone.
  Break-even is about 2 ms of added overhead at C=1 and about 1 ms at C=8.
  The next step is therefore a cheap microbenchmark of these two unmodeled
  costs (tree construction and masked verification) before anyone writes
  serving code.
- On Yu's workload-entropy problem: trees partly mitigate flat chat
  distributions at low load, by hedging over the top-k instead of
  committing to one draft. They do not fix draft/target misalignment, and
  they give nothing at high load, where chat already falls back to short
  chains.

## 1. The papers

### PCTree: "From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding"

arXiv 2608.02123v1 (3 Aug 2026), Li, Li, Xie, Song, Lu (OPPO AI Center).
Read from the arXiv HTML.

**Method.** PCTree is an inference-only extension of DSpark.

- One DSpark backbone forward produces shared base logits `L_0..L_{B-1}` for
  the B block positions.
- B sequential "Markov stages" then expand the tree one depth at a time.
  Every frontier parent `p` gets its own correction
  `z_d(p) = L_d + Markov(p)`. This uses the same vanilla Markov head (rank
  256) that DSpark already has, and all parents at one depth are batched
  into a single Markov call.
- Children are scored by joint log-probability. Each parent proposes its
  top-k children, and the top k nodes of each layer form the next frontier,
  a beam-like expansion in the style of EAGLE-2. The candidate pool is at
  most `k + (B-1)k²` nodes. The global top-N nodes by score are kept, with
  ties going to the shallower node, so the result is a prefix-closed tree.
- Defaults are k=4 and N=32.
- Verification uses a flat token tensor with an ancestor-only attention
  mask. The position of each node is its depth plus the prefix length.
  Greedy verification takes the longest accepted root-to-leaf path, then
  gathers and compacts its KV and hidden states.

**Training.** None. Within each matched comparison, the paper uses the same
DSpark checkpoint for chain and tree.

**Results.** All results are greedy, batch size 1, Hugging Face generation
with SDPA, on one H20, for Qwen3-4B/8B/14B.

Acceptance τ, Qwen3-4B, B=7, DSpark chain → PCTree:

| Benchmark | Chain | PCTree | Gain |
| --- | ---: | ---: | ---: |
| GSM8K | 6.31 | 7.24 | +15% |
| MT-Bench | 3.82 | 5.07 | +33% |
| Alpaca | 3.67 | 4.83 | +32% |
| Arena-Hard | 3.81 | 4.83 | +27% |

Speedup over autoregressive, Qwen3-4B:

- B=7: MT-Bench 2.48x → 3.20x (+29%), Alpaca 2.39x → 3.02x, Arena-Hard
  2.52x → 2.93x, GSM8K 4.24x → 4.50x (+6%).
- B=16: chat +10-15%, math +5-12%.

PCTree beats the chain in every cell. Gains saturate at k=4, and speedup
peaks at N=32 (B=7) or N=64 (B=16).

**Costs (Table 6, Qwen3-4B).** The tree cuts rounds per sample from 39.9 to
35.1 at B=7. In exchange, the Markov and tree-building cost rises from
0.65 ms to 3.52 ms per round (from 1.09 ms to 5.24 ms at B=16), against a
backbone of about 3.5 ms. This is eager PyTorch.

**Serving.** No vLLM or SGLang, no code release, and no measurement at batch
size above 1. There is no entropy analysis and no adaptive tree size; the
paper disables confidence scheduling.

### TreeSpark: "Learning to Draft the Trees You Verify for Speculative Decoding"

arXiv 2610.20175v1 (14 Oct 2026), Zhao et al. (Renmin University, Nanjing
University, Huawei). Code: github.com/Robot2050/TreeSpark. Read from the
arXiv HTML.

**Method.** TreeSpark builds the same parent-conditioned tree as PCTree:

- One DSpark backbone pass, then a per-node correction
  `l_d(x) + W·e(t_u)` with rank 256.
- B=7, k=4, frontier k, N=32 non-root nodes, a pool of at most 100
  candidates, global top-N packing, and an ancestor tree mask.

The difference is that the drafter is trained for the tree:

- Loss: `L = L_base + 0.1·L_surv + 0.1·L_cov`.
  - `L_surv` is NLL on the packed tree edges, weighted by the reference
    target path mass.
  - `L_cov` is a capped hinge that pushes the target's top-4 tokens above
    the k-th draft logit, including on pruned parents.
- Training replays the drafter on its own proposed parents (128 anchors
  from 512 blocks per sequence) and reuses reference distributions, so it
  needs no extra target passes.
- Draft attention head dimension grows from 128 to 192.
- 10 epochs on 50k prompts (30% math, 40% code, 20% dialogue, 10%
  writing).

**Results.** Greedy decoding, Qwen3-4B/8B, on Ascend 910B. The framework is
not named.

Acceptance (4B, DSpark chain → TreeSpark):

- General mean 4.23 → 5.25 (+24%).
- Chat: MT-Bench 2.80 → 3.59, Alpaca 2.63 → 3.42, Arena-Hard-v2
  2.90 → 3.64 (+25-30%).
- WritingBench +31%.

Throughput (4B):

- Single request: +15% (general), +22% (writing).
- System throughput: +16.3% at C=6, +17.2% at C=12, and +12.6% at C=36.
  The C=36 run used three replicas of 12 requests, so it is still about 12
  requests per device.

**Control that matters.** "DSpark + Tree" has the same capacity as TreeSpark
but uses no tree losses; it is effectively PCTree decoding. TreeSpark
improves on it by only 4.93 → 5.25 acceptance (+6.4%) and by +4.2% / +5.0%
system throughput at C=12 / C=36. Most of TreeSpark's gain comes from the
tree, not from the retraining.

**Limits.** Greedy only and no error bars. There is no high-entropy
analysis, the tree size is fixed, and the inference-side tree overhead is
not isolated.

### What the two papers agree on

1. The tree comes almost free from DSpark's Markov head: no extra backbone
   passes, only B small batched corrections.
2. Gains are largest on chat and open-ended writing, where chains are
   shortest (+27-33% acceptance), and smallest on math.
3. Neither paper shows the high-concurrency regime on GPUs or in a
   production server. The largest measured load is about 12 requests per
   device.

## 2. Workload entropy (Yu's problem)

Yu's finding is that chat target distributions are flatter (higher entropy)
than math or code, and that the drafter aligns worse with them.

Neither paper measures entropy, but the mechanism fits. A chain commits to
the drafter's argmax, so on a flat target the chain dies at the first
near-tie. A tree keeps the top 4-10 siblings and lets the target pick
among them. Our own replay shows this directly (DSpark e10, greedy):

| Workload | Chain k=5 → k=7 | Tree, 7 nodes, vs chain k=7 | Tree, 32 / 48 nodes, vs chain k=7 |
| --- | ---: | ---: | ---: |
| OPB (low entropy) | 4.22 → 4.82 (+14%) | 5.11 (+6%) | 6.11 / 6.27 (+27% / +30%) |
| chat (high entropy) | 2.54 → 2.63 (+3%) | 2.88 (+10%) | 3.44 / 3.55 (+31% / +35%) |

On chat, making the chain deeper is already useless, while making it wider
still pays. This is exactly the regime where trees help.

Limits:

- Even a 48-node tree on chat (3.55) stays below the OPB chain (4.82).
  Trees treat the symptom (a flat target) but not the alignment gap that Yu
  is attacking with data and objective changes.
- The benefit disappears under load. At C>=64 the best chat policy is a
  1-2 token chain, and a tree adds at most 1.5%.
- Sampling at temperature > 0 is untested by both papers. Tree acceptance
  with stochastic verification (multi-draft rejection sampling) is harder
  and usually gains less.

A cheap extension follows from this: size the tree by the drafter's own
confidence or entropy, wide on flat positions and chain-like on peaked
ones. DSpark already has a confidence head, but neither paper does this.

## 3. Feasibility on our stack

### Can DSpark e10 be reused as-is?

**For PCTree: yes.**

- Our head (`inputs/dspark-config.json` of the screen) has
  `markov_head_type: vanilla`, `markov_rank: 256`,
  `sample_from_anchor: true` and `block_size: 8`. This is the configuration
  PCTree runs on.
- The earlier simulator
  (`methods/tree/apertus_tree/candidates.py` on branch `tree-drafters`)
  already builds exactly this tree: slot logits plus the Markov bias of
  each parent, EAGLE-2 frontier expansion, and a global top-N.
- It used width = top-k = 10 rather than the papers' k=4. Per PCTree's
  ablation (k=4 ≈ k=8), that should not change the conclusion.

**For TreeSpark: no.** It needs a full retrain: head_dim 192 changes the
attention projections, and the new losses add replay. Expected incremental
value over PCTree decoding is +5% throughput (from the paper's control).

### What vLLM needs

Status in the versions we use:

- The V1 `TREE_ATTN` backend (EAGLE-only, static `speculative_token_tree`)
  existed up to v0.20.0.
- It was removed by vllm-project/vllm#42121 (merged 2026-05-09) because "not
  fully supported and there's no plans to support it in the short to medium
  term".
- It is absent in v0.25.0, v0.30.0 and v0.31.0, checked against the Git
  trees of those tags. `SpeculativeConfig` in v0.31.0 has no tree option.
  Our earlier screen's image (0.23.1rc1) also lacked it.
- An earlier PR that tried to add tree infrastructure, #59091, was closed
  unmerged (2026-09-28).

Upstream path: **vllm-project/vllm#64844**, "[Spec Decode] Verify
tree-shaped drafts and size each step's drafts".

- Opened 2026-10-31. Open, no review, merge conflicts. +6,241 / -109 lines.
- It builds on #63205 (custom speculators on model runner V2), which merged
  on 2026-10-30, after v0.32.0. It therefore needs a post-0.32 base, not our
  v0.31 image.
- It lets a speculator plugin return a verify plan (tree), a per-step
  candidate budget, the accept walk and commit kernels. It also handles
  tree-depth positions and a device-side path check.
- "vLLM ships no tree kernel": the plugin must also provide an attention
  backend that accepts the ancestor mask.
- The per-step candidate budget is exactly the "tree at low load, short
  chain at high load" switch we would want.

Work we would have to write on either path:

1. **DSpark tree proposer:** B sequential batched Markov stages, top-k
   selection, and global top-N packing, all inside CUDA graphs for a few
   fixed (C, N) shapes.
2. **Tree-masked verification attention for the target:** an
   ancestor-bitmask path in the Triton or FlashInfer kernel. FlashInfer
   supports custom masks but on a slower path, and FA3 does not support
   arbitrary masks.
3. **Tree accept and KV compaction:** rejected sibling KV must be discarded
   or invalidated, and the accepted path's KV moved to contiguous slots.
   The DSpark drafter's context state also has to follow the accepted path.
4. **A load-based fallback** to the existing chain (k=3 to k=7).

The previous review's estimate still stands: 2-4 engineer-weeks for a
research prototype plus serving validation, with kernel integration as the
main risk. If #64844 lands, items 3-4 and the scheduler plumbing mostly come
for free and the work drops to the plugin (items 1-2). A private patch on
v0.31 is possible, but it would be a large, hard-to-rebase fork of model
runner V2.

### Projected headroom, and the costs the projection does not include

These numbers come from the corrected screen
(`results/8b/tree-drafters/screen-20261024/analysis-v2/`, on main). They are
decode-only cost-model projections relative to plain decoding, comparing the
best DSpark chain with the best DSpark tree.

| Workload | C | Best chain | Best tree | Gain | Allowed extra tree overhead (break-even) |
| --- | ---: | ---: | ---: | ---: | ---: |
| OPB | 1 | 3.52x (k=7) | 4.38x (N=32) | +24% | 1.9 ms/step |
| chat | 1 | 1.92x (k=7) | 2.49x (N=48) | +30% | 2.4 ms/step |
| OPB | 8 | 3.29x (k=7) | 3.65x (N=10) | +11% | 1.0 ms/step |
| chat | 8 | 1.85x (k=5) | 2.06x (N=10) | +11% | 1.0 ms/step |
| OPB / chat | 64 | 1.75x / 1.28x | ≈ +1-2% | none | — |

The break-even column is the extra per-step time (on top of the modelled
7.8-9.0 ms step) at which the tree's projected gain disappears.

Two costs are not in the model:

1. **Tree construction.** The model charged 0.006-0.02 ms. PCTree measured
   3-4 ms extra in eager PyTorch. Seven sequential small kernels per step
   should fit in about 0.2-0.5 ms under CUDA graphs, but that has not been
   measured.
2. **Masked verification.** The model used causal chain-verify timings for
   the same token count. A custom-mask attention path may be slower.

Agreement with the papers: our C=1 chat acceptance gain (+31% at N=32) is in
line with PCTree (+27-33%) and TreeSpark (+25-30%), and our C=8 throughput
gain (+11%) is in line with TreeSpark's +13-17% at 6-12 requests per
device. The headroom is real. Whether it survives depends on the
implementation overheads.

For the production deployment, the gap is larger: production runs
DSpark e10 at k=3, so the tree's C=1 gain over the deployed chain is about
+40% (chat). That is a statement about k at low load, though, not about
trees. A per-load k policy captures part of it without trees.

## 4. Proposed experiment (not run)

The offline acceptance replay is already done; only its cost assumptions
need checking. The experiment has three stages, each gated by a decision
rule.

**Stage 1: microbenchmark the unmodeled costs.** About 1 GH200 node-hour,
no serving code needed.

- On the pinned v0.31 serving image, load the target and the DSpark e10
  head as plain PyTorch modules and time:
  1. PCTree tree construction (7 Markov stages, top-k, top-N packing),
     eager and CUDA-graphed, for C ∈ {1, 2, 4, 8} and N ∈ {10, 16, 32, 48}.
  2. A target forward over C×(N+1) query tokens with the ancestor mask
     (FlashInfer custom mask and a Triton mask kernel), against a causal
     forward with the same token count at context lengths 1k and 4k.
- Reuse the screen's captures (restricted, on scratch) for real tree shapes,
  and record provenance per AGENTS.md.
- **Decision rule:** continue only if the measured construction time plus
  the mask penalty is at most half of break-even (≤1 ms at C=1, ≤0.5 ms at
  C=8), and the reprojected C=1 chat gain is ≥15%. Otherwise stop and
  record the trees as not viable on this stack.

**Stage 2: low-concurrency serving prototype.** 2-4 engineer-weeks, about
4 GH200 node-hours of measurement.

- Implement it as a #64844-style speculator plugin if that PR has merged by
  then; otherwise as a patch on our v0.31 image, using model runner V2 and
  the DSpark speculator.
- Tree for C ≤ 8 with N ∈ {10, 32}, the chain (k=3) above that.
- Correctness: greedy outputs must match the chain arm's token for token,
  within the known ~1% numerical mismatch, on the 28-slice task-speed
  suite. Include rejection, EOS, max-token caps and preemption cases.
- Timing: controlled plain / chain / tree triples with the task-speed
  harness (`experiments/task-speed/submit-controlled.sh`) at C = 1/2/4/8/32,
  ID and OOD slices, at least 3 repeats.
- **Decision rule:** adopt as a low-load mode only if, against the best
  chain:
  - TPOT improves by ≥15% at C=1 on chat;
  - TPOT improves by ≥5% at C=8;
  - no slice regresses by more than 2% at C ≥ 32 (where the fallback must
    engage);
  - KV and CUDA-graph memory costs are ≤5% of KV capacity.

**Stage 3 (only if Stage 2 passes): TreeSpark retraining.** This costs one
full DSpark e10-scale training run. Expected value is +5% throughput (from
the paper's control). Do it only if the low-load mode ships and that 5% is
worth a training run.

## Recommendation

**Worth doing, in a bounded way: run Stage 1 next (about 1 node-hour), and
commit to Stage 2 only if the overhead gate passes.**

- **For:**
  - No retraining is needed.
  - Two independent papers and our own replay agree on +25-35%
    acceptance on chat at low load, which is where our speedups are worst.
  - It directly addresses the low-concurrency part of Yu's chat-entropy
    problem.
- **Against:**
  - The gain is confined to C ≤ 8 (about +11% there, about 0 at C ≥ 64).
  - vLLM has no tree support in 0.30/0.31, and the only upstream path is an
    unmerged plugin PR without a tree kernel.
  - The tree-construction cost measured by PCTree (3-5 ms eager) would
    erase the gain unless it is graph-captured well below 1-2 ms.
- **Not recommended now:**
  - TreeSpark retraining.
  - Multi-drafter trees, already rejected in `docs/tree-drafters.md`.
- **Cheaper alternatives to weigh against Stage 2:**
  - A load-aware k for the chain (k=7 at low load instead of the deployed
    k=3).
  - Yu's alignment work on chat data.

## Sources

- PCTree: <https://arxiv.org/abs/2608.02123> (v1, 3 Aug 2026), read from the
  arXiv HTML.
- TreeSpark: <https://arxiv.org/abs/2610.20175> (v1, 14 Oct 2026), read from
  the arXiv HTML. Code: <https://github.com/Robot2050/TreeSpark>.
- vLLM: tree-attention removal
  [#42121](https://github.com/vllm-project/vllm/pull/42121); tag trees
  checked for `tree_attn` on v0.20.0 (present) and v0.25.0, v0.30.0 and
  v0.31.0 (absent); custom speculators
  [#63205](https://github.com/vllm-project/vllm/pull/63205) (merged
  2026-10-30); tree-verify plugin contract
  [#64844](https://github.com/vllm-project/vllm/pull/64844) (open, checked
  2026-11-22); closed earlier attempt
  [#59091](https://github.com/vllm-project/vllm/pull/59091).
- Earlier screen: [`docs/tree-drafters.md`](tree-drafters.md) and
  [`results/8b/tree-drafters/screen-20261024/`](../results/8b/tree-drafters/screen-20261024/README.md).
  All projected numbers above come from `analysis-v2/tree-sim.json` and
  `analysis-v2/summary.json` there.
- Huddle transcript 2026-11-20, 39:56-41:14 (Yu's suggestion); the entropy
  discussion is at 9:31-14:02.

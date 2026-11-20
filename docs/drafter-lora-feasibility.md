# Drafter LoRA feasibility (canvas A3, 8B)

Desk check, 2026-11-20, branch `experiment/drafter-lora-feasibility`. Question:
can domain specialization ship as LoRA adapters on one shared 8B drafter
(DSpark e10 / EAGLE 3.1 e10) instead of one full head per domain?
No GPU job was run (see "Smoke test" below).

## Verdict

**Not feasible now. Serving needs a vLLM patch and training needs a
speculators extension.** Quality could be tested cheaply first, without any
serving change, by training an adapter and merging it offline.

- Serving: our vLLM applies LoRA only to the target. The drafter is built
  with its own `get_model()` call and never wrapped by a LoRA manager. The
  speculative config has no adapter field, so per-request drafter-adapter
  selection does not exist.
- Training: speculators (pinned e3041ddb and upstream main a01c3b8 of
  2026-11-20) has no LoRA/PEFT code path for draft models.
- Upstream: drafter LoRA is an open, stale-flagged RFC with no
  implementation merged.

## Evidence

### (a) vLLM serving: LoRA reaches the target only

The served build is the Swiss AI vLLM fork at `a601a9d998ddeb488f0c17e8512874b116aa7658`,
the revision validated by the Kubernetes deployment
(`serving/kubernetes/README.md` on `serving/dspark-e10-kubernetes`). DSpark
comes in through our overlays/patches.

- `vllm/v1/worker/gpu_model_runner.py:5224-5234`: the target is loaded, then
  `if self.lora_config: self.model = self.load_lora_model(self.model, ...)`.
  After that the drafter is loaded with `self.drafter.load_model(self.model)`
  and is not wrapped.
- `vllm/v1/spec_decode/llm_base_proposer.py:1305-1319` (`_get_model`): the
  drafter comes from `get_model(vllm_config=draft_vllm_config, model_config=
  speculative_config.draft_model_config, ...)`, with no LoRA manager.
  `dflash.py`, `eagle.py`, `draft_model.py` and `config/speculative.py`
  contain no `lora` reference.
- `vllm/v1/worker/lora_model_runner_mixin.py:31-46`: there is one
  `LRUCacheWorkerLoRAManager`, created for the model passed in, which is the
  target.
- Our patches under `serving/patches/` contain no LoRA code. The three hits
  for "lora" are incidental context lines, not adapter logic.
- Upstream vLLM (checked 2026-11-20) has no drafter-side adapter either:
  - #63186 (merged 2026-10-23) only stops rejecting `--enable-lora` when the
    drafter lacks `SupportsLoRA`. Target LoRA and spec decode coexist; the
    drafter still gets no adapter.
  - #66254 (open) fixes DSpark startup with target LoRA by stripping
    `lora_config` from the drafter config. This confirms the drafter is not
    adapter-wrapped.
  - #55947 (open, MRV2) is "Uno" self-drafting: a draft-only LoRA on the
    *target*. It is a different method from a shared external drafter.

### (b) Training: no LoRA path in speculators

`/home/projects/speculators` at e3041ddb, and upstream main a01c3b8 fetched
2026-11-20: `git grep -i -E "lora|peft" -- src` returns nothing. Draft
models train full parameters. The only freezing is of embeddings and the
verifier norm (`models/eagle3/core.py:68,120`). Adding PEFT wrapping of the
draft model's `nn.Linear` layers (and exporting the adapter) would be new
code.

### (c) vLLM RFC #52038, "LoRA adapter support for DFlash draft models"

- Opened 2026-08-12. Still **open**. The bot marked it stale on 2026-11-11;
  comments resumed on 2026-11-17 and 2026-11-20. No linked PR is merged.
- Proposal: an adapter on all drafter layers except `lm_head` and
  `embed_tokens`, selected through new speculative-config arguments.
- The author's claims, which we have not verified: tested with a
  GPT-OSS-120B target and a 0.8B DFlash drafter. An r64 adapter is about
  28x smaller than the drafter, at about 98% of full per-domain drafter
  acceptance length (comment of 2026-11-20). Training is a stage-1 base plus
  stage-2 adapter fine-tune. Some details are private (classified data).

### (d) Memory and switching cost

Computed from the tensor shapes in the safetensors headers of the private HF
exports. Adapters cover q/k/v/o/gate/up/down of every drafter layer
(the RFC convention, no `lm_head`/embeddings), bf16:

| Head (full size) | r8 | r16 | r64 |
| --- | --- | --- | --- |
| DSpark e10, 5 layers, 1,717M params, 3.20 GiB (incl. Markov head) | 7.9 MiB (415x smaller) | 15.8 MiB (207x) | 63.1 MiB (52x) |
| EAGLE 3.1 e10, 1 layer, 831M params, 1.55 GiB (537M is `lm_head`) | 1.4 MiB (1,102x) | 2.9 MiB (551x) | 11.5 MiB (138x) |

Adding the `fc` input projection costs +0.4 to +3 MiB. Raw shapes and
script output: `results/8b/lora-feasibility/` (this branch).

Switching: the measured same-architecture full-head swap already pauses
streams for only **0.33-0.56 s** (job 3722022; read 0.23-0.49 s, copy 0.06 s;
`docs/drafter-updates.md`), against 207 s for a restart. So adapters barely
help *switching*. Their real advantage is memory when **many specialists
must be resident at once**, and **per-request mixing of domains in one
batch**. Our full-head swap cannot mix: one head serves the whole replica.

## Smoke test

Skipped. Step (a) shows the served vLLM has no code path that applies a
LoRA to the drafter. A random-adapter smoke test would only fail at startup
or silently ignore the adapter. Spending GPU on it would prove nothing new.

## What would justify building it

Run this sequence; each step is gated on the one before.

1. **Quality, with no serving change (0.5-1 day + about 2-4 GPUh per domain).**
   - Add PEFT wrapping to the speculators trainer for the warm-started e10
     head. Train an r16/r64 adapter on the same domain traces the full-head
     grid uses, then merge it offline into a full head.
   - Serve the merged head with the existing pipeline and `reload_drafter`
     swap. Compare against the full fine-tuned specialist and against e10.
   - Proceed only if the adapter keeps at least about 90% of the
     full-specialist *served* gain over e10.
2. **Need.** The full-head domain grid (A3, run by the other agents) must
   first show specialists that pass the existing rule: at least 10% more
   throughput **and** at least 10% lower median TPOT in both GPU-swapped
   repeats on their domain, without material loss on OPB/other domains.
   - Precedent: EAGLE multilingual +17% passed; DSpark multilingual +3-5%
     at k3 did not.
   - If only one or two domains pass, swapping full heads (0.3-0.6 s) or a
     routed replica per domain is cheaper than building adapter serving.
   - Adapters pay off only with several passing domains, mixed in the same
     traffic, where resident memory or batch mixing is binding (DSpark full
     head 3.2 GiB per domain versus about 16-63 MiB as an adapter).
3. **Serving patch (estimate 3-5 engineer-days, not tested).**
   - Give the proposer its own LoRA manager.
   - Pass the request-to-adapter mapping into the drafter forward,
     including DSpark block drafting and the confidence/Markov heads.
   - Handle CUDA-graph capture per active-adapter count, as the target
     path does (`gpu_model_runner.py:5730-5760`).
   - Verify bitwise-greedy correctness with the calibrated gate.
   - Alternatively, contribute to or track RFC #52038 (DFlash only so far;
     DSpark/EAGLE need the same plumbing).

Estimates are planning guesses, not measurements.

# Drafter updates: workload-specific adaptation and live swap (8B)

Branch `drafter-updates` (T5). Question from Yu (15 Oct): the drafter should keep
updating for a specific workload, together with the inference service. When and
where to update it is a scheduling question. This note measures three things:
how much served-workload data a drafter needs to recover acceptance on a weak
domain, how quickly it gets there, and whether a running server can take
updated drafter weights without a restart.

## Summary

PENDING: summary table (domain, traces, steps, simulated acceptance gain on
the domain, OPB forgetting, serving acceptance at k=7, swap pause).

## Method

### Stand-in served traces (step A)

There is no captured production traffic yet. Each weak domain gets a prompt
pool the e10 drafters never saw, disjoint from the 8B test workloads, and the
target answers it as a deployment would: plain vLLM, thinking off, Apertus
sampling (T 0.7, top-p 0.8, top-k 20). Each row is seeded from its id
(`generate_targets --temperature`). Builder:
`methods/eagle/apertus_eagle/build_adaptation_pools.py`. Job script:
`methods/eagle/launch/drafter-traces.sh`.

| domain | source | input tokens | pool | dev | test |
| --- | --- | --- | --- | --- | --- |
| chat | SFT-mix `train` rows, chat_qa / multilingual / instruction_following (the chat test workload uses the `heldout_test` rows of the same domains) | 128-2048 | 8,192 | 256 | existing `eagle-8b-test` chat |
| summarization | finepdfs-edu eng/deu/fra/ita, files 4-7 (the workloads read files 0-1), same instruction and document window as the test workload, EN/DE/FR/IT 40/20/20/20 | 2048-4096 | 8,192 | 256 | existing `eagle-8b-test` summarization |
| multilingual | Infomaniak-AI/speculators-multilingual-en-fr-de-it-es (revision b1d6938), fr/de/it/es rows only: translated open-perfectblend prompts. Pool from `train_final.json`; dev and a new 128-prompt test workload from `test_final.json`; source rows used in its test file are excluded from the pool in every language | 16-2048 | 8,192 | 256 | 128 (new) |

Every pool, dev and test row is checked against the committed test and
validation workloads by rendered prompt-token digest, by normalized text and
by 13-gram near-duplicates. The multilingual test and dev rows are also
protected against the pool. Pools are in hash order, so the first N rows nest
for every N. The text stays on the cluster; the manifest is in
`results/8b/drafter-updates/data/`.

Licences: SFT-mix components and finepdfs-edu (ODC-By) as for the existing
workloads. The Infomaniak set is "inherited from source dataset", the
open-perfectblend regeneration. It is used here for a measurement and is not
redistributed.

### Adaptation curve (step B)

`train_rollout` gained `model.init_from`: it loads a trained draft's weights
before the optimizer is built, so the optimizer state, schedule and step
count start fresh. The frozen embedding and lm_head must equal the current
target's. A run that already has `checkpoints/resume` resumes as before. It
also gained `dataset.extra_eval`, which evaluates further held-out sets at
every eval. Every fine-tune starts from e10's served checkpoint
(`e31-opb-thinkoff-stage1/checkpoints/best`, step 251,955). The settings:

- online target features, as e10;
- 32 rows per step on 1 node (4 ranks);
- constant learning rate 3e-5 after 19 warmup steps (e10 peaked at 1e-4 and
  decayed to 0); one arm uses 1e-4;
- 384 steps (12,288 rows: 24.6 epochs of 500 traces, 6.1 of 2k, 1.5 of 8k);
- evaluation every 32 steps on the domain dev traces (`eval_data`) and on the
  first 512 of e10's 2,000 open-perfectblend held-out rows (`extra_eval`
  "opb", the forgetting check);
- replay arms add N/4 open-perfectblend training rows (20% of the mix);
- the final checkpoint is exported.

With a constant learning rate, the eval trace of one run approximates the
curve over step budgets. Simulated acceptance (`simulated_acc_len`, 7 rollout
steps) is the trainer's cheap metric. It is not serving acceptance; the node
sweeps below measure that.

### Live swap (step D)

The fork's `update_weights` and `reload_weights` reload the target.
`src/apertus_bench/drafter_reload.py` is a vLLM worker extension
(`--worker-extension-cls apertus_bench.drafter_reload.DrafterReloadExtension`)
with `reload_drafter(path)`. Each worker reads the new head's safetensors and
passes them to the draft model's own `load_weights`, which applies the same
name mapping and sharding as at startup and copies into the existing
parameters. Parameter addresses must not move: CUDA graphs and compiled code
stay valid, and no KV cache or request state is touched. The engine core runs
the RPC between two steps.

The new head must meet two conditions, or the reload is refused before any
copy:

- tensors it shares with the target (the embedding, and the lm_head when the
  head's copy is frozen) must equal the target's; they are never written;
- its architecture keys must equal those of the served head.

The RPC is reached through the dev endpoint `/collective_rpc`
(`VLLM_SERVER_DEV_MODE=1`). In production, that endpoint must sit behind the
gateway's admin interface.

`src/apertus_bench/drafter_swap.py session` runs the measurement:

1. Serve the target with head A at k=7.
2. Run a greedy pass on 32 fixed prompts.
3. Run a closed-loop sampled load at fixed concurrency. Halfway, it swaps to
   head B and the load keeps running.
4. Run a greedy pass with B.
5. Swap back to A and run a greedy pass with A again. This last pass is the
   control: the same drafter, so its differences are numerical noise.

It records:

- each stream's event times (the stall across the swap);
- spec-decode counters every second (acceptance before and after);
- parameter digests of the drafter and of a target sample.

## Results

### Live swap smoke (job 3722022)

The served e10 head (`e31-opb-thinkoff-stage1-se`) was swapped under load to
the epoch-3 head of the same run (`-ep3-se`), which has the same architecture
and lower acceptance, and later swapped back. k=7, one GPU per arm, 360 s of
closed-loop sampled load, swap at 180 s. Record:
`results/8b/sweeps/drafter-swap-smoke-20261022T142432Z/`.

| | chat, C=32 | summarization, C=64 |
| --- | --- | --- |
| requests (errors) | 5,691 (0) | 2,631 (0) |
| swap RPC, under load | 0.56 s (read 0.49 s, copy 0.06 s) | 0.33 s (read 0.23 s, copy 0.06 s) |
| swap back (warm page cache) | 0.14 s | 0.14 s |
| longest stream gap across the swap | 0.56 s | 0.30 s |
| stream gap before the swap, p50 / p99 / max | 0.019 / 0.070 / 0.10 s | 0.037 / 0.26 / 0.41 s |
| acceptance length before → after | 2.47 → 2.42 | 2.18 → 2.14 |
| greedy exact match, post vs pre / back vs pre | 32/32 / 32/32 | 32/32 / 32/32 |
| drafter digest changed, then restored; target digest unchanged | yes | yes |

- **Pause:** the swap stalls every stream for about the RPC time, at most
  half a second, and no request is lost. A restart costs far more: this
  server took 207 s from launch to its first completion (load, compile,
  CUDA graph capture), and in-flight requests would be dropped or drained.
- **Where the time goes:** most of it is reading 0.5 GB of safetensors from
  /iopsstor with a cold page cache. Reading the file into host memory before
  the RPC would cut the stall to the copy (about 60 ms).
- **Shared tensors:** 15 tensors were loaded. The frozen lm_head in the new
  head was checked equal to the target's and not written.
- **Acceptance:** the drop after swapping to e3 is small but has the
  expected sign, so the new weights are live. Each window is about 150 s
  after a 30 s settle.
- **Shutdown error:** the `AsyncLLM output_handler failed` error in
  replica.out comes from teardown after SIGINT, once the session had
  finished.

## Design note: when and where to update

PENDING.

## Runs

| job | what | outcome |
| --- | --- | --- |
| 3721346 | trace smoke (debug) | failed: no pyarrow in the image (fixed in a33ee4f) |
| 3721372 | trace smoke (debug): 64/16/16 rows per domain, 16 generations per file | ok |
| 3722013 | pools (8,192 + 256 per domain, 128 multilingual test) and traces, Apertus sampling | pending |
| 3722022 | live swap smoke, e10 → e3 → e10, chat C=32 and summarization C=64 | ok (above) |
| 3725574 | fine-tune smoke: 4 steps from e10 on the smoke traces (`adapt-smoke.yaml`) | pending |

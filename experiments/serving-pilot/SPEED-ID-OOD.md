# DSpark e10 k=3: controlled speed comparison

This campaign compares speculation on and off on physical GPU 1 of the existing
Clariden allocation 3791982, node nid007257. The original Model Launch worker on
GPU 0 is preserved. The measurement servers are temporary loopback-only benchmark
engines, managed by the existing `apertus_bench.node_sweep.run_arm` launcher. No
new training or Slurm allocations are needed.

The active capture is `results/8b/speed/id-ood-20261030-c` on Clariden in
`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-speed-code`. Source revision is
`f586455fbad788f94e60ef5ba74892f112439fd0`; launch commands and controller logs
are in sibling directory `launch-20261030-c`. Attempts a and b failed before any
engine launched: the pilot environment does not mount `/users`, including the
workload files and linked-worktree Git directory. They are retained separately.
Inputs were copied unchanged to mounted scratch; Git identity is passed from
the host and recorded separately from the container harness's unavailable Git ID.

The existing target revision, image, exact source overlays and e10 head are reused
from the serving pilot's preserved environment. Each pair launches fresh baseline
and DSpark engines sequentially on the same physical GPU. Three deployment pairs
use AB, BA, AB order. Both use TP=1, bf16, k=3 for DSpark, 32,768 context, a
16,384-token scheduler budget, 256 maximum sequences, 80% GPU memory, disabled
prefix caching, identical thinking-off template, greedy decoding, seed 1 and a
256-token output cap. Warmups are excluded from measurements. Actual engine
logs must be checked for scheduler and cache differences.

The corpus is fixed before observing results. Select 32 prompts per stratum by
sorting source prompt IDs by SHA256 of `speed-20261030:<id>`:

- **ID:** the existing OpenPerfectBlend validation workload, constructed from
  the DSpark held-out split by `tools/make-opb-workload.py`. These are from the
  training distribution, rather than training examples. They have been used in
  prior evaluations; this is runtime confirmation, not a fresh generalization set.
- **OOD chat:** the existing test chat workload.
- **OOD summarization:** the existing long-document summarization workload.
  Long-document summarization is absent from the training mixture described in
  `docs/system-optimizations-8b.md`. OOD denotes the workload shift, not a claim
  that every topic or phrase is absent from the broad training corpus.

Each stratum runs at concurrency 1, 8 and 128 with natural EOS. A separate
mechanical check runs at concurrency 1 and 128 with exactly 256 output tokens
(`ignore_eos=true`). Fixed-length results are never pooled with natural EOS.
At C=128, 128 requests cover four complete passes over the 32-prompt stratum:
this is a burst measurement, not sustained capacity. The same ordered prompts,
request counts and generation settings must match in both arms.

Output tokens/s is the primary throughput metric; also retain E2E latency, TTFT,
TPOT, acceptance, exact token counts, failure rate and raw metric snapshots.
Compare natural output lengths with the fixed-length check before interpreting
a gain. Reject pairs with mismatched metadata, failed warmups, less than 99%
request success, inactive speculation, counter resets or incorrect fixed lengths.
The existing counter parser rejects decreases. The baseline must not speculate.

Report every paired speedup, geometric mean and exact percentile bootstrap 95%
interval over the three deployment pairs, not over individual requests. A gain
requires the full interval above 1.05; a slowdown requires it below 0.95. An
interval wholly inside [0.95, 1.05] means within 5%; otherwise the result is
inconclusive. Three pairs on a single node give limited uncertainty estimates;
they cannot establish universal speedup, production capacity or losslessness.

Generate the report after completion:

```bash
python3 experiments/serving-pilot/report-speed-id-ood.py \
  results/8b/speed/id-ood-20261030-c
```

## Discovery incident

The model was launched through Model Launch's native OpenTela wrapper with
`--dev`. Model Launch's dev default embeds peer ID
`QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9` at `148.187.108.177:43905`.
On 30 October, that endpoint instead presents
`QmTnRyTRyUs2UCpDJHAgSz9CVeYQL1K3KiW1MdTjKEN55z`. The bootstrap server's public
`http://148.187.108.177:8092/v1/dnt/bootstraps` independently advertises the latter.
Debug logs show TLS peer-ID mismatch, not a blocked TCP port. The original worker
logged `bootstrap_connected=false` and its DNT contained only itself.

A bounded ephemeral probe using the stale ID reproduced the failure; the same
probe using the advertised ID logged `bootstrap_connected=true` and
`Connectivity OK: bootstrap connected and relay reserved`. Neither probe served
a model or changed the original deployment. Raw probe commands, reports and logs
are retained in `results/8b/speed/discovery-20261030-{b,c}` on Clariden.

The pilot now exposes Model Launch's existing native override, preserving it in
the frozen profile and launch command:

```bash
./serving/research-pilot.sh prepare \
  --opentela-bootstrap-addr \
  /ip4/148.187.108.177/tcp/43905/p2p/QmTnRyTRyUs2UCpDJHAgSz9CVeYQL1K3KiW1MdTjKEN55z \
  --output /path/to/NEW/deployment
```

Verify the advertised identity again before a future launch. This override fixes
the demonstrated handshake error; actual gateway generation still needs a passing
request check. Dev deployments belong to the dev serving environment, rather than
`https://serving.swissai.svc.cscs.ch/`. No original deployment was restarted while
its allocation hosts these measurements.

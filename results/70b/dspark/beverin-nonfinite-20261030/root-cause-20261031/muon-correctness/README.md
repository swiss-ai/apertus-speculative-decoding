# Standalone DTensor Muon oracle

These tests compare the exact native Muon normalization and five-step
Newton–Schulz function with dense GPU references. No model, corpus, optimizer
training or learning-rate update is loaded. The unchanged probe uses cloned
BF16 inputs, forced rocBLAS and the production all-gather/chunk redistribution
workaround. It checks complete outputs, norm placements, finite values and
raw-update relative L2 error, so BF16 parameter rounding cannot hide an update.

The coefficients are `(3.4445, -4.7750, 2.0315)`, epsilon `1e-7`; raw-update
relative L2 tolerance is 5% and norm tolerance is 2%. Fixtures are zero, tiny
random, random, skewed random and two rank-one controls. The GPU path calls
the installed private function directly without substituting a measured norm.
Its canonical source hash is
`99e2359a95e1a914345207e49f072a658866333f4458b7b45faafa35c628f7c4`.

| Test | Shapes | Outcome |
|---|---|---|
| CPU preflight, pinned function on host 2.15 | `16x80` | Both ranks passed; rank-one errors zero |
| Native wide, job 710498 | `16x80`, `8192x40960` | Both images passed; maximum update error 0.236% |
| CPU tall and placement trace | `80x16` | Both ranks passed; final maximum update error 3.309% |
| Native tall, job 710518 | `80x16`, `43008x8192` | Original aborted; ALPS passed on the same GPUs |

The wide shapes never execute the initial transpose. Tall gate/up weights do:
row sharding becomes column sharding, Gram and Gram-update use Partial(sum),
and nonunit-beta addmm follows a distinct redistribution path. The native tall
comparison therefore addresses a real coverage gap. See the
[tall reproducer report](tall-only/README.md).

The original tall oracle suffered one observed illegal-instruction abort;
this is not a reproduction of the historical NaN or proof of a specific
production optimizer fault. The oracle adds reconstruction/output gathers,
and both native libraries and P2P/SHM transport differ between images. No
further job was launched. One-node production and Clariden were untouched.

The final CPU preflight is in `cpu-preflight-production/`, earlier preflights
remain separate, and the tall source/trace is in `cpu-tall/`. CPU execution
uses the pinned 2.13 function AST with host torch 2.15, not the complete 2.13
dispatcher. The exact pointwise strategy file matches both images and the CPU
host; matrix/placement/redistribution files match the two images but differ
from the host. See `source-audit/` for exact bytes and hashes.

Wide runtime `cdf9adb5c1c511b1aad5a1c9b306c2ba30b22563` and tall runtime
`0030d3123e3bf7b8fa4f6c41e3615ab57b45203c` remain immutable. Exact source,
launch argv, environment, image hashes, native function/strategy sources,
Slurm/node identity and raw results are co-located with each test. CPU norm
rounding can amplify through the five iterations; bitwise equality is not
required except where explicitly checked.

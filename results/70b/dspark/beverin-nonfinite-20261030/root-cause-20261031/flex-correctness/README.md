# Attention kernel correctness discriminator

One-node, one visible GPU, 20-minute hard allocation; no model, optimizer,
training data or distributed process group. Original and ALPS images run on
the same GPU with identical rocBLAS preference and separate compiler caches.
Six large cases per image use the
exact captured mask implementation and production Q4096/KV12288/head128/GQA64:8.
Each image has a six-minute timeout including container start/compilation.

All five production layers use sliding attention with window2048 and
sliding_window_non_causal=False (captured config). The synthetic block is
causal, so slot s sees s+1 synthetic tokens. The mask builder is compiled as
in production. Fixtures use two4096-token documents so window truncation is
exercised. CPU and small-reference fixtures scale the window and document size.

Fixtures use all selected anchors, then the exact padded-anchor convention:
sorted valid anchors followed by zeros. Every query has its own synthetic block
available; many base KV columns have no incoming attention. This checks a
specific sparse-mask forward/backward hypothesis rather than repeating training.

Three analytic controls run for each mask:

- Zero Q/K, constant V: uniform attention has a closed-form dV for every key.
- Random Q/K, zero V: output and Q/K gradients must be exactly zero.
- Saturated random Q/K, constant V: output remains one; gradients must be finite.

The six large cases check dV mass conservation, finite output/gradients and
output values. Two additional small cases (Q256/KV512/GQA8:2) compare random
Q/K/V and nonzero Q/K/V gradients against an independent FP32 SDPA math backend
with a dense mask, at 3% relative L2 tolerance. There are eight cases per image.
BF16 reference tolerances are explicit; tiny dense CPU mask oracles
verify the closed-form reference with and without padded anchors. These are
kernel tests, not a recreation of the missing failed batch, learned weights,
full compiled model, FSDP interactions or full training stability.

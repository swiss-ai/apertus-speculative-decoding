# Tall Muon transpose branch (prepared for final review, not submitted)

The captured checkpoint header confirms all five layers have gate/up
projection weights43008x8192, down8192x43008. Existing GPU job710498 checked
only wide16x80 and8192x40960 inputs. That does not execute the private
Muon's initial M>N transpose and resulting column-sharded Gram/reduction
path. Native tall testing is a different branch, not a repetition of wide.

The exact probe.py remains unchanged. Setting
`MUON_ORACLE_SHAPES="80x16 43008x8192"` selects tall-only inputs via the
inside.sh array. The setting and full argv are retained, and the batch
summary asserts the exact two shape values plus12cases on each rank.
Default shapes remain wide for reproducibility of the prior job.

The CPU2.15/Gloo tall oracle uses pinned2.13 function AST and all6fixtures
on2ranks. Both passed: rankone0rawNSerror, random3.309%, skewed3.017%, no
nonfinite. A separate tiny manual placement trace confirms the branch:
Shard0→transposeShard1→normalizedShard1→GramPartial(sum)→nonunit-beta
GramUpdatePartial(sum)→orthoUpdateShard1. Rankoneintermediatesexactdense;
random maximum intermediate relativeL2error4.666% and final3.309%. Raw
source, traces, native-source hashes, exact commands and results are in
cpu-tall/. CPUdispatch is not the native2.13 GPU runtime.

Planned resource bound: one node20minutes, two ranks GPUs2/3, original and
ALPS cells sequentially, each360seconds, no model/corpus/optimizertraining,
no automatic retry. Submit only after final parent review of the committed
runtime. After this targeted branch test, no further job is authorized.

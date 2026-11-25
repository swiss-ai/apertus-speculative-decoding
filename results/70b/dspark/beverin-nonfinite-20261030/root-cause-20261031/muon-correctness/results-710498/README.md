# Wide DTensor Muon oracle, job710498

Both original and ALPS image cells passed. Job710498 ran on nid002410 for
225seconds, completing0 on31October2026 at16:27:24UTC (start16:23:39).
There was no model, corpus, optimizer training or learning-rate update.
Runtimecdf9adb5c1c511b1aad5a1c9b306c2ba30b22563 is immutable.

Each image checked2ranks,6fixtures and2wide shapes(16x80,8192x40960),
12cases/rank;48rank-cases in total. All inputs reconstruct exactly and all
native private-function outputs remain finite. The production all-gather
redistribution patch and rocBLAS are active. Both complete image hashes were
asserted before cells. The reference is the same exact installed private
Muon function applied to an independent ordinary dense GPU clone.

| Recorded criterion | Original | ALPS |
|---|---:|---:|
| Passed rank-cases |24/24|24/24|
| Raw NS update max relativeL2error |0.00215939|0.00236260|
| Rank-one update errors |0|0|
| Max BF16 norm relative error |0.00636943|0.00636943|
| Nonfinite dense/DTensor output values |0|0|

Thus raw update errors are at most0.2363% (tolerance5%), with norm error
at most0.637% (tolerance2%). Rank-one outputs are bitwise equal. Independent
row-shard norm diagnostics have _NormPartial before clamp and Replicate
after clamp for all cases. The exact function source hash is identical:
99e2359a95e1a914345207e49f072a658866333f4458b7b45faafa35c628f7c4.

**Coverage is wide matrices only.** Neither shape executes M>N transpose.
Captured production gate/up weights are43008x8192; their transposed input
is column-sharded and Gram usesPartial(sum). A negative result here cannot
clear that branch,64-rank topology, model forward/backward, FSDP overlap,
large/rare input values or a long-lived allocator. No cause or ALPS fix for
the original519NaN is established. The separate tall-only test is recorded
under ../tall-only/ and uses a fresh job/directory.

All43raw files/719,865bytes match remote hashes in collection.json. Run
`python3 -B verify_results.py` to recheck every raw byte, case signature,
image hash, nonfinite count and recorded comparison criterion. Exact source,
versions, argv, runtime revision/branch, nonsecret environment, cell logs,
Slurm IDs and node identity are retained. sacct.txt uses cluster localUTC+1.
The first read-only SSH collection attempt timed out at55seconds without
an authentication error; a later bounded retry succeeded. This was a
collection transport delay, not a job/numerical failure.

# Job 710518 evidence

The original image suffered a native illegal-instruction abort; ALPS passed
on the same node and GPUs. The batch correctly ended FAILED with exit 1,
after 255 seconds. This is not a numerical-assertion failure or reproduction
of the historical training NaN.

The original has six completed small cases per rank and no completed large
case. ALPS has all 12 cases per rank, finite outputs and maximum relative L2
error 3.774% against its dense reference, below the 5% tolerance.

All 43 collected files (621,779 bytes) match `collection.json`. Four large
CPU/GPU cores were retained remotely and recorded by path and size under
`excluded_files`; they were not copied or fully hashed. Run
`python3 -B verify_results.py` to verify raw hashes, image identities, actual
tall argv, completed cases, abort signatures and the paired classification.
A successful verifier run means the evidence is consistent, not that the
original cell passed. `analysis.json` contains its computed findings.

See [the reproducer report](../README.md) and
[bounded core metadata](../core-metadata/README.md) for scope and limitations.
Slurm timestamps in `sacct.txt` use cluster local time, UTC+1.

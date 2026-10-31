# Exact image DTensor strategy sources

Files were read from immutable images with `unsquashfs -cat IMAGE PATH`;
image and internal path, byte count and SHA256 are retained in collection.json
(pointwise) and matrix-collection.json (matrix/placement/redistribution).
No container or job was modified. Upstream and ALPS installed files match
byte for byte for all four modules:

| Module | Bytes | SHA256 |
|---|---:|---|
|_pointwise_ops.py|19983|b25444743c8c707cf30aaa3e27a9b2d850a59f6309ed93eca4a466244b3a73db|
|_matrix_ops.py|50085|5fb482c6f2aff2cb38cd4a39455a0db3b0d441aa64af781b50c152083ca9dbaa|
|placement_types.py|71231|4a58b47fbacfdab0a02184d360240923ffccab7cdfef9d262349f0fbcb1bcbd5|
|_redistribute.py|86079|39ee270d5c24b77b81217daad22d9376817ba930fb19bf35f5e839bbeb607de9|

CPUtorch2.15 has the same pointwise file, but different matrix/placement/
redistribution source bytes (host-matrix-comparison.json). CPU preflight is
therefore not an execution of the entire image2.13 dispatcher. The source
identity between images also does not establish equal native libraries,
compiler output, execution order or correctness.

For source navigation, _matrix_ops.py301-327 derives addmm bias placement
from output (including cloningPartial), and419+ registers the current
single-dimension addmm strategy. placement_types.py1550+ documents and
implements sum partition(v)=v/world. Tiny CPU tall tracing observes
GramPartial(sum) and nonunit-betaGramUpdatePartial(sum). No source-level
mismatch was proven in these files.

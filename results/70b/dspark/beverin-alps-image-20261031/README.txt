ALPS vLLM ROCm image: Beverin DTensor comparison, 2026-10-31
=========================================================

Result
------
The ALPS image removes the large default-backend slowdown in the standalone
one-node reproducer. Successful job 710006 completed all four cases on
nid002346, GPUs 2,3, with two ranks and OMP_NUM_THREADS=1.

Sharded-operation warmed medians, milliseconds:

Image                         Default backend    Forced rocBLAS
Original vLLM v0.31 image          470.236559          0.395563
CSCS ALPS vLLM ROCm image            0.614610          0.423804

Each median pools two ranks and iterations 1 through 4: eight samples per
placement/case. Iteration 0 is excluded, matching the earlier comparison.
All 120 samples (three placements x five iterations x two ranks x four cases)
are present. Plain and replicated controls are in each cell's summary.json.

The image default reported _BlasBackend.Cublaslt in both images. Explicit
rocBLAS reported _BlasBackend.Cublas. These CUDA API names are retained on
ROCm. The operation still includes DTensor redistribution and synchronization.

This is a small synthetic performance test. It does not validate full DSpark
training, Muon/FSDP performance, model extraction, checkpoint saves/resume,
or multi-node operation. The entire software environment changes between
images, including network defaults; this comparison does not establish
which native component removed the slowdown.

Image identities
----------------
ALPS reference:
ghcr.io/eth-cscs/alps-extended-images/vllm-rocm@sha256:7d95232cbc7511be5672d272f7a567ea52a3c5fbd5310ce0903956f0f37b5cc7
Tag at resolution: alps7-dev
Image source commit: bf992658020317ca9c4564aa6593ddad7dc42240
Imported file:
/capstor/scratch/cscs/faruk_zahiragic/container-images/alps-vllm-rocm-7d95232cbc75.sqsh
Imported-file SHA-256:
416216c9de21d110fbb7cb12cef90923af51e04c955d7d7dffa9786f125a4928
Runtime: Python 3.12.3, torch 2.13.0+rocm10.0.0, torch.version.hip
7.15.26333, vLLM 0.31.0. Image environment reports ROCM_VERSION=10.0.0.

Original v0.31 control file:
/capstor/scratch/cscs/faruk_zahiragic/container-images/vllm-openai-rocm-v0.31.0.sqsh
SHA-256:
b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f
Runtime: torch 2.13.0+git733fca1, torch.version.hip 7.2.53211.

Full OCI manifests, configuration/labels, and the matching upstream source
archive (including image build recipes and patches) are under each
deployment's image/ directory. Package lists and resolved runtime settings
are retained per cell.

Launch details required by the ALPS image
---------------------------------------
1. Clear ROCR_VISIBLE_DEVICES, GPU_DEVICE_ORDINAL, and HIP_VISIBLE_DEVICES
   before setting CUDA_VISIBLE_DEVICES=2,3. The new ROCm image aborts on the
   contradictory inherited GPU_DEVICE_ORDINAL mask that the old image only
   warns about. All four successful cases used the same cleared masks.
2. Retain the ALPS image's built-in communication libraries by setting EDF
   annotations com.hooks.cxi.enabled=false and
   com.hooks.aws_ofi_nccl.enabled=false. netstack.source=host prevents the
   artifact mount; the disabled hooks prevent host library injection.
3. Use srun --network=disable_rdzv_get for the ALPS cells, as requested by
   the image's runtime diagnostic for its alt_read rendezvous configuration.
4. Use the image's normal Bash environment initialization. Its resolved
   settings include NCCL_P2P_DISABLE=1. This differs from the original
   image/control environment and is retained in the artifacts.

The ALPS raw logs confirm AWS Libfabric 1.20.0, libfabric 2.6, and provider
cxi with four NICs. The one-node result does not establish inter-node speed.
The native RCCL log also notes the host kernel lacks its newer cuMem support;
the probe completed successfully. No training or serving source overlay was
applied to either image.

Run ledger and provenance
-------------------------
Repository branch: experiment/beverin-alps-vllm-dtensor-20261031
Successful runtime commit: b66244a5c9ac797afee93eaa60d2981e9964d5ff
Original probe source: job 694440, unchanged probe.py/probe-rocblas.py.
Logical cluster run:
/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/alps-vllm-dtensor-20261031T110319Z/

709993, commit fa7e58e9: image import failed on a large GHCR layer with
  curl HTTP/2 PROTOCOL_ERROR and incomplete deflate data. No probe ran.
710005, commit cf5ddb6a: fetched the large layer in bounded HTTP/1.1 ranges,
  verified its original compressed SHA-256, and imported the image. Both
  original-image controls completed (491.136 ms default, 0.413913 ms
  rocBLAS); both ALPS cases stopped at startup on conflicting GPU masks.
710006, commit b66244a5: cleared the redundant masks for all cases and
  applied the ALPS Slurm network option. All four cases completed with
  exit code 0; Slurm COMPLETED, elapsed 00:04:11.

Each deployment is separate and retains its exact launcher, Git revision,
empty runtime patch/status records, job/node identities, image hashes,
commands, environments, raw logs, Python profiles, and timing summaries.
The failed attempts have not been replaced or overwritten. The two startup
core dumps remain on the cluster, indexed in core-files-left-on-cluster.json;
they are excluded from this small repository artifact copy.

source.bundle plus source-cf5ddb6a.bundle and source-b66244a5.bundle retain
the source transfer history (initial prerequisite commit 8084f4bc). Git
history in the repository also contains all three runtime commits.

Reproduction and verification
-----------------------------
The launcher and recording scripts are at:
methods/dspark/beverin/alps_compare/
Run run.sbatch from an isolated checkout of the recorded runtime commit,
passing the checkout path and a new run directory. Exact sbatch and srun
commands are in submit-command-*.txt and deployments/710006/cells/*/command.txt.
Use a new deployment directory and preserve the original results.

deployments/710006/summary.json is the complete successful comparison.
Each cell contains raw.log, samples.jsonl, summary.json, metadata.json,
runtime.json, environment.toml, command.txt, packages.txt, torch-config.txt,
and exit-code.txt. SHA256SUMS covers every file in this artifact snapshot
except the manifest itself; verify with sha256sum -c SHA256SUMS.

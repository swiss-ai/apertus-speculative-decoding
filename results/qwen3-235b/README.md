# Qwen3-235B-A22B speculative decoding (one GH200 node)

Plan, setup, results and verdict: `../../docs/qwen3-235b-specdec.md`.
Launcher: `serving/qwen3/submit-arm.sh`. Comparison: `experiments/qwen3-235b/compare.py`.

| Run | Job | Node | Arm | Status |
| --- | --- | --- | --- | --- |
| `smoke-20261122T171650Z-eagle3-k3` | 4004146 | nid005396 | EAGLE3 k3, 8 probe requests at C=1/8 | ok |
| `sweep-20261122T173201Z-plain` | 4004250 | nid006230 | no speculation | ok, 0 failures |
| `sweep-20261122T173203Z-eagle3-k3` | 4004251 | nid006230 | EAGLE3 k3 | ok, 0 failures |
| `sweep-20261122T173206Z-eagle3-k5` | 4004252 | nid006988 | EAGLE3 k5 | ok, 0 failures |

Each run directory keeps the following files. The cell directories are
`cells/<workload>/<workload>/c<C>/`; each holds `metadata.json`, `requests.jsonl`,
`summary.json` and the `metrics_*.prom` and `metrics_timeseries.jsonl` snapshots.

| File | Contents |
| --- | --- |
| `git-revision.txt`, `checkout.patch` | the code revision and any uncommitted diff |
| `launch-command.txt` | the exact `vllm serve` command |
| `submit-command.txt` | the submit command line and its settings |
| `job.sh`, `job.txt`, `job.out`, `slurm.txt` | the Slurm script, job ID, output, node and timing |
| `environment.toml`, `image.sha256` | the container environment and image digest |
| `model-identity.txt` | the pinned HF repositories and revisions |
| `target.sha256`, `drafter.sha256` | full-file SHA-256 manifests of the target and drafter |
| `workloads.sha256` | the prompt corpus digests |
| `runtime-versions.txt`, `pip-freeze.txt` | the torch and vLLM versions and the full package list |
| `server.log`, `bench-*.log` | the server and bench logs |
| `gpu-memory.csv` | 1 Hz `nvidia-smi` samples |
| `status.txt`, `ready-probe.json` | the per-workload outcome and the readiness check |

The weights are not committed. They stay on Clariden under
`/iopsstor/scratch/cscs/faruk_zahiragic/qwen3-235b-specdec/models/` and are verified
against the Hugging Face LFS hashes.

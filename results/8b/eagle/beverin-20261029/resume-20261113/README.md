# Resume of e31-opb-thinkon-stage1 on Beverin, 2026-11-13

The 10-link Beverin chain 707002..707011 (../submit-chain-20261030.sh) completed on
2026-11-10 06:34 at step 299,964 of 493,350 (epoch 6 of 10). Link 707011 stopped
cleanly at its deadline (`training-incomplete.json`). Nothing was queued after it.
The run stayed idle until this resume.

## Why a restored snapshot

On 2026-11-12 (directory mtimes about 16:09) most files in
`/iopsstor/scratch/cscs/faruk_zahiragic/beverin-eagle/repo-80d511ff` were deleted.
41 files survived, and they are the ones the run had read recently (logs,
`train_rollout.py`, `.pyc` caches). The deletion order looks like a scratch cleaner
working by access time. The `scratch/pydeps` and `scratch/TorchSpec` trees were
thinned as well, and `pydeps-x86_64-20261029b` is empty. Without
`apertus_eagle/__init__.py` and similar files, a resume from that snapshot cannot
import its code.

The same cleaner is also emptying old checkpoint directories in the run dir, oldest
first. As of 2026-11-13 18:00 CET, `final`, `cutover-clariden-173534` and
`step-012334` .. `step-172676` are empty. `resume`, `best` and `step-185010` ..
`step-296016` are intact. The damaged snapshot was left as it is: it holds the logs
of links 707002..707011.

## Backup (before the resume)

`/capstor/scratch/cscs/faruk_zahiragic/apertus-eagle-backup/e31-opb-thinkon-stage1-20261113/`
holds `checkpoints/resume` (step 299,964), `checkpoints/best` (step 296,040, eval
4.2505), the run JSON files and `history.jsonl`, with `SHA256SUMS`. The checksums
were compared against the source files after the copy.

## Restore, verified against the original records

- Code: `git archive 80d511ff` into `beverin-eagle/repo-80d511ff-restored-20261113`
  (`REVISION`). The surviving `train_rollout.py`, `train-eagle-on-node.sh`, the
  training config, `contract.json` and `start_guard.py` are byte-identical.
- TorchSpec: clone at pin 6c042a87 plus `apply-torchspec-patches.sh`.
  `torchspec-restored.patch` makes the same tracked-file changes as
  `../torchspec.patch`, and the file status matches `../torchspec-status.txt`. The
  untracked `renderers/apertus.py` is in the new diff only, because of `git add -N`.
- pydeps: `build-pydeps.sbatch`, job 739406. All 3,416 file hashes in
  `pydeps-meta/pydeps-files.sha256` are identical to `../pydeps-meta/pydeps-files.sha256`.
  The image hash and both pip freezes are identical as well.
- `/users` is at its 50 GB quota, so `SML_ENV_DIR` points at
  `beverin-eagle/sml-env`. `chain-sbatch/739422-739437.sbatch` differs from
  `../chain-sbatch/707002.sbatch` only in paths. `env_resolved_arm64_xPy2aT.toml`
  differs from the original only in `workdir`.

## Chain

`submit-chain-20261113.sh` submitted jobs 739422..739437: 16 links of 24 h, 2 mi300
nodes, rocBLAS, `afterany` chaining. The start guard passed: "resume step 299964,
last link stopped cleanly (job 707011)". The submit was dry-run first. Logs are in
`repo-80d511ff-restored-20261113/results/8b/eagle/logs/<jobid>.{out,err}`.

## First link

739422 logged `resumed` at step 299,964 with epoch 6, best 4.2505 at step 296,040,
and the optimizer state loaded. It reached step 300,002 after 359 s. The LR
continues where link 707011 stopped: 3.42442e-5 at step 299,964, then 3.42325e-5 at
step 300,002. The `get_name, Error when calling libdrm` lines from rocm-smi also
appear in 707011.out.

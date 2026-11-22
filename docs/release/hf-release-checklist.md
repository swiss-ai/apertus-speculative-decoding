# Hugging Face release checklist: Apertus 1.5 8B drafters

Status 2026-11-22: **prepared, not released. Blocked on the licence audit and on
leads approval.** Nothing in this checklist has been uploaded or made public.

This comes from the 2026-11-20 huddle (Rob). Before any release, present the
results to the leads (Rob suggested a slide deck and a meeting with
"probably Emmanuel Martin", as the transcript has it). If they approve, publish on the Hugging Face Hub,
and possibly on the CSCS inference platform. A prerequisite is that every
subpart of the training data is permissively licensed.

## Candidates

| Head | Role | Source of weights | `model.safetensors` sha256 |
|---|---|---|---|
| DSpark e10 thinking-off | Deployed head (`SwissAI-Research/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3`) | `/capstor/store/cscs/swissai/infra01/speculative-decoding/DSpark_Apertus_1.5_8B/dspark_apertus15-8b_open-perfectblend_thinking-off_2026-10-04_epoch10_final/` (Yu's store, not purgeable) | `002d1167be0cd49f0e3ecb04bb0978552936f04567d0ab7c7f5f2180b03eb56a` |
| EAGLE 3.1 e10 thinking-off | Comparison head, standard `eagle3` method | private repo `faruk-zahiragic/Apertus-v1.5-8B-EAGLE3.1-e10-thinking-off` @ `4c9d4c9e` (see warning below) | `42b0549629bed5366be812ea0a08897d3631bff2329bb7294adddd8ab100d6ff` |

**Warning: EAGLE weights have a single copy.** The Clariden scratch purge of
2026-11-20 01:45 emptied these directories:
- `apertus-eagle/8b/runs/e31-opb-thinkoff-stage1/checkpoints/{best,final,…}`
- `apertus-eagle/8b/heads/e31-opb-thinkoff-stage1-se`
- `hf-upload-*-e10-thinkoff`

The only known complete copy of the EAGLE e10 weights is now the private
pre-upload repo. Copy it to `/capstor/store` (not purged) before anything else.
The training records (provenance, summary, evals, history) also survive in the
Clariden home checkout and are now committed here, except `history.json`
(228 MB).

Staged layout in this repo (small files only; weights stay on the cluster):

```
release/apertus15-8b-drafters/
  dspark-e10-thinking-off/
    README.md              draft model card
    MANIFEST.json          every file of the HF repo: source, sha256, size
    config.json config.py run.yaml speculators.patch train_command.txt
    training_state.json val_metrics.json
    (+ at staging: model.safetensors, serving-patches/*.patch, SHA256SUMS)
  eagle31-e10-thinking-off/
    README.md MANIFEST.json config.json export-report.json
    configs/{contract.json,draft-e31-config.json,train-e31-opb-thinkoff.yaml}
    training/{provenance.json,train-summary.json,evals.json,best-checkpoint-state.json}
    (+ at staging: model.safetensors, training/history.json, serving-patches/*.patch, SHA256SUMS)
```

All committed hashes match the `SHA256SUMS` of the private pre-uploads
(2026-11-05/06) and the deployment record
(`results/8b/kubernetes/preparation-20261031-a/deployment-provenance.json`).

## Checklist

### Before the leads meeting

- [ ] **Licence audit resolved.** See
  [license-audit-8b-drafters.md](license-audit-8b-drafters.md). Verdict:
  **not clean**. ReClor (non-commercial) and TabMWP (CC BY-NC-SA) prompts are
  inside Open-PerfectBlend's UltraInteract slice: 11,727 rows, 0.83%. The leads
  choose one of:
  - (1) retrain on a filtered corpus;
  - (2) accept the residual risk;
  - (3) publish under a restrictive licence.

  If they choose (1), the candidates above are replaced by the retrained heads,
  and this checklist restarts with new hashes.
- [ ] Make a copy of the EAGLE e10 weights on `/capstor/store`, verified against
  `42b05496…`.
- [ ] Pending k3/k5/k7 headline run finished. Replace every `TODO(headline run)`
  in both cards.
- [ ] Slide deck with the results: per-task speedups, k grid, Arena-Hard,
  limitations, licence verdict.
- [ ] Leads meeting booked and held. Record the decision (date, attendees,
  outcome) here.

### Decisions needed from the leads

- [ ] **Hugging Face organisation and repository names: TBD.** Placeholders used
  below:
  - `<ORG>/Apertus-v1.5-8B-DSpark-thinking-off`
  - `<ORG>/Apertus-v1.5-8B-EAGLE3-thinking-off`

  Likely `<ORG>` = `swiss-ai`. The uploader account `faruk-zahiragic` is a
  member of `epfl-ml-ytf` and `epfl-ch`, **not** `swiss-ai`; an org admin must
  grant write access or do the upload.
- [ ] Re-upload, or transfer the existing private repos? HF can move a repo to
  an org and keep its history, but the history then includes the pre-audit
  card. Re-uploading from the staged folder gives a clean history.
  Recommendation: re-upload, then delete the private pre-uploads only after the
  public copies are verified.
- [ ] Licence field and wording in both cards (`TODO(leads)` markers).
- [ ] Gating. Recommendation: gate both repos (`gated="auto"`) with the Apertus
  v1.5 AUP, as the target is. EAGLE must be gated, because it contains the
  Apertus LM head.
- [ ] Release both heads, or DSpark only? DSpark is deployed and faster at low
  load. EAGLE uses the standard `eagle3` method and so needs fewer patches.
- [ ] CSCS inference platform: offer the deployed DSpark k=3 configuration
  publicly? That needs a separate request to the platform owners.
- [ ] Citation and authorship. Yu Zhu trained DSpark; Faruk Zahiragić trained
  EAGLE and ran the serving evaluation.

### Cards and files

- [ ] Replace every `TODO` in both `README.md` files:
  - links pinned to pushed commit SHAs (`docs/dspark-e10-task-speed.md` is on
    `experiment/dspark-e10-task-speed` @ `ca6717ae`; `docs/system-optimizations-8b.md`
    and this audit on `main` once merged);
  - the vLLM upstream note;
  - licence and citation.
- [ ] Decide whether to add the forced-thinking-off chat template (sha256
  `7c5a2b5a…`) as a usage example. It is a modified copy of the Apertus template
  (Apache-2.0).
- [ ] Re-run `tools/release/stage-8b-drafters.sh` (below). It must report `ok`
  for every manifest entry. The sha256 of the final `README.md` goes into the
  generated `SHA256SUMS`; record it in `MANIFEST.json` after staging.

### Stage (on Clariden, after approval)

```bash
# Clariden checkout of this branch, login node is fine (copies ~5 GB, no GPU)
tools/release/stage-8b-drafters.sh /iopsstor/scratch/cscs/faruk_zahiragic/hf-release-8b-drafters-<stamp>
```

It refuses to overwrite an existing staging directory, copies each manifest
entry, verifies its sha256 and writes `SHA256SUMS`. It does not upload.

### Upload: NOT EXECUTED; run only after written leads approval

Use `~/miniconda/bin/python` on Clariden (`huggingface_hub` 1.11, logged in as
`faruk-zahiragic`). The `hf` CLI there crashes (typer). Keep the HF caches off
the home directory, which is over quota.

```bash
STAGE=/iopsstor/scratch/cscs/faruk_zahiragic/hf-release-8b-drafters-<stamp>
ORG=<ORG>   # TBD by the leads
export HF_HUB_CACHE=$STAGE/.hf-cache HF_XET_CACHE=$STAGE/.hf-cache/xet
nice ~/miniconda/bin/python - "$STAGE" "$ORG" <<'EOF'
import sys
from huggingface_hub import HfApi
stage, org = sys.argv[1], sys.argv[2]
api = HfApi()
for local, name in [("dspark-e10-thinking-off", "Apertus-v1.5-8B-DSpark-thinking-off"),
                    ("eagle31-e10-thinking-off", "Apertus-v1.5-8B-EAGLE3-thinking-off")]:
    repo = f"{org}/{name}"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=False)
    api.update_repo_settings(repo, gated="auto")
    api.upload_folder(repo_id=repo, folder_path=f"{stage}/{local}",
                      ignore_patterns=[".hf-cache/*"],
                      commit_message="Release Apertus-v1.5-8B drafter (thinking off, epoch 10)")
    print(repo, api.model_info(repo).sha)
EOF
```

- The repos are created **private**. Flip them to public only after the
  verification below:
  `api.update_repo_settings(repo, private=False)`.
- Record the commit SHAs that the script prints.

### Verify after upload

- [ ] `api.model_info(repo, files_metadata=True)`: every LFS sha256 matches
  `SHA256SUMS`.
- [ ] Gate shows the AUP text. Test access from a second account.
- [ ] Fresh download, then serve with the card's exact command on one GH200
  (a debug-partition smoke). Acceptance on the 64-prompt probe must match the
  reports:
  - DSpark k=7: 6.68 accepted tokens per step (k=3: 3.75 on GSM8K);
  - EAGLE k=7: 6.52.
- [ ] Add the repo URLs and commit SHAs to this file and to the action-items
  canvas. Announce in #swiss-ai-spec-decode with the results-post format.
- [ ] Delete the private pre-upload repos `faruk-zahiragic/Apertus-v1.5-8B-*-e10-thinking-off`,
  but only after the capstor copy and the public copies are verified.

## Provenance index

| Artifact | Where |
|---|---|
| DSpark trainer provenance | `release/.../dspark-e10-thinking-off/{train_command.txt,run.yaml,speculators.patch,val_metrics.json,training_state.json}` (byte copies of Yu's store directory) |
| EAGLE training records | `release/.../eagle31-e10-thinking-off/training/` (+ `history.json` sha256 `ddbca514…` in the Clariden home checkout `results/8b/eagle/runs/e31-opb-thinkoff-stage1/`) |
| EAGLE export validation | `results/8b/eagle/heads/e31-opb-thinkoff-stage1-se/export-report.json` |
| Deployed DSpark engine argv, image digest, asset hashes | `results/8b/kubernetes/preparation-20261031-a/deployment-provenance.json` (branch `experiment/dspark-e10-task-speed`) |
| Serving patches | `serving/patches/vllm-apertus-{dspark-anchor-layout,dspark-mrv2-inner,eagle3-aux-layers,eagle-mrv2-inner,image-token}.patch` |
| Measurements | `docs/dspark-e10-task-speed.md`, `docs/system-optimizations-8b.md`, `docs/loadtest-8b.md`, `docs/benchmark-contamination.md` |
| Licence audit | `docs/release/license-audit-8b-drafters.md`, `docs/release/opb-subsource-counts.jsonl`, `tools/release/opb_subsource_counts.py` |

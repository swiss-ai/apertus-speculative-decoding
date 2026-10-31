#!/usr/bin/env bash
# Non-training correctness checks. Submit only after review, from immutable code.
#SBATCH --account=infra01
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --job-name=bev-comm-correctness
set -euo pipefail
repo=${1:?immutable committed source snapshot}
run=${2:?fresh result directory}
tools="$repo/results/70b/dspark/beverin-nonfinite-20261030/root-cause-20261031/comm-correctness"
mkdir -p "$run/deployments"
deployment="$run/deployments/$SLURM_JOB_ID"
mkdir "$deployment"
cp "$0" "$deployment/launch.sbatch"
cp "$tools/probe.py" "$tools/inside.sh" "$deployment/"
cp "$repo/REVISION" "$deployment/source-revision.txt"
cp "$repo/SOURCE_BRANCH" "$deployment/source-branch.txt"
export PROBE_SOURCE_REVISION PROBE_SOURCE_BRANCH
PROBE_SOURCE_REVISION=$(cat "$repo/REVISION")
PROBE_SOURCE_BRANCH=$(cat "$repo/SOURCE_BRANCH")
scontrol show job "$SLURM_JOB_ID" > "$deployment/slurm-job.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$deployment/nodes.txt"
export HEAD_NODE
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export SLURM_EXPORT_ENV=ALL
trap 'rc=$?; printf "%s\n" "$rc" > "$deployment/job-exit-code.txt"' EXIT
upstream=/capstor/scratch/cscs/faruk_zahiragic/container-images/vllm-openai-rocm-v0.31.0.sqsh
alps=/capstor/scratch/cscs/faruk_zahiragic/container-images/alps-vllm-rocm-7d95232cbc75.sqsh
sha256sum "$upstream" > "$deployment/upstream-image.sha256" &
upstream_hash_pid=$!
sha256sum "$alps" > "$deployment/alps-image.sha256" &
alps_hash_pid=$!
wait "$upstream_hash_pid"
wait "$alps_hash_pid"
read -r actual _ < "$deployment/upstream-image.sha256"
[[ "$actual" == b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f ]]
read -r actual _ < "$deployment/alps-image.sha256"
[[ "$actual" == 416216c9de21d110fbb7cb12cef90923af51e04c955d7d7dffa9786f125a4928 ]]
failed=0
for cell in upstream-default upstream-p2p-disabled alps-default; do
  cell_dir="$deployment/$cell"
  mkdir "$cell_dir"
  network=()
  if [[ "$cell" == alps-default ]]; then
    image="$alps"
    cp "$deployment/alps-image.sha256" "$cell_dir/image.sha256"
    annotations='"com.hooks.netstack.source" = "host"
"com.hooks.cxi.enabled" = "false"
"com.hooks.aws_ofi_nccl.enabled" = "false"'
    network=(--network=disable_rdzv_get)
  else
    image="$upstream"
    cp "$deployment/upstream-image.sha256" "$cell_dir/image.sha256"
    annotations='"com.hooks.aws_ofi_nccl.enabled" = "true"
"com.hooks.aws_ofi_nccl.variant" = "rocm6"'
  fi
  cat > "$cell_dir/environment.toml" <<EOF
image = "$image"
mounts = ["/capstor", "/iopsstor", "/users"]
workdir = "$deployment"
entrypoint = false
[env]
PYTHONNOUSERSITE = "1"
HWLOC_COMPONENTS = "-gl"
[annotations]
$annotations
EOF
  command=(srun "${network[@]}" --input=none --kill-on-bad-exit=1 -N2 -n2 --ntasks-per-node=1
    --environment="$cell_dir/environment.toml" --export=ALL
    bash "$tools/inside.sh" "$tools" "$cell_dir" "$cell")
  printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
  printf '\n' >> "$cell_dir/command.txt"
  rc=0
  timeout --signal=TERM --kill-after=20s 240s "${command[@]}" > "$cell_dir/raw.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$cell_dir/exit-code.txt"
  if [[ "$rc" != 0 ]]; then failed=1; fi
  printf 'CELL %s exit=%s UTC=%s\n' "$cell" "$rc" "$(date -u -Iseconds)"
done
python3 - "$deployment" <<'VERIFY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);summary={};failed=False
for name in ['upstream-default','upstream-p2p-disabled','alps-default']:
    cell=root/name
    reports=[json.loads(p.read_text()) for p in sorted(cell.glob('rank[0-3].json'))]
    code=int((cell/'exit-code.txt').read_text())
    passed=code==0 and len(reports)==4 and {r['rank'] for r in reports}==set(range(4)) and all(
        r['status']=='passed' and r['total_checks']==1536 and
        all(x['mismatches']==0 and x['nonfinite']==0 and not x['local_mismatch_detected']
            for x in r['checks']) for r in reports)
    summary[name]={'passed':passed,'exit_code':code,'rank_count':len(reports),
                   'ranks':[{'rank':r['rank'],'status':r['status'],'checks':len(r['checks'])} for r in reports]}
    failed=failed or not passed
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
raise SystemExit(1 if failed else 0)
VERIFY
exit "$failed"

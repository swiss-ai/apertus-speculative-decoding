#!/bin/bash
# Compare the original DTensor probe in upstream and digest-pinned ALPS images.
# Usage: sbatch --chdir=REPO --output=RUN/job-%j.out run.sbatch REPO RUN
#SBATCH -A infra01 -p mi300 -N 1 --gpus-per-node=4 --ntasks-per-node=1
#SBATCH -t 01:00:00 --nice=10000 --job-name=bev-alps-dtensor
set -euo pipefail
repo=${1:?isolated committed source checkout required}
run=${2:?new logical run directory required}
tools="$repo/methods/dspark/beverin/alps_compare"
deployment="$run/deployments/$SLURM_JOB_ID"
mkdir -p "$run/deployments"
mkdir "$deployment"
cp "$0" "$deployment/launch.sbatch"
git -C "$repo" rev-parse HEAD > "$deployment/git-revision.txt"
git -C "$repo" branch --show-current > "$deployment/git-branch.txt"
git -C "$repo" diff HEAD > "$deployment/source.patch"
git -C "$repo" status --porcelain > "$deployment/git-status.txt"
scontrol show job "$SLURM_JOB_ID" > "$deployment/slurm-job.txt"
hostname > "$deployment/hostname.txt"
uname -a > "$deployment/host-kernel.txt"
cat /sys/module/amdgpu/version > "$deployment/amdgpu-version.txt" 2>/dev/null || true
lscpu > "$deployment/lscpu.txt"
df -h /dev/shm /ritom /capstor > "$deployment/filesystems.txt"
cp -a "$tools/image" "$deployment/image"
mkdir "$deployment/enroot-hooks"
cp /etc/enroot/hooks.d/10-netstack.sh /etc/enroot/hooks.d/89-libfabric-cxi.sh \
    /etc/enroot/hooks.d/90-aws-ofi-nccl.sh "$deployment/enroot-hooks/"
cp "$repo/results/70b/dspark/beverin-20261025/694440/probe.py" "$deployment/probe.py"
cp "$repo/results/70b/dspark/beverin-20261025/694440/probe-rocblas.py" "$deployment/probe-rocblas.py"
trap 'rc=$?; printf "%s\n" "$rc" > "$deployment/job-exit-code.txt"' EXIT
printf 'Started %s on %s at %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)"

image=/capstor/scratch/cscs/faruk_zahiragic/container-images/alps-vllm-rocm-7d95232cbc75.sqsh
upstream=/capstor/scratch/cscs/faruk_zahiragic/container-images/vllm-openai-rocm-v0.31.0.sqsh
reference=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["reference"])' "$tools/image/identity.json")
printf 'enroot import -o %q %q\n' "${image}.part-${SLURM_JOB_ID}" "docker://${reference}" > "$deployment/import-command.txt"
if [[ ! -f "$image" ]]; then
    parts=/capstor/scratch/cscs/faruk_zahiragic/container-images/alps-layer-parts
    cache=/ritom/scratch/cscs/faruk_zahiragic/.enroot
    printf 'python3 %q %q %q %q\n' "$tools/prefetch.py" "$tools/image/manifest.json" "$parts" "$cache" > "$deployment/prefetch-command.txt"
    python3 "$tools/prefetch.py" "$tools/image/manifest.json" "$parts" "$cache" > "$deployment/prefetch.log" 2>&1
    ENROOT_MAX_PROCESSORS=32 ENROOT_MAX_CONNECTIONS=4 \
        enroot import -o "${image}.part-${SLURM_JOB_ID}" "docker://${reference}" > "$deployment/import.log" 2>&1
    test ! -e "$image"
    mv -n "${image}.part-${SLURM_JOB_ID}" "$image"
else
    printf 'Reused existing digest-named image %s\n' "$image" > "$deployment/import.log"
fi
printf 'Image import complete at %s\n' "$(date -u +%FT%TZ)"
sha256sum "$image" > "$deployment/alps-image.sha256" &
hash_pid=$!
sha256sum "$upstream" > "$deployment/upstream-image.sha256"
wait "$hash_pid"
python3 "$tools/records.py" prepare "$deployment" "$repo" "$image" "$upstream"

export SLURM_EXPORT_ENV=ALL
failed=0
for cell in upstream-default upstream-rocblas alps-default alps-rocblas; do
    cell_dir="$deployment/cells/$cell"
    printf 'Running %s at %s\n' "$cell" "$(date -u +%FT%TZ)"
    command=(srun --input=none --kill-on-bad-exit=1 -N1 -n1 --ntasks-per-node=1
        --environment="$cell_dir/environment.toml" --export=ALL
        bash "$tools/inside.sh" "$tools" "$cell_dir" "$deployment")
    printf '%q ' "${command[@]}" > "$cell_dir/command.txt"
    printf '\n' >> "$cell_dir/command.txt"
    set +e
    timeout --signal=TERM --kill-after=30s 300s "${command[@]}" > "$cell_dir/raw.log" 2>&1
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$cell_dir/exit-code.txt"
    if [[ "$rc" != 0 ]]; then failed=1; fi
    printf 'Completed %s with exit code %s\n' "$cell" "$rc"
done
python3 "$tools/records.py" summarize "$deployment"
exit "$failed"

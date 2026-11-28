#!/usr/bin/env bash
# Stage this committed campaign and an already rendered public prompt set on Clariden.
set -euo pipefail
prompts=${1:?provide prompts.json from the pinned DAPO extraction}
rev=$(git rev-parse HEAD)
branch=$(git branch --show-current)
case "$branch" in
    experiment/public-hybrid-recovery) study=public-hybrid-recovery; job=public-gdn-rollout ;;
    experiment/public-hybrid-queue-pressure|archive/qwen-hybrid-queue-pressure) study=public-queue-pressure; job=public-gdn-queue ;;
    *) echo "Unexpected experiment branch: $branch" >&2; exit 1 ;;
esac
remote_user=$(ssh clariden1 id -un)
[[ $remote_user =~ ^[a-zA-Z0-9._-]+$ ]]
dst=/iopsstor/scratch/cscs/$remote_user/gdn-public-$rev
ssh clariden1 "mkdir -p $dst"
git archive HEAD experiments/public-hybrid-recovery "experiments/$study" | ssh clariden1 "tar -x -C $dst"
ssh clariden1 "test ! -e '$dst/prompts.json' && cat > '$dst/prompts.json' || cmp - '$dst/prompts.json'" < "$prompts"
rotation=${GDN_ROTATE:-0}
[[ $rotation =~ ^[0-3]$ ]]
ssh clariden1 "cd $dst && GDN_BRANCH=$branch GDN_SOURCE=$rev GDN_ROTATE=$rotation GDN_STUDY=$study sbatch --job-name=$job --parsable --output=launch-%j.out experiments/public-hybrid-recovery/run.sh"

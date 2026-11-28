#!/bin/bash
#SBATCH --account=d_research_serving
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --job-name=public-gdn-rollout
#SBATCH --time=00:40:00
set -euo pipefail
: "${GDN_SOURCE:?source commit}" "${GDN_BRANCH:?source branch}"
base=$PWD/experiments/public-hybrid-recovery
code=$PWD/experiments/${GDN_STUDY:-public-hybrid-recovery}
run=$PWD/run-$SLURM_JOB_ID
model=/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.5-9B
mkdir "$run"
cp "$base"/{arm.sh,run.sh,submit.sh,environment.toml} "$code/groups.py" prompts.json "$run/"
if [[ $code != "$base" ]]; then
    cp "$code"/{configure.sh,cache_budget.py} "$run/"
fi
cd "$run"
printf '%s\n' "$GDN_SOURCE" "$GDN_BRANCH" > source.txt
scontrol show job "$SLURM_JOB_ID" > slurm.txt
nvidia-smi -q > gpu.txt
printf '%s\n' "${GDN_ROTATE:-0}" > gpu-rotation.txt
sha256sum prompts.json *.sh *.py *.toml > source.sha256
image_path=$(awk -F'"' '/^image = / {print $2}' environment.toml)
sha256sum "$image_path" "$model"/*.safetensors "$model"/*.json "$model"/*.jinja > runtime.sha256 &
hashpid=$!
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used --format=csv,nounits -lms 200 > memory.csv &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
srun --kill-on-bad-exit=0 --cpu-bind=cores --environment="$PWD/environment.toml" --output="$PWD/rank-%t.log" bash "$PWD/arm.sh" "$PWD" "$PWD" "$model" || status=$?
wait "$hashpid"
exit "${status:-0}"

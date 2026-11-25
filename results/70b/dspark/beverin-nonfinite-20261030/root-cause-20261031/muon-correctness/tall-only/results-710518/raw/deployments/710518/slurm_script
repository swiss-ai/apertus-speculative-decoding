#!/usr/bin/env bash
# Two ranks, no model/optimizer/training. Compare raw DTensor Muon to dense.
#SBATCH --account=infra01
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --job-name=bev-muon-correctness
set -euo pipefail
repo=${1:?immutable committed snapshot} run=${2:?fresh result directory}
base="$repo/results/70b/dspark/beverin-nonfinite-20261030/root-cause-20261031/muon-correctness"
deployment="$run/deployments/$SLURM_JOB_ID"
mkdir -p "$deployment"
cp "$0" "$base/probe.py" "$base/inside.sh" "$deployment/"
cp "$repo/REVISION" "$repo/SOURCE_BRANCH" "$deployment/"
scontrol show job "$SLURM_JOB_ID" > "$deployment/slurm-job.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$deployment/nodes.txt"
uname -a > "$deployment/host-kernel.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$deployment/job-exit-code.txt"' EXIT
failed=0
for cell in upstream alps; do
  output="$deployment/$cell"
  mkdir "$output"
  network=()
  if [[ "$cell" == alps ]]; then
    image=/capstor/scratch/cscs/faruk_zahiragic/container-images/alps-vllm-rocm-7d95232cbc75.sqsh
    expected=416216c9de21d110fbb7cb12cef90923af51e04c955d7d7dffa9786f125a4928
    annotations='"com.hooks.netstack.source" = "host"
"com.hooks.cxi.enabled" = "false"
"com.hooks.aws_ofi_nccl.enabled" = "false"'
    network=(--network=disable_rdzv_get)
  else
    image=/capstor/scratch/cscs/faruk_zahiragic/container-images/vllm-openai-rocm-v0.31.0.sqsh
    expected=b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f
    annotations='"com.hooks.aws_ofi_nccl.enabled" = "true"
"com.hooks.aws_ofi_nccl.variant" = "rocm6"'
  fi
  cat > "$output/environment.toml" <<EOF
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
  sha256sum "$image" > "$output/image.sha256"
  read -r actual _ < "$output/image.sha256"
  [[ "$actual" == "$expected" ]]
  command=(srun "${network[@]}" --input=none --kill-on-bad-exit=1 -N1 -n1
    --environment="$output/environment.toml" --export=ALL
    bash "$base/inside.sh" "$repo" "$output")
  printf '%q ' "${command[@]}" > "$output/command.txt"
  printf '\n' >> "$output/command.txt"
  rc=0
  timeout --signal=TERM --kill-after=20s 360s "${command[@]}" > "$output/raw.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$output/exit-code.txt"
  if [[ "$rc" != 0 ]]; then failed=1; fi
done
python3 - "$deployment" <<'VERIFY'
import json,sys,os
from pathlib import Path
root=Path(sys.argv[1]); summary={}
expected_shapes={tuple(map(int,value.split('x'))) for value in os.environ.get('MUON_ORACLE_SHAPES','16x80 8192x40960').split()}
assert len(expected_shapes)==2
for name in ['upstream','alps']:
    cell=root/name
    code=int((cell/'exit-code.txt').read_text())
    paths=sorted(cell.glob('rank*.json'))
    reports=[json.loads(path.read_text()) for path in paths]
    passed=(code==0 and len(reports)==2 and sorted(r.get('rank') for r in reports)==[0,1]
            and all(r.get('status')=='passed' and len(r.get('cases',[]))==12
                    and {tuple(c.get('shape',[])) for c in r['cases']}==expected_shapes
                    and all(c.get('status')=='passed' and c.get('input_nonfinite')==0
                            and c.get('input_mismatches')==0
                            and all(v==0 for v in c.get('nonfinite',{}).values())
                            and c.get('relative_l2_error',float('inf'))<=.05
                            and c.get('norm_relative_error',float('inf'))<=.02
                            and c.get('clamp_relative_error',float('inf'))<=.02
                            for c in r['cases']) for r in reports))
    summary[name]={'passed':passed,'exit_code':code,'rank_count':len(reports),
                   'cases_per_rank':[len(r.get('cases',[])) for r in reports],
                   'expected_shapes':[list(shape) for shape in sorted(expected_shapes)]}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary)); raise SystemExit(0 if all(r['passed'] for r in summary.values()) else 1)
VERIFY
exit "$failed"

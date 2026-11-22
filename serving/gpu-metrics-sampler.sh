#!/usr/bin/env bash
# Node-wide GPU hardware counters for a whole sweep job (SWEEP_GPU_METRICS=1 in
# serving/node-sweep.sh): Nsight Systems GPU-metrics sampling of every GPU
# (SMs Active, Tensor Active, DRAM bandwidth, ...) with no CUDA tracing, so the
# served processes run unprofiled. Sampling is device-wide, independent of
# which process uses the GPU, and runs in chunks so a job killed at its time
# limit loses at most the chunk in progress. apertus_bench.gpu_metrics turns
# the reports into per-second means and per-cell averages.
#   gpu-metrics-sampler.sh OUT_DIR [FREQUENCY_HZ] [CHUNK_SECONDS]
set -uo pipefail
OUT="${1:?usage: gpu-metrics-sampler.sh OUT_DIR [FREQUENCY_HZ] [CHUNK_SECONDS]}"
FREQUENCY="${2:-20}"
CHUNK="${3:-600}"
NSYS="${NSYS:-/iopsstor/scratch/cscs/faruk_zahiragic/pydeps/nsight-systems-2025.6.3/opt/nvidia/nsight-systems/2025.6.3/target-linux-sbsa-armv8/nsys}"
mkdir -p "${OUT}"
# Physical index <-> PCI bus of each GPU, to map nsys GPU ids to the indices
# the arms were given (CUDA_VISIBLE_DEVICES).
nvidia-smi --query-gpu=index,pci.bus_id,uuid,name --format=csv > "${OUT}/gpus.csv" 2>&1
"${NSYS}" --version > "${OUT}/nsys-version.txt" 2>&1
child=""
stop() {
  [ -n "${child}" ] && kill -INT "${child}" 2> /dev/null && wait "${child}"
  exit 0
}
trap stop TERM INT
number=0
while true; do
  number=$((number + 1))
  name="$(printf 'chunk-%03d' "${number}")"
  echo "${name} launched_unix=$(date +%s.%N)" >> "${OUT}/chunks.txt"
  "${NSYS}" profile --trace none --sample none --cpuctxsw none \
    --gpu-metrics-devices all --gpu-metrics-frequency "${FREQUENCY}" \
    --duration "${CHUNK}" --force-overwrite true --output "${OUT}/${name}" \
    sleep "$((CHUNK + 120))" >> "${OUT}/sampler.log" 2>&1 &
  child=$!
  wait "${child}"
  echo "${name} exit=$? ended_unix=$(date +%s.%N)" >> "${OUT}/chunks.txt"
done

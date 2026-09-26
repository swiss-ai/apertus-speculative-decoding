#!/usr/bin/env bash
# Upstream TorchSpec route (offline.generate + train_entry). Needs Ray and
# Mooncake in the training environment, which the serving image lacks; the
# active 8B path is launch/train-eagle-on-node.sh. Teacher/trainer GPU counts
# and aux layers come from the environment and the selected contract.
set -euo pipefail

if [ "${I_AM_ON_AN_ALLOCATED_GPU_JOB:-0}" != "1" ]; then
  echo "refusing to start a teacher / EAGLE training on this shell" >&2
  echo "set I_AM_ON_AN_ALLOCATED_GPU_JOB=1 after A2 parity passes" >&2
  exit 2
fi

: "${EAGLE_TRAIN_CONFIG:?}"
: "${EAGLE_REPLAY_DIR:?}"
: "${EAGLE_TRAIN_OUTPUT:?}"
: "${APERTUS_EAGLE_CONTRACT:?set the target contract}"
TEACHER_GPUS="${EAGLE_TEACHER_GPUS:?set EAGLE_TEACHER_GPUS (8B: 1, 70B: 4)}"
TRAINER_GPUS="${EAGLE_TRAINER_GPUS:?set EAGLE_TRAINER_GPUS}"

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${LAUNCH_DIR}/.." && pwd)"
TORCHSPEC_ROOT="${TORCHSPEC_ROOT:-${REPO_ROOT}/scratch/TorchSpec}"
export PYTHONPATH="${REPO_ROOT}/training:${TORCHSPEC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

TEACHER_ENGINE="${EAGLE_TEACHER_ENGINE:-vllm}"
AUX_LAYERS="$(python3 -c 'import json,os; from apertus_eagle.contract import aux_layer_ids, load_contract; print(json.dumps(aux_layer_ids(load_contract(os.environ["APERTUS_EAGLE_CONTRACT"]), "hf")).replace(" ", ""))')"
echo "teacher_gpus=${TEACHER_GPUS} trainer_gpus=${TRAINER_GPUS} aux_hf=${AUX_LAYERS} config=${EAGLE_TRAIN_CONFIG}"

if [ "${EAGLE_SKIP_GENERATE:-0}" != "1" ] && [ ! -f "${EAGLE_REPLAY_DIR}/dataset.json" ]; then
  python3 -m torchspec.offline.generate \
    --config "${EAGLE_TRAIN_CONFIG}" --output "${EAGLE_REPLAY_DIR}" \
    inference.inference_engine_type="${TEACHER_ENGINE}" \
    inference.inference_num_gpus="${TEACHER_GPUS}" \
    inference.inference_num_gpus_per_engine="${TEACHER_GPUS}" \
    inference.inference_num_gpus_per_node="${TEACHER_GPUS}" \
    inference.vllm.tp_size="${TEACHER_GPUS}" \
    inference.aux_hidden_states_layers="${AUX_LAYERS}"
else
  echo "skipping teacher materialization; replay at ${EAGLE_REPLAY_DIR}"
fi

python3 -m torchspec.train_entry \
  --config "${EAGLE_TRAIN_CONFIG}" \
  inference.inference_engine_type=offline \
  inference.offline.data_path="${EAGLE_REPLAY_DIR}" \
  inference.offline.num_engines=1 \
  training.training_num_gpus_per_node="${TRAINER_GPUS}" \
  output_dir="${EAGLE_TRAIN_OUTPUT}"

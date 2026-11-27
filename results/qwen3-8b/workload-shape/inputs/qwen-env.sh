# Sourced on the login node before submit-shape.sh (Qwen3-8B workload-shape run).
W=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b
export TARGET=$W/models/Qwen--Qwen3-8B
export EAGLE_HEAD=$W/models/RedHatAI--Qwen3-8B-speculator.eagle3
export DSPARK=$W/models/RedHatAI--Qwen3-8B-speculator.dflash
export CORPUS_FROM=/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape/runs/full-20261122/inputs
export SHAPE_ARGS="--eagle-format speculators --block-format dflash --names dflash,eagle3 --depth 15 --dspark-hf-layers 1,9,17,25,33"
export AGG_ARGS="--sampled"
export HF_HOME=$W/hf_home
export SML_ENV_DIR=$W/sml-env
export SHAPE_JOB_NAME=shape-qwen3-8b
export TREE_PYDEPS=/users/faruk_zahiragic/apertus-spec-decoding/scratch/pydeps:/iopsstor/scratch/cscs/faruk_zahiragic/workload-shape-qwen3-8b/pydeps
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

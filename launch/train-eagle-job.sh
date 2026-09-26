#!/usr/bin/env bash
# Superseded. This body hardcoded the 70B overfit configs and pip-installed into
# the container. Submit with an explicit config instead:
#   EAGLE_TRAIN_CONFIG=configs/eagle/8b/train-e31-overfit.yaml ./launch/submit-eagle-train.sh
echo "launch/train-eagle-job.sh is superseded by launch/submit-eagle-train.sh" >&2
exit 2

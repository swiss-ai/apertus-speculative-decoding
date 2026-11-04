# Sourced by train-node.sh. Speculators accepts exactly one decoder-definition
# source: a checkpoint already carries its depth, so --num-layers must only be
# sent when synthesizing a fresh draft. Passing it with --from-pretrained aborts
# before training (the original depth-l1-warm job 3722177).
draft_init_args=()
if [ -n "${FROM_PRETRAINED:-}" ]; then
  draft_init_args=(--from-pretrained "${FROM_PRETRAINED}")
else
  draft_init_args=(--num-layers "${NUM_LAYERS:-5}")
fi

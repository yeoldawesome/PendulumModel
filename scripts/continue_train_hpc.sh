#!/bin/bash
# continue_train_hpc.sh
# Usage: bash scripts/continue_train_hpc.sh <model_prefix> <additional_episodes> [extra_train_args]
# Example: bash scripts/continue_train_hpc.sh model_inverteddoublependulum_20260412_212409_ep5000 50000

set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <model_prefix> <additional_episodes> [extra_train_args]"
  exit 1
fi

MODEL_PREFIX="$1"
ADDITIONAL_EPISODES="$2"
shift 2
EXTRA_ARGS="$@"

ARTIFACTS_DIR="artifacts"
METADATA_PATH="$ARTIFACTS_DIR/${MODEL_PREFIX}_metadata.json"

if [ ! -f "$METADATA_PATH" ]; then
  echo "Metadata file not found: $METADATA_PATH"
  exit 1
fi

# Extract config from metadata.json
get_json_value() {
  python3 -c "import sys, json; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"
}

ENV_ID=$(get_json_value "$METADATA_PATH" env)
SEED=$(get_json_value "$METADATA_PATH" seed)
ACTOR_LR=$(get_json_value "$METADATA_PATH" actor_lr)
CRITIC_LR=$(get_json_value "$METADATA_PATH" critic_lr)
TAU=$(get_json_value "$METADATA_PATH" tau)
BUFFER_CAPACITY=$(get_json_value "$METADATA_PATH" buffer_capacity)
BATCH_SIZE=$(get_json_value "$METADATA_PATH" batch_size)
NOISE_START=$(get_json_value "$METADATA_PATH" noise_start)
NOISE_END=$(get_json_value "$METADATA_PATH" noise_end)
NOISE_DECAY_EPISODES=$(get_json_value "$METADATA_PATH" noise_decay_episodes)
MAX_STEPS_PER_EPISODE=$(get_json_value "$METADATA_PATH" max_steps_per_episode)

# Use a new run prefix for continued training
RUN_STAMP=$(date +"%Y%m%d_%H%M%S")
NEW_PREFIX="${MODEL_PREFIX}_cont_${RUN_STAMP}_ep${ADDITIONAL_EPISODES}"

# Copy weights to new prefix for continued training
cp "$ARTIFACTS_DIR/${MODEL_PREFIX}_actor.weights.h5" "$ARTIFACTS_DIR/${NEW_PREFIX}_actor.weights.h5"
cp "$ARTIFACTS_DIR/${MODEL_PREFIX}_critic.weights.h5" "$ARTIFACTS_DIR/${NEW_PREFIX}_critic.weights.h5"
cp "$ARTIFACTS_DIR/${MODEL_PREFIX}_target_actor.weights.h5" "$ARTIFACTS_DIR/${NEW_PREFIX}_target_actor.weights.h5"
cp "$ARTIFACTS_DIR/${MODEL_PREFIX}_target_critic.weights.h5" "$ARTIFACTS_DIR/${NEW_PREFIX}_target_critic.weights.h5"

# Submit a SLURM job using train_hpc.slurm with the continued run prefix and all config
sbatch scripts/train_hpc.slurm \
  --episodes "$ADDITIONAL_EPISODES" \
  --env-id "$ENV_ID" \
  --output-dir "$ARTIFACTS_DIR" \
  --seed "$SEED" \
  --actor-lr "$ACTOR_LR" \
  --critic-lr "$CRITIC_LR" \
  --tau "$TAU" \
  --buffer-capacity "$BUFFER_CAPACITY" \
  --batch-size "$BATCH_SIZE" \
  --noise-start "$NOISE_START" \
  --noise-end "$NOISE_END" \
  --noise-decay-episodes "$NOISE_DECAY_EPISODES" \
  --max-steps-per-episode "$MAX_STEPS_PER_EPISODE" \
  --checkpoint-name "$NEW_PREFIX" \
  $EXTRA_ARGS

echo "Submitted SLURM job for continued training. New run prefix: $NEW_PREFIX"

#!/bin/bash

# continue_train.sh
# Usage: bash continue_train.sh <model_prefix> <additional_episodes> [extra_train_args]
# Example: bash continue_train.sh model_pendulum_20260408_173452_ep5000 500 --env-id Pendulum-v1

set -e

# === HPC ENVIRONMENT SETUP (matches train_hpc.slurm) ===
module purge
module load python/3.11

WORKDIR="/work/classtmp/$USER/ddpg-pendulum"
VENV_DIR="$WORKDIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Ensure dependencies are installed
pip install --upgrade pip
pip install -r requirements.txt

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

# Extract env, seed, and other config from metadata.json
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

# Call train.py with --episodes, --env-id, --output-dir, --seed, and checkpoint name
python3 train.py \
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

echo "Continued training complete. New run prefix: $NEW_PREFIX"

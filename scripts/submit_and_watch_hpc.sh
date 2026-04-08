#!/bin/bash
set -euo pipefail

# Usage:
# bash scripts/submit_and_watch_hpc.sh
# bash scripts/submit_and_watch_hpc.sh --episodes 150 --email yournetid@iastate.edu

EMAIL="dnlong5@iastate.edu"
ACCOUNT="s2026.se.4390.01"
PARTITION="instruction"
EPISODES="100"
OUTPUT_DIR="artifacts"
SEED="42"
MAX_STEPS_PER_EPISODE="200"
NUM_ENVS="1"
BATCH_SIZE="256"
GPUS="1"
CPUS="6"
CPUS_SET_BY_USER="0"
SCRIPT_PATH="scripts/train_hpc.slurm"
AUTO_PUSH="1"
PUSH_BRANCH=""
PUSH_REMOTE="origin"
STRICT_PUSH="0"
GIT_USER_NAME="yeoldawesome"
GIT_USER_EMAIL="dnlonglett@gmail.com"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --email)
      EMAIL="$2"
      shift 2
      ;;
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS_PER_EPISODE="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      CPUS_SET_BY_USER="1"
      shift 2
      ;;
    --script)
      SCRIPT_PATH="$2"
      shift 2
      ;;
    --push)
      AUTO_PUSH="1"
      shift
      ;;
    --no-push)
      AUTO_PUSH="0"
      shift
      ;;
    --branch)
      PUSH_BRANCH="$2"
      shift 2
      ;;
    --remote)
      PUSH_REMOTE="$2"
      shift 2
      ;;
    --strict-push)
      STRICT_PUSH="1"
      shift
      ;;
    --git-user-name)
      GIT_USER_NAME="$2"
      shift 2
      ;;
    --git-user-email)
      GIT_USER_EMAIL="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [options]

Defaults in this file:
  EMAIL=dnlong5@iastate.edu
  ACCOUNT=s2026.se.4390.01
  PARTITION=instruction
  EPISODES=100

Optional overrides:
  --email       Email address for Slurm notifications
  --account     Slurm account (example: s2026.se.4390.01)
  --partition   Slurm partition (default: instruction)
  --episodes    Number of DDPG training episodes (default: 100)
  --output-dir  Training output directory (default: artifacts)
  --seed        Training random seed (default: 42)
  --max-steps   Max env steps per episode (default: 200)
  --num-envs    Number of parallel simulation envs (default: 1)
  --batch-size  Training batch size per update (default: 256)
  --gpus        Number of A100 GPUs to request (default: 1)
  --cpus        CPUs per task (default: 6)
  --script      Slurm script path (default: scripts/train_hpc.slurm)
  --push        Auto-commit and push artifacts after successful training (default)
  --no-push     Disable auto-push
  --branch      Branch to push to (default: current branch)
  --remote      Git remote name (default: origin)
  --strict-push Fail the Slurm job if push fails
  --git-user-name  Git user.name to use for auto-commit
  --git-user-email Git user.email to use for auto-commit
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$EMAIL" ]]; then
  echo "EMAIL is empty. Set EMAIL in this file or pass --email." >&2
  exit 1
fi

if [[ -z "$ACCOUNT" ]]; then
  echo "ACCOUNT is empty. Set ACCOUNT in this file or pass --account." >&2
  exit 1
fi

if ! [[ "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --gpus value: $GPUS (must be a positive integer)." >&2
  exit 1
fi

if ! [[ "$CPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --cpus value: $CPUS (must be a positive integer)." >&2
  exit 1
fi

if ! [[ "$MAX_STEPS_PER_EPISODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --max-steps value: $MAX_STEPS_PER_EPISODE (must be a positive integer)." >&2
  exit 1
fi

if ! [[ "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --num-envs value: $NUM_ENVS (must be a positive integer)." >&2
  exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --batch-size value: $BATCH_SIZE (must be a positive integer)." >&2
  exit 1
fi

if [[ "$CPUS_SET_BY_USER" == "0" ]]; then
  suggested_cpus=$((NUM_ENVS + 2))
  if (( suggested_cpus > CPUS )); then
    CPUS="$suggested_cpus"
  fi
fi

mkdir -p logs

echo "Submitting DDPG pendulum job..."
echo "  account: $ACCOUNT"
echo "  partition: $PARTITION"
echo "  episodes: $EPISODES"
echo "  max-steps: $MAX_STEPS_PER_EPISODE"
echo "  num-envs: $NUM_ENVS"
echo "  batch-size: $BATCH_SIZE"
echo "  output: $OUTPUT_DIR"
echo "  seed: $SEED"
echo "  gpus: $GPUS"
echo "  cpus: $CPUS"
echo "  email: $EMAIL"
echo "  auto-push: $AUTO_PUSH"
if [[ "$AUTO_PUSH" == "1" ]]; then
  echo "  push branch: ${PUSH_BRANCH:-<current-branch>}"
  echo "  push remote: $PUSH_REMOTE"
  echo "  strict push: $STRICT_PUSH"
  if [[ -n "$GIT_USER_NAME" ]]; then
    echo "  git user.name: $GIT_USER_NAME"
  fi
  if [[ -n "$GIT_USER_EMAIL" ]]; then
    echo "  git user.email: $GIT_USER_EMAIL"
  fi
fi

submit_output=$(sbatch \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
  --gres="gpu:a100:$GPUS" \
  --ntasks=1 \
  --cpus-per-task="$CPUS" \
  --mail-user="$EMAIL" \
  --mail-type=BEGIN,END,FAIL \
  --export=ALL,EPISODES="$EPISODES",OUTPUT_DIR="$OUTPUT_DIR",SEED="$SEED",MAX_STEPS_PER_EPISODE="$MAX_STEPS_PER_EPISODE",NUM_ENVS="$NUM_ENVS",BATCH_SIZE="$BATCH_SIZE",AUTO_PUSH="$AUTO_PUSH",PUSH_BRANCH="$PUSH_BRANCH",PUSH_REMOTE="$PUSH_REMOTE",STRICT_PUSH="$STRICT_PUSH",GIT_USER_NAME="$GIT_USER_NAME",GIT_USER_EMAIL="$GIT_USER_EMAIL" \
  "$SCRIPT_PATH")

echo "$submit_output"
job_id=$(echo "$submit_output" | awk '{print $4}')

if [[ -z "$job_id" ]]; then
  echo "Could not parse job id from sbatch output" >&2
  exit 1
fi

log_file="logs/ddpg-pendulum-${job_id}.out"
err_file="logs/ddpg-pendulum-${job_id}.err"

echo "Job ID: $job_id"
echo "Live stdout: $log_file"
echo "Live stderr: $err_file"

echo "Waiting for log file to appear..."
for _ in $(seq 1 120); do
  if [[ -f "$log_file" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -f "$log_file" ]]; then
  echo "Log file not created yet. You can monitor manually with: tail -f $log_file"
  exit 0
fi

echo "Streaming live logs (Ctrl+C to stop tail; job keeps running)..."
tail -f "$log_file"

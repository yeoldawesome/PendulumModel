#!/bin/bash

# ============================================================
# ISU Nova HPC - DDPG Pendulum training job
# Cluster:   Nova (nova.its.iastate.edu)
# Partition: instruction (for class accounts)
# ============================================================
# Replace the account below with your own class account.
# ============================================================

#SBATCH --job-name=ddpg-pendulum
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-core=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=instruction
#SBATCH --account=s2026.se.4390.01

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs artifacts

# Keep caches and venv on /work to avoid home quota issues.
WORKDIR="/work/classtmp/$USER/ddpg-pendulum"
mkdir -p "$WORKDIR"
export KERAS_HOME="$WORKDIR/.keras"
mkdir -p "$KERAS_HOME"

export KERAS_BACKEND=tensorflow
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

EPISODES="${EPISODES:-500}"
ENV_ID="${ENV_ID:-InvertedTriplePendulum-v0}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"
SEED="${SEED:-42}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-2000}"
NUM_ENVS="${NUM_ENVS:-1}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-100}"
ACTOR_LR="${ACTOR_LR:-0.0003}"
CRITIC_LR="${CRITIC_LR:-0.001}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-200000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-4096}"
NOISE_START="${NOISE_START:-0.3}"
NOISE_END="${NOISE_END:-0.05}"
NOISE_DECAY_EPISODES="${NOISE_DECAY_EPISODES:-300}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
EXPECTED_PYTHON="${EXPECTED_PYTHON:-3.11}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
FORCE_PIP_INSTALL="${FORCE_PIP_INSTALL:-0}"
AUTO_PUSH="${AUTO_PUSH:-0}"
PUSH_BRANCH="${PUSH_BRANCH:-}"
PUSH_REMOTE="${PUSH_REMOTE:-origin}"
STRICT_PUSH="${STRICT_PUSH:-0}"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"

module purge
if ! module load "$PYTHON_MODULE"; then
  echo "Failed to load module $PYTHON_MODULE"
  exit 1
fi

ACTIVE_PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python module requested: $PYTHON_MODULE"
echo "Python runtime detected: $ACTIVE_PYTHON_VERSION"
if [[ "$ACTIVE_PYTHON_VERSION" != "$EXPECTED_PYTHON" ]]; then
  echo "Expected Python $EXPECTED_PYTHON but got $ACTIVE_PYTHON_VERSION"
  echo "Set EXPECTED_PYTHON or fix PYTHON_MODULE before running."
  exit 1
fi

VENV_DIR="$WORKDIR/.venv"
VENV_PY_FILE="$VENV_DIR/.python-version"
if [[ -d "$VENV_DIR" ]]; then
  EXISTING_VENV_PYTHON=""
  if [[ -x "$VENV_DIR/bin/python3" ]]; then
    EXISTING_VENV_PYTHON=$($VENV_DIR/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
  elif [[ -f "$VENV_PY_FILE" ]]; then
    EXISTING_VENV_PYTHON=$(cat "$VENV_PY_FILE")
  fi

  if [[ -z "$EXISTING_VENV_PYTHON" ]]; then
    echo "Could not determine existing venv Python version. Rebuilding venv."
    rm -rf "$VENV_DIR"
  elif [[ "$EXISTING_VENV_PYTHON" != "$ACTIVE_PYTHON_VERSION" ]]; then
    echo "Rebuilding venv: found Python $EXISTING_VENV_PYTHON, need $ACTIVE_PYTHON_VERSION"
    rm -rf "$VENV_DIR"
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
  echo "$ACTIVE_PYTHON_VERSION" > "$VENV_PY_FILE"
fi
source "$VENV_DIR/bin/activate"

export PIP_CACHE_DIR="$WORKDIR/.pip-cache"
mkdir -p "$PIP_CACHE_DIR"

CPU_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OMP_NUM_THREADS="$CPU_THREADS"
export TF_NUM_INTRAOP_THREADS="$CPU_THREADS"
export TF_NUM_INTEROP_THREADS=2

echo "CPU threads: $CPU_THREADS"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "TF_NUM_INTRAOP_THREADS: $TF_NUM_INTRAOP_THREADS"
echo "TF_NUM_INTEROP_THREADS: $TF_NUM_INTEROP_THREADS"

REQ_HASH=$(sha256sum requirements.txt | awk '{print $1}')
DEP_FINGERPRINT="py=${ACTIVE_PYTHON_VERSION};req=${REQ_HASH};tf=tensorflow[and-cuda]>=2.16.0"
DEP_FINGERPRINT_FILE="$VENV_DIR/.deps-fingerprint"
EXISTING_DEP_FINGERPRINT=""
if [[ -f "$DEP_FINGERPRINT_FILE" ]]; then
  EXISTING_DEP_FINGERPRINT=$(cat "$DEP_FINGERPRINT_FILE")
fi

if [[ "$FORCE_PIP_INSTALL" == "1" || "$DEP_FINGERPRINT" != "$EXISTING_DEP_FINGERPRINT" ]]; then
  echo "Installing/updating Python dependencies (cache miss or forced)..."
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
  python3 -m pip install --upgrade "tensorflow[and-cuda]>=2.16.0"
  echo "$DEP_FINGERPRINT" > "$DEP_FINGERPRINT_FILE"
else
  echo "Dependency fingerprint unchanged; skipping pip install for faster startup."
fi

SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])')
CUDA_LIB_PATHS=(
  "$SITE_PACKAGES/nvidia/cublas/lib"
  "$SITE_PACKAGES/nvidia/cuda_runtime/lib"
  "$SITE_PACKAGES/nvidia/cuda_nvrtc/lib"
  "$SITE_PACKAGES/nvidia/cudnn/lib"
  "$SITE_PACKAGES/nvidia/cufft/lib"
  "$SITE_PACKAGES/nvidia/curand/lib"
  "$SITE_PACKAGES/nvidia/cusolver/lib"
  "$SITE_PACKAGES/nvidia/cusparse/lib"
  "$SITE_PACKAGES/nvidia/nccl/lib"
  "$SITE_PACKAGES/nvidia/nvjitlink/lib"
)

for lib_path in "${CUDA_LIB_PATHS[@]}"; do
  if [[ -d "$lib_path" ]]; then
    export LD_LIBRARY_PATH="$lib_path:${LD_LIBRARY_PATH:-}"
  fi
done

echo "LD_LIBRARY_PATH configured for TensorFlow CUDA runtime."

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi

echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

TF_GPU_COUNT=$(python3 -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices("GPU")))')
TF_BUILT_WITH_CUDA=$(python3 -c 'import tensorflow as tf; print(tf.test.is_built_with_cuda())')
echo "TensorFlow built with CUDA: $TF_BUILT_WITH_CUDA"
echo "TensorFlow GPU devices detected: $TF_GPU_COUNT"
if [[ "$REQUIRE_GPU" == "1" && "$TF_GPU_COUNT" -eq 0 ]]; then
  echo "TensorFlow does not see any GPUs. Aborting to avoid CPU-only training."
  exit 1
fi

echo "========== DDPG Pendulum Job Configuration =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Episodes: $EPISODES"
echo "Env id: $ENV_ID"
echo "Max steps/episode: $MAX_STEPS_PER_EPISODE"
echo "Parallel envs: $NUM_ENVS"
echo "Log interval steps: $LOG_INTERVAL_STEPS"
echo "Actor lr: $ACTOR_LR"
echo "Critic lr: $CRITIC_LR"
echo "Replay capacity: $BUFFER_CAPACITY"
echo "Batch size: $BATCH_SIZE"
echo "Updates per step: $UPDATES_PER_STEP"
echo "Warmup steps: $WARMUP_STEPS"
echo "Noise start/end/decay: $NOISE_START / $NOISE_END / $NOISE_DECAY_EPISODES"
echo "Seed: $SEED"
echo "Output dir: $OUTPUT_DIR"
echo "Require GPU: $REQUIRE_GPU"
echo "Auto push: $AUTO_PUSH"
if [[ "$AUTO_PUSH" == "1" ]]; then
  echo "Push branch override: ${PUSH_BRANCH:-<current-branch>}"
  echo "Push remote: $PUSH_REMOTE"
  echo "Strict push: $STRICT_PUSH"
  if [[ -n "$GIT_USER_NAME" ]]; then
    echo "Git user.name override: $GIT_USER_NAME"
  fi
  if [[ -n "$GIT_USER_EMAIL" ]]; then
    echo "Git user.email override: $GIT_USER_EMAIL"
  fi
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU(s):"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  echo "GPU info unavailable: nvidia-smi not found"
fi
echo "===================================================="

if [[ -n "${SLURM_GPUS:-}" && "${SLURM_GPUS}" -gt 1 ]]; then
  echo "Note: This training script is single-process DDPG; requesting >1 GPU usually does not speed it up much."
fi

srun python3 train.py \
  --episodes "$EPISODES" \
  --env-id "$ENV_ID" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --max-steps-per-episode "$MAX_STEPS_PER_EPISODE" \
  --num-envs "$NUM_ENVS" \
  --log-interval-steps "$LOG_INTERVAL_STEPS" \
  --actor-lr "$ACTOR_LR" \
  --critic-lr "$CRITIC_LR" \
  --buffer-capacity "$BUFFER_CAPACITY" \
  --batch-size "$BATCH_SIZE" \
  --updates-per-step "$UPDATES_PER_STEP" \
  --warmup-steps "$WARMUP_STEPS" \
  --noise-start "$NOISE_START" \
  --noise-end "$NOISE_END" \
  --noise-decay-episodes "$NOISE_DECAY_EPISODES" \
  --require-gpu "$REQUIRE_GPU"

auto_push_artifacts() {
  local target_branch current_branch
  if ! command -v git >/dev/null 2>&1; then
    echo "Auto-push skipped: git is not available on this node."
    return 1
  fi

  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Auto-push skipped: $SLURM_SUBMIT_DIR is not a git repository."
    return 1
  fi

  current_branch=$(git rev-parse --abbrev-ref HEAD)
  target_branch="$current_branch"
  if [[ -n "$PUSH_BRANCH" ]]; then
    target_branch="$PUSH_BRANCH"
  fi

  if [[ "$current_branch" != "$target_branch" ]]; then
    if git show-ref --verify --quiet "refs/heads/$target_branch"; then
      git switch "$target_branch"
    else
      git switch -c "$target_branch"
    fi
  fi

  if [[ -n "$GIT_USER_NAME" ]]; then
    git config user.name "$GIT_USER_NAME"
  fi
  if [[ -n "$GIT_USER_EMAIL" ]]; then
    git config user.email "$GIT_USER_EMAIL"
  fi

  # Artifacts are often gitignored locally; force-add so HPC outputs can still be pushed.
  git add -f "$OUTPUT_DIR"
  if git diff --cached --quiet; then
    echo "Auto-push debug: staged diff is empty after adding $OUTPUT_DIR"
    git status --short "$OUTPUT_DIR" || true
    echo "Auto-push: no artifact changes to commit in $OUTPUT_DIR"
    return 0
  fi

  git commit -m "hpc: pendulum artifacts from job ${SLURM_JOB_ID}"
  git push "$PUSH_REMOTE" "$target_branch"
  echo "Auto-push complete: pushed to $PUSH_REMOTE/$target_branch"
  return 0
}

if [[ "$AUTO_PUSH" == "1" ]]; then
  echo "Auto-push step starting..."
  if ! auto_push_artifacts; then
    echo "Auto-push step failed."
    if [[ "$STRICT_PUSH" == "1" ]]; then
      echo "STRICT_PUSH=1 so failing job on push error."
      exit 1
    fi
  fi
fi

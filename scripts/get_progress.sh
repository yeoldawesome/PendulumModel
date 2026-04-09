#!/bin/bash
set -euo pipefail

JOB_ID=""
OUTPUT_DIR=""
REMOTE="origin"
BRANCH=""
STRICT_PUSH="0"
GIT_USER_NAME=""
GIT_USER_EMAIL=""

usage() {
  cat <<EOF
Usage: $0 [options]

Find the newest actor model file and push it to git.

Options:
  --job-id <id>          Optional Slurm job id (used to discover output dir from logs)
  --output-dir <path>    Output directory to search (default: artifacts)
  --remote <name>        Git remote (default: origin)
  --branch <name>        Branch to push to (default: current branch)
  --strict               Exit non-zero if push fails
  --git-user-name <name> Override git user.name for this repo
  --git-user-email <mail> Override git user.email for this repo
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id)
      JOB_ID="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --strict)
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
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Current directory is not a git repository." >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" && -n "$JOB_ID" ]]; then
  LOG_FILE="logs/ddpg-pendulum-${JOB_ID}.out"
  if [[ -f "$LOG_FILE" ]]; then
    PARSED_DIR=$(grep -E '^Output dir:' "$LOG_FILE" | tail -n1 | sed -E 's/^Output dir:[[:space:]]*//')
    if [[ -n "$PARSED_DIR" ]]; then
      OUTPUT_DIR="$PARSED_DIR"
    fi
  fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"
if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Output directory not found: $OUTPUT_DIR" >&2
  exit 1
fi

MODEL_PATH=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*_actor.weights.h5' ! -name '*target_actor*' -printf '%T@ %p\n' | sort -nr | head -n1 | awk '{print $2}')
if [[ -z "$MODEL_PATH" ]]; then
  echo "No actor model file found in $OUTPUT_DIR" >&2
  exit 1
fi

if [[ -n "$GIT_USER_NAME" ]]; then
  git config user.name "$GIT_USER_NAME"
fi
if [[ -n "$GIT_USER_EMAIL" ]]; then
  git config user.email "$GIT_USER_EMAIL"
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
TARGET_BRANCH="$CURRENT_BRANCH"
if [[ -n "$BRANCH" ]]; then
  TARGET_BRANCH="$BRANCH"
fi

if [[ "$CURRENT_BRANCH" != "$TARGET_BRANCH" ]]; then
  if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
    git switch "$TARGET_BRANCH"
  else
    git switch -c "$TARGET_BRANCH"
  fi
fi

git add -f "$MODEL_PATH"
if git diff --cached --quiet; then
  echo "No changes to commit for model: $MODEL_PATH"
  exit 0
fi

MODEL_NAME=$(basename "$MODEL_PATH")
JOB_LABEL="${JOB_ID:-latest}"
git commit -m "hpc: push actor model from job ${JOB_LABEL} (${MODEL_NAME})"

if git push "$REMOTE" "$TARGET_BRANCH"; then
  echo "Pushed model to ${REMOTE}/${TARGET_BRANCH}: ${MODEL_NAME}"
else
  echo "Push failed." >&2
  if [[ "$STRICT_PUSH" == "1" ]]; then
    exit 1
  fi
fi

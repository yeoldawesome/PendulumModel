#!/bin/bash
set -euo pipefail

# Submit HPC training using the highest-episode actor checkpoint found in artifacts.
#
# Examples:
#   bash scripts/submit_resume_latest_hpc.sh
#   bash scripts/submit_resume_latest_hpc.sh --episodes 2000
#   bash scripts/submit_resume_latest_hpc.sh --env-id InvertedDoublePendulum-v5 --episodes 1000
#   bash scripts/submit_resume_latest_hpc.sh --match-any-env --episodes 500
#   bash scripts/submit_resume_latest_hpc.sh --print-only --episodes 2000

ARTIFACTS_DIR="artifacts"
ENV_ID="InvertedTriplePendulum-v0"
MATCH_ANY_ENV=0
PRINT_ONLY=0

usage() {
  cat <<EOF
Usage: $0 [options passed through to submit_and_watch_hpc.sh]

Helper options:
  --artifacts-dir DIR   Directory to scan for actor checkpoints (default: artifacts)
  --env-id ENV          Environment id used for filtering + submit (default: InvertedTriplePendulum-v0)
  --match-any-env       Do not filter by env slug; pick highest episode across all actor checkpoints
  --print-only          Print resolved command without submitting
  -h, --help            Show help

All other options are passed to scripts/submit_and_watch_hpc.sh.
EOF
}

env_to_slug() {
  local env="$1"
  env=$(echo "$env" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  env=$(echo "$env" | tr -cd 'a-z0-9_')
  if [[ -z "$env" ]]; then
    echo "unknown_env"
  else
    echo "$env"
  fi
}

PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifacts-dir)
      ARTIFACTS_DIR="$2"
      shift 2
      ;;
    --env-id)
      ENV_ID="$2"
      shift 2
      ;;
    --match-any-env)
      MATCH_ANY_ENV=1
      shift
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d "$ARTIFACTS_DIR" ]]; then
  echo "Artifacts directory not found: $ARTIFACTS_DIR" >&2
  exit 1
fi

ENV_SLUG="$(env_to_slug "$ENV_ID")"
BEST_FILE=""
BEST_EP=-1

shopt -s nullglob
for file in "$ARTIFACTS_DIR"/*actor.weights.h5; do
  [[ -f "$file" ]] || continue

  base_name="$(basename "$file")"
  if [[ "$MATCH_ANY_ENV" -eq 0 && "$base_name" != *"$ENV_SLUG"* ]]; then
    continue
  fi

  if [[ "$base_name" =~ _ep([0-9]+)_ ]]; then
    ep="${BASH_REMATCH[1]}"
    if (( ep > BEST_EP )); then
      BEST_EP=$ep
      BEST_FILE="$file"
    elif (( ep == BEST_EP )) && [[ -n "$BEST_FILE" ]] && [[ "$file" -nt "$BEST_FILE" ]]; then
      # Tie-break on recency.
      BEST_FILE="$file"
    fi
  fi
done
shopt -u nullglob

if (( BEST_EP < 0 )) || [[ -z "$BEST_FILE" ]]; then
  if [[ "$MATCH_ANY_ENV" -eq 1 ]]; then
    echo "No actor checkpoints with _ep<NUM>_ found in: $ARTIFACTS_DIR" >&2
  else
    echo "No actor checkpoints for env slug '$ENV_SLUG' found in: $ARTIFACTS_DIR" >&2
    echo "Tip: use --match-any-env to ignore env filtering." >&2
  fi
  exit 1
fi

echo "Selected checkpoint: $BEST_FILE"
echo "Detected episode offset: $BEST_EP"

CMD=(
  bash scripts/submit_and_watch_hpc.sh
  --env-id "$ENV_ID"
  --resume-actor-weights "$BEST_FILE"
  --resume-episode-offset "$BEST_EP"
)

if [[ "${#PASSTHROUGH[@]}" -gt 0 ]]; then
  CMD+=("${PASSTHROUGH[@]}")
fi

echo ""
echo "Submitting command:"
printf ' %q' "${CMD[@]}"
echo ""

if [[ "$PRINT_ONLY" -eq 1 ]]; then
  exit 0
fi

"${CMD[@]}"

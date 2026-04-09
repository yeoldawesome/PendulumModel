#!/bin/bash
set -euo pipefail

# Submit HPC training using either best-eval or latest-episode actor checkpoint.
#
# Examples:
#   bash scripts/submit_resume_latest_hpc.sh
#   bash scripts/submit_resume_latest_hpc.sh --episodes 2000
#   bash scripts/submit_resume_latest_hpc.sh --env-id InvertedDoublePendulum-v5 --episodes 1000
#   bash scripts/submit_resume_latest_hpc.sh --select-mode latest --episodes 1000
#   bash scripts/submit_resume_latest_hpc.sh --match-any-env --episodes 500
#   bash scripts/submit_resume_latest_hpc.sh --print-only --episodes 2000

ARTIFACTS_DIR="artifacts"
ENV_ID="InvertedTriplePendulum-v0"
MATCH_ANY_ENV=0
PRINT_ONLY=0
SELECT_MODE="best-eval"
USE_EXPLORATION_PROFILE=1
EXPLORE_NOISE_START="0.20"
EXPLORE_NOISE_END="0.08"
EXPLORE_NOISE_DECAY_EPISODES="3000"

usage() {
  cat <<EOF
Usage: $0 [options passed through to submit_and_watch_hpc.sh]

Helper options:
  --artifacts-dir DIR   Directory to scan for actor checkpoints (default: artifacts)
  --env-id ENV          Environment id used for filtering + submit (default: InvertedTriplePendulum-v0)
  --select-mode MODE    Checkpoint selection mode: best-eval (default) or latest
  --no-exploration-profile
                       Disable default resume exploration tuning.
  --explore-noise-start VALUE
                       Exploration profile noise-start (default: 0.20)
  --explore-noise-end VALUE
                       Exploration profile noise-end (default: 0.08)
  --explore-noise-decay-episodes N
                       Exploration profile noise-decay-episodes (default: 3000)
  --match-any-env       Do not filter by env slug when selecting checkpoints/CSV files
  --print-only          Print resolved command without submitting
  -h, --help            Show help

All other options are passed to scripts/submit_and_watch_hpc.sh.
EOF
}

find_latest_checkpoint() {
  local artifacts_dir="$1"
  local env_slug="$2"
  local match_any_env="$3"
  local best_file=""
  local best_ep=-1

  shopt -s nullglob
  for file in "$artifacts_dir"/*actor.weights.h5; do
    [[ -f "$file" ]] || continue

    local base_name
    base_name="$(basename "$file")"
    if [[ "$match_any_env" -eq 0 && "$base_name" != *"$env_slug"* ]]; then
      continue
    fi

    if [[ "$base_name" =~ _ep([0-9]+)_ ]]; then
      local ep
      ep="${BASH_REMATCH[1]}"
      if (( ep > best_ep )); then
        best_ep=$ep
        best_file="$file"
      elif (( ep == best_ep )) && [[ -n "$best_file" ]] && [[ "$file" -nt "$best_file" ]]; then
        best_file="$file"
      fi
    fi
  done
  shopt -u nullglob

  if (( best_ep < 0 )) || [[ -z "$best_file" ]]; then
    echo ""
    return
  fi

  echo "$best_file|$best_ep"
}

find_best_eval_checkpoint() {
  local artifacts_dir="$1"
  local env_slug="$2"
  local match_any_env="$3"
  local newest_csv=""

  shopt -s nullglob
  for csv_file in "$artifacts_dir"/*_eval_metrics.csv; do
    [[ -f "$csv_file" ]] || continue
    local csv_name
    csv_name="$(basename "$csv_file")"
    if [[ "$match_any_env" -eq 0 && "$csv_name" != *"$env_slug"* ]]; then
      continue
    fi
    if [[ -z "$newest_csv" || "$csv_file" -nt "$newest_csv" ]]; then
      newest_csv="$csv_file"
    fi
  done
  shopt -u nullglob

  if [[ -z "$newest_csv" ]]; then
    echo ""
    return
  fi

  local best_eval_line
  best_eval_line="$(awk -F, '
    NR==1 {
      for (i = 1; i <= NF; i++) {
        if ($i == "episode") episode_col = i
        if ($i == "success_at_100_rate") s100_col = i
        if ($i == "success_at_50_rate") s50_col = i
        if ($i == "success_at_300_rate") s300_col = i
        if ($i == "avg_time_to_failure_steps") ttf_col = i
        if ($i == "avg_resets_per_episode") resets_col = i
        if ($i == "mean_return") mean_col = i
      }
      next
    }
    NR > 1 && episode_col > 0 {
      if ($(episode_col) == "") next
      ep = $(episode_col) + 0
      s100 = (s100_col > 0 && $(s100_col) != "") ? ($(s100_col) + 0) : -1
      s50 = (s50_col > 0 && $(s50_col) != "") ? ($(s50_col) + 0) : -1
      s300 = (s300_col > 0 && $(s300_col) != "") ? ($(s300_col) + 0) : -1
      ttf = (ttf_col > 0 && $(ttf_col) != "") ? ($(ttf_col) + 0) : -1
      resets = (resets_col > 0 && $(resets_col) != "") ? ($(resets_col) + 0) : 999999
      mean = (mean_col > 0 && $(mean_col) != "") ? ($(mean_col) + 0) : -999999

      # Balance-first ranking:
      # 1) success@100, 2) success@50 (or success@300 fallback), 3) avg time-to-failure,
      # 4) fewer resets, 5) mean return, 6) newer episode.
      s50_or_fallback = (s50 >= 0) ? s50 : s300

      if (!seen \
          || s100 > best_s100 \
          || (s100 == best_s100 && s50_or_fallback > best_s50) \
          || (s100 == best_s100 && s50_or_fallback == best_s50 && ttf > best_ttf) \
          || (s100 == best_s100 && s50_or_fallback == best_s50 && ttf == best_ttf && resets < best_resets) \
          || (s100 == best_s100 && s50_or_fallback == best_s50 && ttf == best_ttf && resets == best_resets && mean > best_mean) \
          || (s100 == best_s100 && s50_or_fallback == best_s50 && ttf == best_ttf && resets == best_resets && mean == best_mean && ep > best_ep)) {
        seen = 1
        best_s100 = s100
        best_s50 = s50_or_fallback
        best_ttf = ttf
        best_resets = resets
        best_mean = mean
        best_ep = ep
      }
    }
    END {
      if (seen) {
        printf "%d|%.10f|%.10f|%.10f|%.10f", best_ep, best_s100, best_ttf, best_resets, best_mean
      }
    }
  ' "$newest_csv")"

  if [[ -z "$best_eval_line" ]]; then
    echo ""
    return
  fi

  local best_ep
  local best_s100
  local best_ttf
  local best_resets
  local best_mean
  best_ep="${best_eval_line%%|*}"
  REST_LINE="${best_eval_line#*|}"
  best_s100="${REST_LINE%%|*}"
  REST_LINE="${REST_LINE#*|}"
  best_ttf="${REST_LINE%%|*}"
  REST_LINE="${REST_LINE#*|}"
  best_resets="${REST_LINE%%|*}"
  best_mean="${REST_LINE##*|}"
  local best_file=""

  shopt -s nullglob
  for file in "$artifacts_dir"/*_ep${best_ep}_*_actor.weights.h5; do
    [[ -f "$file" ]] || continue
    local base_name
    base_name="$(basename "$file")"
    if [[ "$match_any_env" -eq 0 && "$base_name" != *"$env_slug"* ]]; then
      continue
    fi
    if [[ -z "$best_file" || "$file" -nt "$best_file" ]]; then
      best_file="$file"
    fi
  done
  shopt -u nullglob

  if [[ -z "$best_file" ]]; then
    echo ""
    return
  fi

  echo "$best_file|$best_ep|$newest_csv|$best_s100|$best_ttf|$best_resets|$best_mean"
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
    --select-mode)
      SELECT_MODE="$2"
      shift 2
      ;;
    --no-exploration-profile)
      USE_EXPLORATION_PROFILE=0
      shift
      ;;
    --explore-noise-start)
      EXPLORE_NOISE_START="$2"
      shift 2
      ;;
    --explore-noise-end)
      EXPLORE_NOISE_END="$2"
      shift 2
      ;;
    --explore-noise-decay-episodes)
      EXPLORE_NOISE_DECAY_EPISODES="$2"
      shift 2
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

if [[ "$SELECT_MODE" != "best-eval" && "$SELECT_MODE" != "latest" ]]; then
  echo "Invalid --select-mode: $SELECT_MODE (expected best-eval or latest)" >&2
  exit 1
fi

ENV_SLUG="$(env_to_slug "$ENV_ID")"
BEST_FILE=""
BEST_EP=""

if [[ "$SELECT_MODE" == "best-eval" ]]; then
  BEST_EVAL_RESULT="$(find_best_eval_checkpoint "$ARTIFACTS_DIR" "$ENV_SLUG" "$MATCH_ANY_ENV")"
  if [[ -n "$BEST_EVAL_RESULT" ]]; then
    BEST_FILE="${BEST_EVAL_RESULT%%|*}"
    REST="${BEST_EVAL_RESULT#*|}"
    BEST_EP="${REST%%|*}"
    REST="${REST#*|}"
    SOURCE_CSV="${REST%%|*}"
    REST="${REST#*|}"
    BEST_S100="${REST%%|*}"
    REST="${REST#*|}"
    BEST_TTF="${REST%%|*}"
    REST="${REST#*|}"
    BEST_RESETS="${REST%%|*}"
    BEST_MEAN="${REST##*|}"
    echo "Selection mode: best-eval"
    echo "Source eval CSV: $SOURCE_CSV"
    echo "Best balance snapshot: success@100=$BEST_S100, avg_ttf=$BEST_TTF, avg_resets=$BEST_RESETS, mean_return=$BEST_MEAN (episode $BEST_EP)"
  else
    echo "No usable eval CSV/checkpoint mapping found. Falling back to latest checkpoint selection."
  fi
fi

if [[ -z "$BEST_FILE" || -z "$BEST_EP" ]]; then
  LATEST_RESULT="$(find_latest_checkpoint "$ARTIFACTS_DIR" "$ENV_SLUG" "$MATCH_ANY_ENV")"
  if [[ -z "$LATEST_RESULT" ]]; then
    if [[ "$MATCH_ANY_ENV" -eq 1 ]]; then
      echo "No actor checkpoints with _ep<NUM>_ found in: $ARTIFACTS_DIR" >&2
    else
      echo "No actor checkpoints for env slug '$ENV_SLUG' found in: $ARTIFACTS_DIR" >&2
      echo "Tip: use --match-any-env to ignore env filtering." >&2
    fi
    exit 1
  fi
  BEST_FILE="${LATEST_RESULT%%|*}"
  BEST_EP="${LATEST_RESULT##*|}"
  echo "Selection mode: latest"
fi

echo "Selected checkpoint: $BEST_FILE"
echo "Detected episode offset: $BEST_EP"
if [[ "$USE_EXPLORATION_PROFILE" -eq 1 ]]; then
  echo "Exploration profile: noise-start=$EXPLORE_NOISE_START noise-end=$EXPLORE_NOISE_END noise-decay-episodes=$EXPLORE_NOISE_DECAY_EPISODES"
else
  echo "Exploration profile: disabled"
fi

CMD=(
  bash scripts/submit_and_watch_hpc.sh
  --env-id "$ENV_ID"
  --resume-actor-weights "$BEST_FILE"
  --resume-episode-offset "$BEST_EP"
)

if [[ "$USE_EXPLORATION_PROFILE" -eq 1 ]]; then
  CMD+=(
    --noise-start "$EXPLORE_NOISE_START"
    --noise-end "$EXPLORE_NOISE_END"
    --noise-decay-episodes "$EXPLORE_NOISE_DECAY_EPISODES"
  )
fi

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

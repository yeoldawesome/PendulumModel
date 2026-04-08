#!/bin/bash
set -euo pipefail

# Follow logs for a running Slurm job.
# Usage:
# bash scripts/watch_running_job_logs.sh
# bash scripts/watch_running_job_logs.sh --job 12345678
# bash scripts/watch_running_job_logs.sh --stderr

JOB_ID=""
SHOW_STDERR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      JOB_ID="$2"
      shift 2
      ;;
    --stderr)
      SHOW_STDERR=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--job <job_id>] [--stderr]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$JOB_ID" ]]; then
  JOB_ID=$(squeue --me -h -o "%i %T" | awk '$2=="RUNNING"{print $1; exit}')
  if [[ -z "$JOB_ID" ]]; then
    JOB_ID=$(squeue --me -h -o "%i" | head -n 1)
  fi
fi

if [[ -z "$JOB_ID" ]]; then
  echo "No jobs found for user $USER."
  exit 1
fi

if [[ "$SHOW_STDERR" == "1" ]]; then
  log_file="logs/ddpg-pendulum-${JOB_ID}.err"
else
  log_file="logs/ddpg-pendulum-${JOB_ID}.out"
fi

echo "Job ID: $JOB_ID"
echo "Log file: $log_file"

for _ in $(seq 1 120); do
  if [[ -f "$log_file" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -f "$log_file" ]]; then
  echo "Log file not found yet: $log_file"
  echo "Tip: run squeue --me to confirm the job is still queued/running."
  exit 1
fi

echo "Showing header:"
head -n 120 "$log_file" || true

echo "Following live log stream (Ctrl+C to stop viewing)..."
tail -f "$log_file"

#!/bin/bash
set -euo pipefail

# Cancel all Slurm jobs for the current user.
# Usage:
# bash scripts/cancel_all_jobs.sh
# bash scripts/cancel_all_jobs.sh --force

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

echo "Jobs for user $USER:"
squeue --me || true

job_count=$(squeue --me -h | wc -l | tr -d ' ')
if [[ "$job_count" == "0" ]]; then
  echo "No jobs to cancel."
  exit 0
fi

if [[ "$FORCE" != "1" ]]; then
  read -r -p "Cancel ALL $job_count jobs for $USER? (y/N): " reply
  case "$reply" in
    y|Y|yes|YES)
      ;;
    *)
      echo "Cancelled by user."
      exit 0
      ;;
  esac
fi

scancel -u "$USER"
echo "Requested cancellation of all jobs for $USER."
echo "Current queue state:"
squeue --me || true

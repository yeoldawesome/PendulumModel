# DDPG Pendulum (Keras) with ISU Nova HPC Workflow

This project runs the Keras DDPG Pendulum example on Iowa State Nova HPC using the same Slurm + GPU pattern from your Assignment3 setup.

## Files

- `train.py`: DDPG training script for `Pendulum-v1`.
- `requirements.txt`: Python dependencies.
- `scripts/train_hpc.slurm`: main Nova Slurm job script.
- `scripts/submit_and_watch_hpc.sh`: one-command submit + live log tail.
- `scripts/watch_running_job_logs.sh`: follow current job logs.
- `scripts/cancel_all_jobs.sh`: cancel all your queued/running jobs.

## Local quick test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --episodes 5 --output-dir artifacts
```

## Run on ISU Nova HPC

1. Copy this repo to Nova and `cd` into it.
2. Edit account/email defaults in:
   - `scripts/train_hpc.slurm`
   - `scripts/submit_and_watch_hpc.sh`
3. Submit with one command:

```bash
bash scripts/submit_and_watch_hpc.sh --email yournetid@iastate.edu --account s2026.se.4390.01 --partition instruction --episodes 100
```

Windows PowerShell note:

- Use the full local path when changing directories, for example:
   `cd "C:/Users/dnlon/Downloads/Ai-Class/PendulumModel"`
- Do not copy Markdown link text. Run the script path directly:
   `bash ./scripts/submit_and_watch_hpc.sh --email yournetid@iastate.edu --account s2026.se.4390.01 --partition instruction --episodes 100`

This wrapper submits:

- `--gres=gpu:a100:1`
- `--cpus-per-task=6`
- exported vars: `EPISODES`, `OUTPUT_DIR`, `SEED`

## Direct Slurm submit (without wrapper)

```bash
sbatch scripts/train_hpc.slurm
```

Optional variable overrides:

```bash
EPISODES=150 OUTPUT_DIR=artifacts REQUIRE_GPU=1 sbatch scripts/train_hpc.slurm
```

## Monitor and control jobs

```bash
bash scripts/watch_running_job_logs.sh
bash scripts/watch_running_job_logs.sh --stderr
bash scripts/cancel_all_jobs.sh
```

## Artifacts

Training writes to `artifacts/`:

- `pendulum_actor_<timestamp>.weights.h5`
- `pendulum_critic_<timestamp>.weights.h5`
- `pendulum_target_actor_<timestamp>.weights.h5`
- `pendulum_target_critic_<timestamp>.weights.h5`
- `pendulum_rewards_<timestamp>.npy`
- `metadata.json`

## Notes

- Slurm script uses `/work/classtmp/$USER/ddpg-pendulum` for venv and Keras cache, mirroring your Assignment3 disk-quota-safe approach.
- It validates Python module version and checks TensorFlow GPU visibility before training.
- For Nova environments where TensorFlow defaults to CPU wheels, the script enforces `tensorflow[and-cuda]` on the compute node.

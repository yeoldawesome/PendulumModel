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

## View Pendulum Episodes with a GUI

After training at least one model, you can launch a desktop viewer that loads actor weights and runs rendered Pendulum episodes.

```bash
python viewer_gui.py
```

In the GUI:

- Select a `pendulum_actor_*.weights.h5` file from `artifacts/` (or browse to any weights file).
- Set number of episodes, max steps, and seed.
- Click `Run Loaded Model` to watch your trained policy in the Gym window.
- Click `Run Random Policy` for a baseline comparison.

## Run on ISU Nova HPC

1. Copy this repo to Nova and `cd` into it.
2. Edit account/email defaults in:
   - `scripts/train_hpc.slurm`
   - `scripts/submit_and_watch_hpc.sh`
3. Submit with one command:

```bash
bash scripts/submit_and_watch_hpc.sh --email yournetid@iastate.edu --account s2026.se.4390.01 --partition instruction --episodes 100
```

To auto-commit and push trained artifacts to your branch after training finishes:

```bash
bash scripts/submit_and_watch_hpc.sh --email yournetid@iastate.edu --account s2026.se.4390.01 --partition instruction --episodes 100 --push --branch your-branch-name
```

Optional push controls:

- `--remote origin` (default remote)
- `--strict-push` (fail the Slurm job if git push fails)
- `--git-user-name "Your Name" --git-user-email "you@iastate.edu"` if HPC git identity is not configured

Windows PowerShell note:

- Use the full local path when changing directories, for example:
   `cd "C:/Users/dnlon/Downloads/Ai-Class/PendulumModel"`
- Do not copy Markdown link text. Run the script path directly:
   `bash ./scripts/submit_and_watch_hpc.sh --email yournetid@iastate.edu --account s2026.se.4390.01 --partition instruction --episodes 100`

This wrapper submits:

- `--gres=gpu:a100:<gpus>`
- `--cpus-per-task=<cpus>`
- exported vars: `EPISODES`, `OUTPUT_DIR`, `SEED`, `MAX_STEPS_PER_EPISODE`, `NUM_ENVS`

Speed tuning examples:

```bash
bash scripts/submit_and_watch_hpc.sh --email yournetid@iastate.edu --account s2026.se.4390.01 --partition instruction --episodes 100 --max-steps 150 --num-envs 8 --cpus 12 --gpus 1
```

Optional two-GPU request:

```bash
bash scripts/submit_and_watch_hpc.sh --email yournetid@iastate.edu --account s2026.se.4390.01 --partition instruction --episodes 100 --max-steps 150 --num-envs 8 --cpus 12 --gpus 2
```

Note: this trainer now supports parallel simulation via `--num-envs`, which usually improves speed more than adding a second GPU for this single-process DDPG setup.

## Direct Slurm submit (without wrapper)

```bash
sbatch scripts/train_hpc.slurm
```

Optional variable overrides:

```bash
EPISODES=150 OUTPUT_DIR=artifacts REQUIRE_GPU=1 sbatch scripts/train_hpc.slurm
```

Include max steps override in direct submit when needed:

```bash
EPISODES=150 MAX_STEPS_PER_EPISODE=150 OUTPUT_DIR=artifacts REQUIRE_GPU=1 sbatch scripts/train_hpc.slurm
```

Direct Slurm auto-push example:

```bash
EPISODES=150 OUTPUT_DIR=artifacts AUTO_PUSH=1 PUSH_BRANCH=your-branch-name PUSH_REMOTE=origin STRICT_PUSH=1 sbatch scripts/train_hpc.slurm
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

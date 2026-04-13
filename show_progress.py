import csv
import matplotlib.pyplot as plt
import pathlib

CSV_PATH = pathlib.Path("artifacts/progress.csv")

def load_progress(csv_path):
    episodes = []
    avg_rewards = []
    eval_avg_rewards = []
    eval_avg_lengths = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            # Try all possible rolling avg reward column names
            if "avg_reward_10" in row:
                avg_rewards.append(float(row["avg_reward_10"]))
            elif "avg_reward_40" in row:
                avg_rewards.append(float(row["avg_reward_40"]))
            elif "avg_reward" in row:
                avg_rewards.append(float(row["avg_reward"]))
            else:
                avg_rewards.append(float('nan'))
            eval_avg_rewards.append(float(row.get("eval_avg_reward", 'nan')))
            eval_avg_lengths.append(float(row.get("eval_avg_length", 'nan')))
    return episodes, avg_rewards, eval_avg_rewards, eval_avg_lengths

def plot_progress(episodes, avg_rewards_40, eval_avg_rewards, eval_avg_lengths):
    import numpy as np
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # First graph: Rewards
    axs[0].plot(episodes, avg_rewards_40, label="Rolling Avg Reward (10)", marker="o")
    axs[0].plot(episodes, eval_avg_rewards, label="Eval Avg Reward", marker="x")
    # Trend lines for rewards
    if len(episodes) > 1:
        z1 = np.polyfit(episodes, avg_rewards_40, 1)
        p1 = np.poly1d(z1)
        axs[0].plot(episodes, p1(episodes), linestyle="--", color="blue", alpha=0.5, label="Trend: Rolling Avg Reward")
        z2 = np.polyfit(episodes, eval_avg_rewards, 1)
        p2 = np.poly1d(z2)
        axs[0].plot(episodes, p2(episodes), linestyle="--", color="green", alpha=0.5, label="Trend: Eval Avg Reward")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].set_title("Training Progress Over Time")
    axs[0].legend()
    axs[0].grid(True)

    # Second graph: Episode Lengths
    axs[1].plot(episodes, eval_avg_lengths, label="Eval Avg Episode Length", marker="s", color="orange")
    # Trend line for episode lengths
    if len(episodes) > 1:
        z3 = np.polyfit(episodes, eval_avg_lengths, 1)
        p3 = np.poly1d(z3)
        axs[1].plot(episodes, p3(episodes), linestyle="--", color="red", alpha=0.5, label="Trend: Eval Avg Length")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Avg Episode Length")
    axs[1].set_title("Evaluation Episode Length Over Time")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    csv_path = CSV_PATH
    if not csv_path.exists():
        # Try to find the most recent *_progress.csv in artifacts
        artifact_dir = pathlib.Path("artifacts")
        progress_files = sorted(artifact_dir.glob("*_progress.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not progress_files:
            print(f"No progress CSV files found in {artifact_dir}")
            return
        csv_path = progress_files[0]
        print(f"CSV file not found: {CSV_PATH}\nUsing most recent: {csv_path}")
    episodes, avg_rewards_40, eval_avg_rewards, eval_avg_lengths = load_progress(csv_path)
    plot_progress(episodes, avg_rewards_40, eval_avg_rewards, eval_avg_lengths)

if __name__ == "__main__":
    main()

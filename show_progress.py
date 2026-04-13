import csv
import matplotlib.pyplot as plt
import pathlib

CSV_PATH = pathlib.Path("artifacts/progress.csv")

def load_progress(csv_path):
    episodes = []
    avg_rewards_40 = []
    eval_avg_rewards = []
    eval_avg_lengths = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            avg_rewards_40.append(float(row["avg_reward_40"]))
            eval_avg_rewards.append(float(row["eval_avg_reward"]))
            eval_avg_lengths.append(float(row["eval_avg_length"]))
    return episodes, avg_rewards_40, eval_avg_rewards, eval_avg_lengths

def plot_progress(episodes, avg_rewards_40, eval_avg_rewards, eval_avg_lengths):
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, avg_rewards_40, label="Rolling Avg Reward (40)", marker="o")
    plt.plot(episodes, eval_avg_rewards, label="Eval Avg Reward", marker="x")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, eval_avg_lengths, label="Eval Avg Episode Length", marker="s", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Avg Episode Length")
    plt.title("Evaluation Episode Length Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    if not CSV_PATH.exists():
        print(f"CSV file not found: {CSV_PATH}")
        return
    episodes, avg_rewards_40, eval_avg_rewards, eval_avg_lengths = load_progress(CSV_PATH)
    plot_progress(episodes, avg_rewards_40, eval_avg_rewards, eval_avg_lengths)

if __name__ == "__main__":
    main()

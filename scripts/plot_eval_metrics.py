import argparse
import csv
import math
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evaluation trends from train.py eval metrics CSV.")
    parser.add_argument(
        "csv_path",
        type=str,
        nargs="?",
        default="",
        help="Path to *_eval_metrics.csv file. If omitted, uses newest file in artifacts/.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output image path. If omitted, no PNG is written.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=1,
        choices=[0, 1],
        help="Show interactive plot window (1, default) or skip display (0).",
    )
    return parser.parse_args()


def read_eval_rows(csv_path: pathlib.Path) -> list[dict[str, float]]:
    def get_float(row: dict[str, str], key: str) -> float:
        value = row.get(key, "")
        if value is None or value == "":
            return float("nan")
        return float(value)

    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "episode": get_float(row, "episode"),
                    "mean_return": get_float(row, "mean_return"),
                    "median_return": get_float(row, "median_return"),
                    "return_ci95_low": get_float(row, "return_ci95_low"),
                    "return_ci95_high": get_float(row, "return_ci95_high"),
                    "success_rate": get_float(row, "success_rate"),
                    "success_ci95_low": get_float(row, "success_ci95_low"),
                    "success_ci95_high": get_float(row, "success_ci95_high"),
                    "success_at_300_rate": get_float(row, "success_at_300_rate"),
                    "success_at_300_ci95_low": get_float(row, "success_at_300_ci95_low"),
                    "success_at_300_ci95_high": get_float(row, "success_at_300_ci95_high"),
                    "success_at_500_rate": get_float(row, "success_at_500_rate"),
                    "success_at_500_ci95_low": get_float(row, "success_at_500_ci95_low"),
                    "success_at_500_ci95_high": get_float(row, "success_at_500_ci95_high"),
                    "avg_time_to_failure_steps": get_float(row, "avg_time_to_failure_steps"),
                    "median_time_to_failure_steps": get_float(row, "median_time_to_failure_steps"),
                    "max_time_to_failure_steps": get_float(row, "max_time_to_failure_steps"),
                    "avg_resets_per_episode": get_float(row, "avg_resets_per_episode"),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.csv_path:
        csv_path = pathlib.Path(args.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
    else:
        candidates = sorted(pathlib.Path("artifacts").glob("*_eval_metrics.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("No *_eval_metrics.csv files found in artifacts/. Provide csv_path explicitly.")
        csv_path = candidates[0]
        print(f"Using latest eval CSV: {csv_path}")

    rows = read_eval_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        print("matplotlib is required for plotting. Install with: pip install matplotlib", file=sys.stderr)
        raise exc

    episodes = [r["episode"] for r in rows]
    mean_return = [r["mean_return"] for r in rows]
    median_return = [r["median_return"] for r in rows]
    return_ci_low = [r["return_ci95_low"] for r in rows]
    return_ci_high = [r["return_ci95_high"] for r in rows]
    success_pct = [100.0 * r["success_rate"] for r in rows]
    success_ci_low = [100.0 * r["success_ci95_low"] for r in rows]
    success_ci_high = [100.0 * r["success_ci95_high"] for r in rows]
    success_at_300_pct = [100.0 * r["success_at_300_rate"] for r in rows]
    success_at_300_ci_low = [100.0 * r["success_at_300_ci95_low"] for r in rows]
    success_at_300_ci_high = [100.0 * r["success_at_300_ci95_high"] for r in rows]
    success_at_500_pct = [100.0 * r["success_at_500_rate"] for r in rows]
    success_at_500_ci_low = [100.0 * r["success_at_500_ci95_low"] for r in rows]
    success_at_500_ci_high = [100.0 * r["success_at_500_ci95_high"] for r in rows]
    ttf = [r["avg_time_to_failure_steps"] for r in rows]
    median_ttf = [r["median_time_to_failure_steps"] for r in rows]
    max_ttf = [r["max_time_to_failure_steps"] for r in rows]
    resets = [r["avg_resets_per_episode"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    ax = axes[0][0]
    ax.plot(episodes, mean_return, label="Eval mean return", color="#1f77b4")
    ax.plot(episodes, median_return, label="Eval median return", color="#ff7f0e", linestyle="--")
    ax.fill_between(episodes, return_ci_low, return_ci_high, color="#1f77b4", alpha=0.2, label="95% CI")
    ax.set_title("Return Trend")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[0][1]
    ax.plot(episodes, success_pct, label="Success (no reset)", color="#2ca02c")
    ax.fill_between(episodes, success_ci_low, success_ci_high, color="#2ca02c", alpha=0.15, label="No-reset 95% CI")

    if any(not math.isnan(v) for v in success_at_300_pct):
        ax.plot(episodes, success_at_300_pct, label="Success @300", color="#17becf")
        ax.fill_between(
            episodes,
            success_at_300_ci_low,
            success_at_300_ci_high,
            color="#17becf",
            alpha=0.12,
            label="@300 95% CI",
        )

    if any(not math.isnan(v) for v in success_at_500_pct):
        ax.plot(episodes, success_at_500_pct, label="Success @500", color="#8c564b")
        ax.fill_between(
            episodes,
            success_at_500_ci_low,
            success_at_500_ci_high,
            color="#8c564b",
            alpha=0.12,
            label="@500 95% CI",
        )

    ax.set_title("Success Rate Trend")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1][0]
    ax.plot(episodes, ttf, color="#9467bd", label="Avg")
    if any(not math.isnan(v) for v in median_ttf):
        ax.plot(episodes, median_ttf, color="#bcbd22", linestyle="--", label="Median")
    if any(not math.isnan(v) for v in max_ttf):
        ax.plot(episodes, max_ttf, color="#7f7f7f", linestyle=":", label="Max")
    ax.set_title("Avg Time-to-Failure")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1][1]
    ax.plot(episodes, resets, color="#d62728")
    ax.set_title("Avg Resets per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Resets")
    ax.grid(True, alpha=0.3)

    if args.output:
        output_path = pathlib.Path(args.output)
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot: {output_path}")

    if args.show == 1:
        plt.show()


if __name__ == "__main__":
    main()

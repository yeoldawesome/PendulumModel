import pathlib
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
from typing import Any

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import gymnasium as gym
import numpy as np
import tensorflow as tf

from train import get_actor
from triple_pendulum_env import TRIPLE_PENDULUM_ENV_ID, register_triple_pendulum_env

tf.get_logger().setLevel("ERROR")

JOINTS_TO_ENV_ID = {
    1: "Pendulum-v1",
    2: "InvertedDoublePendulum-v5",
    3: TRIPLE_PENDULUM_ENV_ID,
}

ENV_ID_TO_JOINTS = {env_id: joints for joints, env_id in JOINTS_TO_ENV_ID.items()}


class PendulumViewerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Control Model Viewer")
        self.geometry("760x520")

        register_triple_pendulum_env()

        self.stop_event = threading.Event()
        self.runner_thread: threading.Thread | None = None

        self.model_var = tk.StringVar()
        self.model_b_var = tk.StringVar()
        self.model_c_var = tk.StringVar()
        self.model_d_var = tk.StringVar()
        self.env_id_var = tk.StringVar(value=TRIPLE_PENDULUM_ENV_ID)
        self.episodes_var = tk.StringVar(value="10")
        self.max_steps_var = tk.StringVar(value="1000")
        self.frame_delay_ms_var = tk.StringVar(value="10")
        self.seed_var = tk.StringVar(value="42")
        self.start_episode_var = tk.StringVar(value="1")
        self.use_2d_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.compare_stats_var = tk.StringVar(value="")
        self.compare_plot_data: dict[str, Any] | None = None

        self.sim_window: tk.Toplevel | None = None
        self.sim_canvas: tk.Canvas | None = None
        self.sim_canvas_left: tk.Canvas | None = None
        self.sim_canvas_right: tk.Canvas | None = None
        self.sim_compare_canvases: list[tk.Canvas] = []

        self._build_ui()
        self.refresh_models()

    @staticmethod
    def _summarize_exception(exc: Exception, max_len: int = 240) -> str:
        text = " ".join(str(exc).split())
        # Strip large tensor dumps if present.
        text = re.sub(r"\[[^\]]{120,}\]", "[...trimmed...]", text)
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    @staticmethod
    def _classify_exception_message(exc: Exception) -> str:
        lower = str(exc).lower()
        if "mujoco is not installed" in lower:
            return "MuJoCo not installed"
        if "could not be loaded" in lower or "must match" in lower or "shape" in lower:
            return "weights shape mismatch"
        return "error"

    @staticmethod
    def _guess_joints_from_model_path(model_path: str) -> int | None:
        name = pathlib.Path(model_path).name.lower()
        match = re.search(r"_j(\d+)_", name)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _label_from_model_path(model_path: str) -> str:
        name = pathlib.Path(model_path).name
        match = re.search(r"_ep(\d+)_", name)
        if match:
            return f"ep{match.group(1)}"
        return pathlib.Path(model_path).stem

    def _detect_model_compatible_env(self, model_path: str, preferred_env_id: str | None = None) -> tuple[str | None, str | None]:
        supported_envs = [TRIPLE_PENDULUM_ENV_ID, "InvertedDoublePendulum-v5", "Pendulum-v1"]
        if preferred_env_id in supported_envs:
            supported_envs.remove(preferred_env_id)
            supported_envs.insert(0, preferred_env_id)
        errors: list[str] = []

        for candidate_env_id in supported_envs:
            candidate_env = None
            try:
                candidate_env = gym.make(candidate_env_id)
                if not isinstance(candidate_env.action_space, gym.spaces.Box):
                    errors.append(f"{candidate_env_id}: action space is not Box")
                    continue

                num_states = candidate_env.observation_space.shape[0]
                num_actions = candidate_env.action_space.shape[0]
                upper_bound = candidate_env.action_space.high.astype(np.float32)

                candidate_actor = get_actor(
                    num_states=num_states,
                    num_actions=num_actions,
                    upper_bound=upper_bound,
                )
                candidate_actor(np.zeros((1, num_states), dtype=np.float32), training=False)
                candidate_actor.load_weights(model_path)
                return candidate_env_id, None
            except Exception as exc:  # noqa: BLE001
                kind = self._classify_exception_message(exc)
                if kind == "MuJoCo not installed":
                    errors.append(f"{candidate_env_id}: MuJoCo not installed")
                elif kind == "weights shape mismatch":
                    errors.append(f"{candidate_env_id}: weights shape mismatch")
                else:
                    errors.append(f"{candidate_env_id}: {self._summarize_exception(exc)}")
            finally:
                if candidate_env is not None:
                    candidate_env.close()

        return None, " | ".join(errors)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text="DDPG Episode Viewer", font=("Segoe UI", 14, "bold"))
        title.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))

        ttk.Label(container, text="Environment id:").grid(row=1, column=0, sticky="w")
        env_combo = ttk.Combobox(
            container,
            textvariable=self.env_id_var,
            width=40,
            values=(TRIPLE_PENDULUM_ENV_ID, "InvertedDoublePendulum-v5", "Pendulum-v1"),
        )
        env_combo.grid(row=1, column=1, columnspan=2, sticky="w", padx=(8, 8))

        ttk.Label(container, text="Actor model weights:").grid(row=2, column=0, sticky="w")
        self.model_combo = ttk.Combobox(container, textvariable=self.model_var, width=70, state="readonly")
        self.model_combo.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 8))

        browse_btn = ttk.Button(container, text="Browse...", command=self.browse_model)
        browse_btn.grid(row=2, column=3, sticky="ew")

        ttk.Label(container, text="Actor model B (compare):").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.model_b_combo = ttk.Combobox(container, textvariable=self.model_b_var, width=70, state="readonly")
        self.model_b_combo.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(8, 0))

        browse_b_btn = ttk.Button(container, text="Browse B...", command=self.browse_model_b)
        browse_b_btn.grid(row=3, column=3, sticky="ew", pady=(8, 0))

        ttk.Label(container, text="Actor model C (compare):").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.model_c_combo = ttk.Combobox(container, textvariable=self.model_c_var, width=70, state="readonly")
        self.model_c_combo.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(8, 0))

        browse_c_btn = ttk.Button(container, text="Browse C...", command=self.browse_model_c)
        browse_c_btn.grid(row=4, column=3, sticky="ew", pady=(8, 0))

        ttk.Label(container, text="Actor model D (compare):").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.model_d_combo = ttk.Combobox(container, textvariable=self.model_d_var, width=70, state="readonly")
        self.model_d_combo.grid(row=5, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(8, 0))

        browse_d_btn = ttk.Button(container, text="Browse D...", command=self.browse_model_d)
        browse_d_btn.grid(row=5, column=3, sticky="ew", pady=(8, 0))

        refresh_btn = ttk.Button(container, text="Refresh Artifacts", command=self.refresh_models)
        refresh_btn.grid(row=6, column=3, sticky="ew", pady=(8, 0))

        ttk.Label(container, text="Episodes:").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(container, textvariable=self.episodes_var, width=10).grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(container, text="Max steps:").grid(row=6, column=1, sticky="e", pady=(8, 0))
        ttk.Entry(container, textvariable=self.max_steps_var, width=10).grid(row=6, column=2, sticky="w", pady=(8, 0))

        ttk.Label(container, text="Frame delay (ms):").grid(row=7, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(container, textvariable=self.frame_delay_ms_var, width=10).grid(row=7, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(container, text="Seed:").grid(row=7, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(container, textvariable=self.seed_var, width=10).grid(row=7, column=3, sticky="w", pady=(8, 0))

        ttk.Label(container, text="Replay episodes:").grid(row=8, column=2, sticky="e", pady=(8, 0))
        ttk.Label(container, text="1 (single saved model)").grid(row=8, column=3, sticky="w", pady=(8, 0))

        ttk.Checkbutton(container, text="Use 2D visualizer", variable=self.use_2d_var).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))

        buttons = ttk.Frame(container)
        buttons.grid(row=9, column=0, columnspan=4, sticky="ew", pady=(16, 8))
        buttons.columnconfigure((0, 1, 2, 3), weight=1)

        self.run_model_btn = ttk.Button(buttons, text="Run Loaded Model", command=self.run_loaded_model)
        self.run_model_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.run_random_btn = ttk.Button(buttons, text="Run Random Policy", command=self.run_random_policy)
        self.run_random_btn.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self.run_compare_btn = ttk.Button(buttons, text="Run Compare Models (2-4)", command=self.run_side_by_side_models)
        self.run_compare_btn.grid(row=0, column=2, sticky="ew", padx=(0, 8))

        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop_run, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=3, sticky="ew")

        log_header = ttk.Frame(container)
        log_header.grid(row=10, column=0, columnspan=4, sticky="ew")
        log_header.columnconfigure(0, weight=1)

        ttk.Label(log_header, text="Episode rewards:").grid(row=0, column=0, sticky="w")

        copy_btn = ttk.Button(log_header, text="Copy Console", command=self._copy_log_to_clipboard)
        copy_btn.grid(row=0, column=1, sticky="e", padx=(8, 0))

        clear_btn = ttk.Button(log_header, text="Clear Console", command=self._clear_log)
        clear_btn.grid(row=0, column=2, sticky="e", padx=(8, 0))

        plot_btn = ttk.Button(log_header, text="Plot Compare Stats", command=self._plot_compare_stats)
        plot_btn.grid(row=0, column=3, sticky="e", padx=(8, 0))

        log_frame = ttk.Frame(container)
        log_frame.grid(row=11, column=0, columnspan=4, sticky="nsew", pady=(6, 8))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.results = tk.Text(log_frame, height=12, wrap="none", state="disabled")
        self.results.grid(row=0, column=0, sticky="nsew")
        self.results.bind("<Control-a>", self._select_all_log)

        log_scroll_y = ttk.Scrollbar(log_frame, orient="vertical", command=self.results.yview)
        log_scroll_y.grid(row=0, column=1, sticky="ns")
        self.results.configure(yscrollcommand=log_scroll_y.set)

        status = ttk.Label(container, textvariable=self.status_var)
        status.grid(row=12, column=0, columnspan=4, sticky="w")

        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=1)
        container.rowconfigure(11, weight=1)

    def refresh_models(self) -> None:
        artifacts_dir = pathlib.Path("artifacts")
        candidates = sorted(artifacts_dir.glob("*.weights.h5"), reverse=True)
        candidates = [
            path
            for path in candidates
            if "actor" in path.name.lower() and "target_actor" not in path.name.lower()
        ]
        model_paths = [str(path) for path in candidates]
        self.model_combo["values"] = model_paths
        self.model_b_combo["values"] = model_paths
        self.model_c_combo["values"] = model_paths
        self.model_d_combo["values"] = model_paths
        if model_paths and self.model_var.get() not in model_paths:
            self.model_var.set(model_paths[0])
        if model_paths and self.model_b_var.get() not in model_paths:
            self.model_b_var.set(model_paths[0])
        if model_paths and self.model_c_var.get() not in model_paths:
            self.model_c_var.set("")
        if model_paths and self.model_d_var.get() not in model_paths:
            self.model_d_var.set("")
        if not model_paths:
            self.model_var.set("")
            self.model_b_var.set("")
            self.model_c_var.set("")
            self.model_d_var.set("")
        self.status_var.set(f"Found {len(model_paths)} actor model(s) in artifacts")

    def browse_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select actor weights",
            filetypes=[("Keras weights", "*.weights.h5"), ("All files", "*.*")],
        )
        if selected:
            current_values = list(self.model_combo["values"])
            if selected not in current_values:
                current_values.append(selected)
                self.model_combo["values"] = current_values
            self.model_var.set(selected)

    def browse_model_b(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select actor weights (model B)",
            filetypes=[("Keras weights", "*.weights.h5"), ("All files", "*.*")],
        )
        if selected:
            self._register_selected_model(selected)
            self.model_b_var.set(selected)

    def browse_model_c(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select actor weights (model C)",
            filetypes=[("Keras weights", "*.weights.h5"), ("All files", "*.*")],
        )
        if selected:
            self._register_selected_model(selected)
            self.model_c_var.set(selected)

    def browse_model_d(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select actor weights (model D)",
            filetypes=[("Keras weights", "*.weights.h5"), ("All files", "*.*")],
        )
        if selected:
            self._register_selected_model(selected)
            self.model_d_var.set(selected)

    def _register_selected_model(self, selected: str) -> None:
        combos = [self.model_combo, self.model_b_combo, self.model_c_combo, self.model_d_combo]
        for combo in combos:
            current_values = list(combo["values"])
            if selected not in current_values:
                current_values.append(selected)
                combo["values"] = current_values

    def run_loaded_model(self) -> None:
        model_path = self.model_var.get().strip()
        if not model_path:
            messagebox.showerror("Missing model", "Select an actor weights file first.")
            return
        if not pathlib.Path(model_path).exists():
            messagebox.showerror("Missing model", "Selected model file does not exist.")
            return

        resolved_model_path = model_path
        self.model_var.set(resolved_model_path)
        self.start_episode_var.set("1")
        self.episodes_var.set("1")

        guessed_joints = self._guess_joints_from_model_path(model_path)
        preferred_env_id = JOINTS_TO_ENV_ID.get(guessed_joints) if guessed_joints is not None else None
        compatible_env_id, reason = self._detect_model_compatible_env(resolved_model_path, preferred_env_id=preferred_env_id)
        if compatible_env_id is None:
            details = reason or "unknown mismatch"
            if "MuJoCo not installed" in details and "weights shape mismatch" in details:
                details = (
                    f"Model does not match {TRIPLE_PENDULUM_ENV_ID} or Pendulum-v1, and InvertedDoublePendulum-v5 is unavailable because MuJoCo is missing. "
                    "Install with: pip install \"gymnasium[mujoco]\""
                )
            elif "MuJoCo not installed" in details:
                details = "InvertedDoublePendulum-v5 is unavailable because MuJoCo is missing. Install with: pip install \"gymnasium[mujoco]\""
            messagebox.showerror(
                "Incompatible model",
                f"This model does not match supported viewer environments ({TRIPLE_PENDULUM_ENV_ID}, InvertedDoublePendulum-v5, or Pendulum-v1). "
                f"Details: {details}",
            )
            return

        if self.env_id_var.get().strip() != compatible_env_id:
            self.env_id_var.set(compatible_env_id)

        detected_joints = ENV_ID_TO_JOINTS.get(compatible_env_id)
        if detected_joints is not None:
            self.status_var.set(f"Detected model environment: {compatible_env_id} (joints={detected_joints})")
        else:
            self.status_var.set(f"Detected model environment: {compatible_env_id}")

        self.start_run(use_model=True)

    def run_random_policy(self) -> None:
        self.start_run(use_model=False)

    def run_side_by_side_models(self) -> None:
        selected_models = [
            self.model_var.get().strip(),
            self.model_b_var.get().strip(),
            self.model_c_var.get().strip(),
            self.model_d_var.get().strip(),
        ]
        model_paths = [path for path in selected_models if path]
        if len(model_paths) < 2:
            messagebox.showerror("Missing model", "Select at least two models (A/B/C/D) for comparison.")
            return
        if len(model_paths) > 4:
            messagebox.showerror("Too many models", "Compare supports up to 4 models.")
            return

        unique_paths = list(dict.fromkeys(model_paths))
        if len(unique_paths) != len(model_paths):
            messagebox.showwarning("Duplicate models", "Choose distinct model files when comparing.")
            return

        missing_models = [path for path in model_paths if not pathlib.Path(path).exists()]
        if missing_models:
            messagebox.showerror("Missing model", f"These model files do not exist: {', '.join(missing_models)}")
            return
        if not bool(self.use_2d_var.get()):
            messagebox.showerror("2D required", "Multi-model comparison requires 'Use 2D visualizer' enabled.")
            return

        preferred_env = self.env_id_var.get().strip() or TRIPLE_PENDULUM_ENV_ID
        resolved_envs: list[str] = []
        for idx, model_path in enumerate(model_paths):
            env_match, reason = self._detect_model_compatible_env(model_path, preferred_env_id=preferred_env)
            if env_match is None:
                messagebox.showerror(
                    "Incompatible model",
                    f"Model {idx + 1} could not be loaded in supported envs. Details: {reason or 'unknown'}",
                )
                return
            resolved_envs.append(env_match)

        env_set = set(resolved_envs)
        if len(env_set) != 1:
            messagebox.showerror(
                "Env mismatch",
                "Selected models resolve to different environments. Choose models trained for the same environment.",
            )
            return

        resolved_env = resolved_envs[0]
        self.env_id_var.set(resolved_env)
        self.start_compare_run(model_paths=model_paths, env_id=resolved_env)

    def start_run(self, use_model: bool) -> None:
        if self.runner_thread and self.runner_thread.is_alive():
            messagebox.showwarning("Already running", "An episode run is already in progress.")
            return

        self.compare_plot_data = None

        try:
            episodes = int(self.episodes_var.get())
            max_steps = int(self.max_steps_var.get())
            frame_delay_ms = int(self.frame_delay_ms_var.get())
            seed = int(self.seed_var.get())
            start_episode = int(self.start_episode_var.get())
            if episodes <= 0 or max_steps <= 0 or start_episode <= 0 or frame_delay_ms < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Episodes, max steps, frame delay, seed, and start episode must be valid integers. Episodes/max steps/start episode must be > 0 and frame delay must be >= 0.",
            )
            return

        self.stop_event.clear()
        self._set_running_state(True)

        self.runner_thread = threading.Thread(
            target=self._run_episodes,
            kwargs={
                "use_model": use_model,
                "episodes": episodes,
                "max_steps": max_steps,
                "frame_delay_ms": frame_delay_ms,
                "use_2d": bool(self.use_2d_var.get()),
                "seed": seed,
                "start_episode": start_episode,
                "model_path": self.model_var.get().strip(),
                "env_id": self.env_id_var.get().strip(),
            },
            daemon=True,
        )
        self.runner_thread.start()

    def start_compare_run(self, model_paths: list[str], env_id: str) -> None:
        if self.runner_thread and self.runner_thread.is_alive():
            messagebox.showwarning("Already running", "An episode run is already in progress.")
            return

        self.compare_plot_data = None

        try:
            episodes = int(self.episodes_var.get())
            max_steps = int(self.max_steps_var.get())
            frame_delay_ms = int(self.frame_delay_ms_var.get())
            seed = int(self.seed_var.get())
            start_episode = int(self.start_episode_var.get())
            if episodes <= 0 or max_steps <= 0 or start_episode <= 0 or frame_delay_ms < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Episodes, max steps, frame delay, seed, and start episode must be valid integers. Episodes/max steps/start episode must be > 0 and frame delay must be >= 0.",
            )
            return

        self.stop_event.clear()
        self._set_running_state(True)

        self.runner_thread = threading.Thread(
            target=self._run_compare_episodes,
            kwargs={
                "episodes": episodes,
                "max_steps": max_steps,
                "frame_delay_ms": frame_delay_ms,
                "seed": seed,
                "start_episode": start_episode,
                "model_paths": model_paths,
                "env_id": env_id,
            },
            daemon=True,
        )
        self.runner_thread.start()

    def stop_run(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping after current environment step...")

    def _set_running_state(self, running: bool) -> None:
        self.run_model_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.run_random_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.run_compare_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _append_result(self, episode_idx: int, reward: float) -> None:
        self._append_log_line(f"Episode {episode_idx:03d} | Reward: {reward:.2f}")

    def _append_compare_result(self, episode_idx: int, reward_a: float, reward_b: float) -> None:
        self._append_log_line(
            f"Episode {episode_idx:03d} | Model A: {reward_a:.2f} | Model B: {reward_b:.2f} | Delta(A-B): {reward_a - reward_b:.2f}"
        )

    def _append_compare_result_with_stats(
        self,
        episode_idx: int,
        model_summaries: list[str],
    ) -> None:
        self._append_log_line(f"Episode {episode_idx:03d} | " + " | ".join(model_summaries))

    def _append_log_line(self, line: str) -> None:
        self.results.configure(state="normal")
        self.results.insert(tk.END, line + "\n")
        self.results.see(tk.END)
        self.results.configure(state="disabled")

    def _select_all_log(self, _event: tk.Event) -> str:
        self.results.tag_add("sel", "1.0", tk.END)
        self.results.mark_set("insert", "1.0")
        self.results.see("insert")
        return "break"

    def _copy_log_to_clipboard(self) -> None:
        text = self.results.get("1.0", tk.END).strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("Console copied to clipboard")

    def _clear_log(self) -> None:
        self.results.configure(state="normal")
        self.results.delete("1.0", tk.END)
        self.results.configure(state="disabled")
        self.status_var.set("Console cleared")

    def _plot_compare_stats(self) -> None:
        if not self.compare_plot_data:
            messagebox.showinfo("No compare data", "Run a compare session first to generate stats.")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            messagebox.showerror("Missing dependency", "matplotlib is required to plot compare stats.")
            return

        labels = self.compare_plot_data["labels"]
        reward_histories = self.compare_plot_data["reward_histories"]
        balance_histories = self.compare_plot_data["balance_histories"]
        best_streak_histories = self.compare_plot_data["best_streak_histories"]
        reset_histories = self.compare_plot_data["reset_histories"]

        episodes = list(range(1, len(reward_histories[0]) + 1))
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

        for idx, label in enumerate(labels):
            axes[0][0].plot(episodes, reward_histories[idx], label=label)
            axes[0][1].plot(episodes, balance_histories[idx], label=label)
            axes[1][0].plot(episodes, best_streak_histories[idx], label=label)
            axes[1][1].plot(episodes, reset_histories[idx], label=label)

        axes[0][0].set_title("Episode Reward")
        axes[0][0].set_xlabel("Episode")
        axes[0][0].set_ylabel("Reward")
        axes[0][0].grid(True, alpha=0.3)
        axes[0][0].legend(loc="best")

        axes[0][1].set_title("Balance Time (First Failure Step)")
        axes[0][1].set_xlabel("Episode")
        axes[0][1].set_ylabel("Steps")
        axes[0][1].grid(True, alpha=0.3)
        axes[0][1].legend(loc="best")

        axes[1][0].set_title("Best Continuous Streak")
        axes[1][0].set_xlabel("Episode")
        axes[1][0].set_ylabel("Steps")
        axes[1][0].grid(True, alpha=0.3)
        axes[1][0].legend(loc="best")

        axes[1][1].set_title("Resets Per Episode")
        axes[1][1].set_xlabel("Episode")
        axes[1][1].set_ylabel("Resets")
        axes[1][1].grid(True, alpha=0.3)
        axes[1][1].legend(loc="best")

        plt.show()

    @staticmethod
    def _build_compare_summary_text(
        labels: list[str],
        reward_histories: list[list[float]],
        balance_histories: list[list[int]],
        best_streak_histories: list[list[int]],
        reset_histories: list[list[int]],
    ) -> str:
        if not labels or any(not hist for hist in reward_histories):
            return ""

        avg_rewards = [float(np.mean(hist)) for hist in reward_histories]
        avg_balances = [float(np.mean(hist)) for hist in balance_histories]
        avg_streaks = [float(np.mean(hist)) for hist in best_streak_histories]
        avg_resets = [float(np.mean(hist)) for hist in reset_histories]
        balance_scores = [bal + 0.5 * streak - 8.0 * reset for bal, streak, reset in zip(avg_balances, avg_streaks, avg_resets)]

        leader_idx = int(np.argmax(balance_scores))
        lines: list[str] = [
            (
                "Leader by balance score: "
                f"{labels[leader_idx]} "
                f"(score={balance_scores[leader_idx]:.2f}, "
                f"avg_balance={avg_balances[leader_idx]:.1f}, "
                f"avg_streak={avg_streaks[leader_idx]:.1f}, "
                f"avg_resets={avg_resets[leader_idx]:.2f})"
            )
        ]

        for idx, label in enumerate(labels):
            lines.append(
                (
                    f"{label}: avgR={np.mean(reward_histories[idx]):.1f}, "
                    f"avg/best balance={np.mean(balance_histories[idx]):.1f}/{np.max(balance_histories[idx])}, "
                    f"avg best streak={np.mean(best_streak_histories[idx]):.1f}, "
                    f"avg resets={np.mean(reset_histories[idx]):.2f}"
                )
            )

        return " | ".join(lines)

    def _open_2d_window(self, env_id: str) -> None:
        if self.sim_window is not None and self.sim_window.winfo_exists():
            self.sim_window.destroy()

        self.sim_window = tk.Toplevel(self)
        self.sim_window.title(f"2D Visualizer - {env_id}")
        self.sim_window.geometry("940x620")

        self.sim_canvas = tk.Canvas(self.sim_window, width=900, height=560, bg="white")
        self.sim_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.sim_canvas_left = None
        self.sim_canvas_right = None
        self.sim_compare_canvases = []

    def _open_2d_compare_window(self, env_id: str, model_labels: list[str]) -> None:
        if self.sim_window is not None and self.sim_window.winfo_exists():
            self.sim_window.destroy()

        self.sim_window = tk.Toplevel(self)
        self.sim_window.title(f"2D Multi-Model Compare - {env_id}")
        self.sim_window.geometry("1460x840")

        frame = ttk.Frame(self.sim_window, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)
        for col_idx in range(2):
            frame.columnconfigure(col_idx, weight=1)
        for row_idx in range(4):
            frame.rowconfigure(row_idx, weight=1)

        self.sim_compare_canvases = []
        for idx, label in enumerate(model_labels):
            row_base = (idx // 2) * 2
            col = idx % 2
            ttk.Label(frame, text=f"{label}", anchor="w").grid(row=row_base, column=col, sticky="ew", padx=6)
            canvas = tk.Canvas(frame, width=680, height=320, bg="white")
            canvas.grid(row=row_base + 1, column=col, sticky="nsew", padx=6, pady=(4, 8))
            self.sim_compare_canvases.append(canvas)

        # Keep legacy fields unset in multi-compare mode.
        self.sim_canvas_left = None
        self.sim_canvas_right = None

        self.compare_stats_var.set("Comparison stats will appear here as episodes complete.")
        compare_stats_label = ttk.Label(
            frame,
            textvariable=self.compare_stats_var,
            anchor="w",
            justify="left",
            wraplength=1420,
        )
        compare_stats_label.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.sim_canvas = None

    def _draw_2d_state_on_canvas(self, canvas: tk.Canvas, env_id: str, state: np.ndarray, episode_idx: int, step_idx: int) -> None:
        if self.sim_window is None or not self.sim_window.winfo_exists():
            return

        canvas.delete("all")
        width = int(canvas.winfo_width() or 900)
        height = int(canvas.winfo_height() or 560)

        canvas.create_text(12, 12, anchor="nw", text=f"{env_id} | Episode {episode_idx} | Step {step_idx}", font=("Segoe UI", 11, "bold"))

        if env_id == "Pendulum-v1":
            pivot_x = width // 2
            pivot_y = 130
            length = min(width, height) * 0.30

            theta = float(np.arctan2(state[1], state[0]))
            bob_x = pivot_x + length * np.sin(theta)
            bob_y = pivot_y - length * np.cos(theta)

            canvas.create_line(0, pivot_y + length + 40, width, pivot_y + length + 40, fill="#b0b0b0", width=2)
            canvas.create_line(pivot_x, pivot_y, bob_x, bob_y, fill="#1f77b4", width=8)
            canvas.create_oval(pivot_x - 8, pivot_y - 8, pivot_x + 8, pivot_y + 8, fill="#333", outline="")
            canvas.create_oval(bob_x - 18, bob_y - 18, bob_x + 18, bob_y + 18, fill="#ff7f0e", outline="")
            canvas.create_text(12, 36, anchor="nw", text="2D pendulum: swing-up + balance", font=("Segoe UI", 10))
            return

        if env_id.startswith("InvertedTriplePendulum"):
            rail_y = int(height * 0.70)
            center_x = width // 2
            scale_x = min(width * 0.25, 220)
            cart_w = 95
            cart_h = 40

            cart_x = float(state[0])
            theta1 = float(np.arctan2(state[1], state[4]))
            theta2 = float(np.arctan2(state[2], state[5]))
            theta3 = float(np.arctan2(state[3], state[6]))

            cart_cx = np.clip(center_x + cart_x * scale_x, 70, width - 70)
            cart_left = cart_cx - cart_w / 2
            cart_right = cart_cx + cart_w / 2
            cart_top = rail_y - cart_h
            cart_bottom = rail_y

            canvas.create_line(40, rail_y, width - 40, rail_y, fill="#888", width=4)
            canvas.create_rectangle(cart_left, cart_top, cart_right, cart_bottom, fill="#4e79a7", outline="")

            wheel_r = 10
            canvas.create_oval(cart_left + 10 - wheel_r, rail_y - wheel_r, cart_left + 10 + wheel_r, rail_y + wheel_r, fill="#222", outline="")
            canvas.create_oval(cart_right - 10 - wheel_r, rail_y - wheel_r, cart_right - 10 + wheel_r, rail_y + wheel_r, fill="#222", outline="")

            pivot_x = cart_cx
            pivot_y = cart_top
            l1 = min(width, height) * 0.21
            l2 = min(width, height) * 0.19
            l3 = min(width, height) * 0.17

            p1_x = pivot_x + l1 * np.sin(theta1)
            p1_y = pivot_y - l1 * np.cos(theta1)
            p2_x = p1_x + l2 * np.sin(theta2)
            p2_y = p1_y - l2 * np.cos(theta2)
            p3_x = p2_x + l3 * np.sin(theta3)
            p3_y = p2_y - l3 * np.cos(theta3)

            canvas.create_line(pivot_x, pivot_y, p1_x, p1_y, fill="#f28e2b", width=8)
            canvas.create_line(p1_x, p1_y, p2_x, p2_y, fill="#e15759", width=7)
            canvas.create_line(p2_x, p2_y, p3_x, p3_y, fill="#76b7b2", width=6)
            canvas.create_oval(pivot_x - 7, pivot_y - 7, pivot_x + 7, pivot_y + 7, fill="#333", outline="")
            canvas.create_oval(p1_x - 9, p1_y - 9, p1_x + 9, p1_y + 9, fill="#333", outline="")
            canvas.create_oval(p2_x - 10, p2_y - 10, p2_x + 10, p2_y + 10, fill="#333", outline="")
            canvas.create_oval(p3_x - 12, p3_y - 12, p3_x + 12, p3_y + 12, fill="#59a14f", outline="")
            canvas.create_text(12, 36, anchor="nw", text="2D triple pendulum on cart: move cart + stabilize all 3 links", font=("Segoe UI", 10))
            return

        rail_y = int(height * 0.70)
        center_x = width // 2
        scale_x = min(width * 0.25, 220)
        cart_w = 95
        cart_h = 40

        cart_x = float(state[0])
        theta1 = float(np.arctan2(state[1], state[3]))
        theta2 = float(np.arctan2(state[2], state[4]))

        cart_cx = np.clip(center_x + cart_x * scale_x, 70, width - 70)
        cart_left = cart_cx - cart_w / 2
        cart_right = cart_cx + cart_w / 2
        cart_top = rail_y - cart_h
        cart_bottom = rail_y

        canvas.create_line(40, rail_y, width - 40, rail_y, fill="#888", width=4)
        canvas.create_rectangle(cart_left, cart_top, cart_right, cart_bottom, fill="#4e79a7", outline="")

        wheel_r = 10
        canvas.create_oval(cart_left + 10 - wheel_r, rail_y - wheel_r, cart_left + 10 + wheel_r, rail_y + wheel_r, fill="#222", outline="")
        canvas.create_oval(cart_right - 10 - wheel_r, rail_y - wheel_r, cart_right - 10 + wheel_r, rail_y + wheel_r, fill="#222", outline="")

        pivot_x = cart_cx
        pivot_y = cart_top
        l1 = min(width, height) * 0.24
        l2 = min(width, height) * 0.21

        p1_x = pivot_x + l1 * np.sin(theta1)
        p1_y = pivot_y - l1 * np.cos(theta1)
        p2_x = p1_x + l2 * np.sin(theta2)
        p2_y = p1_y - l2 * np.cos(theta2)

        canvas.create_line(pivot_x, pivot_y, p1_x, p1_y, fill="#f28e2b", width=8)
        canvas.create_line(p1_x, p1_y, p2_x, p2_y, fill="#e15759", width=7)
        canvas.create_oval(pivot_x - 7, pivot_y - 7, pivot_x + 7, pivot_y + 7, fill="#333", outline="")
        canvas.create_oval(p1_x - 9, p1_y - 9, p1_x + 9, p1_y + 9, fill="#333", outline="")
        canvas.create_oval(p2_x - 12, p2_y - 12, p2_x + 12, p2_y + 12, fill="#59a14f", outline="")
        canvas.create_text(12, 36, anchor="nw", text="2D double pendulum on cart: move cart + balance up", font=("Segoe UI", 10))

    def _draw_2d_state(self, env_id: str, state: np.ndarray, episode_idx: int, step_idx: int) -> None:
        if self.sim_window is None or not self.sim_window.winfo_exists() or self.sim_canvas is None:
            return
        self._draw_2d_state_on_canvas(self.sim_canvas, env_id, state, episode_idx, step_idx)

    def _draw_2d_compare_states(
        self,
        env_id: str,
        states: list[np.ndarray],
        episode_idx: int,
        step_idx: int,
    ) -> None:
        if self.sim_window is None or not self.sim_window.winfo_exists():
            return
        if not self.sim_compare_canvases:
            return
        for canvas, state in zip(self.sim_compare_canvases, states):
            self._draw_2d_state_on_canvas(canvas, env_id, state, episode_idx, step_idx)

    def _run_compare_episodes(
        self,
        episodes: int,
        max_steps: int,
        frame_delay_ms: int,
        seed: int,
        start_episode: int,
        model_paths: list[str],
        env_id: str,
    ) -> None:
        envs: list[gym.Env] = []
        try:
            register_triple_pendulum_env()
            for _ in model_paths:
                envs.append(gym.make(env_id))

            if any(not isinstance(env.action_space, gym.spaces.Box) for env in envs):
                raise ValueError("Selected environment must use a continuous Box action space.")

            ref_env = envs[0]
            num_states = ref_env.observation_space.shape[0]
            num_actions = ref_env.action_space.shape[0]
            upper_bound = ref_env.action_space.high.astype(np.float32)
            lower_bound = ref_env.action_space.low.astype(np.float32)

            actors: list[tf.keras.Model] = []
            model_labels = [pathlib.Path(path).name for path in model_paths]
            model_short_labels = [self._label_from_model_path(path) for path in model_paths]
            for model_path in model_paths:
                actor = get_actor(num_states=num_states, num_actions=num_actions, upper_bound=upper_bound)
                actor(np.zeros((1, num_states), dtype=np.float32), training=False)
                actor.load_weights(model_path)
                actors.append(actor)

            window_ready = threading.Event()

            def init_window() -> None:
                labeled_models = [f"{name}: {label}" for name, label in zip(model_short_labels, model_labels)]
                self._open_2d_compare_window(env_id, model_labels=labeled_models)
                window_ready.set()

            self.after(0, init_window)
            window_ready.wait(timeout=2)

            self.after(
                0,
                self.status_var.set,
                f"Running {episodes} compare episode(s) across {len(model_paths)} models in {env_id} from episode {start_episode}...",
            )

            reward_histories: list[list[float]] = [[] for _ in model_paths]
            balance_histories: list[list[int]] = [[] for _ in model_paths]
            best_streak_histories: list[list[int]] = [[] for _ in model_paths]
            reset_histories: list[list[int]] = [[] for _ in model_paths]

            for local_idx in range(episodes):
                if self.stop_event.is_set():
                    break

                episode_idx = start_episode + local_idx
                states: list[np.ndarray] = []
                for model_idx, env in enumerate(envs):
                    state, _ = env.reset(seed=seed + episode_idx - 1 + model_idx * 10000)
                    states.append(state)

                episode_rewards = [0.0 for _ in model_paths]
                first_failures = [max_steps for _ in model_paths]
                resets = [0 for _ in model_paths]
                current_streaks = [0 for _ in model_paths]
                best_streaks = [0 for _ in model_paths]

                self.after(0, self._draw_2d_compare_states, env_id, [state.copy() for state in states], episode_idx, 0)

                for step_idx in range(max_steps):
                    if self.stop_event.is_set():
                        break

                    for model_idx, (actor, env) in enumerate(zip(actors, envs)):
                        state_tensor = tf.convert_to_tensor(states[model_idx][np.newaxis, :], dtype=tf.float32)
                        action = tf.squeeze(actor(state_tensor, training=False), axis=0).numpy()
                        action = np.clip(action, lower_bound, upper_bound).astype(np.float32)

                        next_state, reward, terminated, truncated, _ = env.step(action)
                        states[model_idx] = next_state
                        episode_rewards[model_idx] += float(reward)
                        current_streaks[model_idx] += 1

                        if terminated or truncated:
                            if first_failures[model_idx] == max_steps:
                                first_failures[model_idx] = step_idx + 1
                            resets[model_idx] += 1
                            best_streaks[model_idx] = max(best_streaks[model_idx], current_streaks[model_idx])
                            current_streaks[model_idx] = 0
                            reset_seed = seed + episode_idx * 100000 + step_idx + 1 + model_idx * 1000000
                            states[model_idx], _ = env.reset(seed=reset_seed)

                    self.after(0, self._draw_2d_compare_states, env_id, [state.copy() for state in states], episode_idx, step_idx + 1)
                    if frame_delay_ms > 0:
                        time.sleep(frame_delay_ms / 1000.0)

                for model_idx in range(len(model_paths)):
                    best_streaks[model_idx] = max(best_streaks[model_idx], current_streaks[model_idx])
                    reward_histories[model_idx].append(episode_rewards[model_idx])
                    balance_histories[model_idx].append(first_failures[model_idx])
                    best_streak_histories[model_idx].append(best_streaks[model_idx])
                    reset_histories[model_idx].append(resets[model_idx])

                summary_text = self._build_compare_summary_text(
                    labels=model_short_labels,
                    reward_histories=reward_histories,
                    balance_histories=balance_histories,
                    best_streak_histories=best_streak_histories,
                    reset_histories=reset_histories,
                )

                episode_summaries = []
                for model_idx, model_name in enumerate(model_short_labels):
                    episode_summaries.append(
                        (
                            f"{model_name}: R={episode_rewards[model_idx]:.2f}, "
                            f"Bal={first_failures[model_idx]}, "
                            f"BestStreak={best_streaks[model_idx]}, "
                            f"Resets={resets[model_idx]}"
                        )
                    )

                self.after(
                    0,
                    self._append_compare_result_with_stats,
                    episode_idx,
                    episode_summaries,
                )
                self.after(0, self.compare_stats_var.set, summary_text)

                self.compare_plot_data = {
                    "labels": model_short_labels.copy(),
                    "reward_histories": [hist.copy() for hist in reward_histories],
                    "balance_histories": [hist.copy() for hist in balance_histories],
                    "best_streak_histories": [hist.copy() for hist in best_streak_histories],
                    "reset_histories": [hist.copy() for hist in reset_histories],
                }

            if self.stop_event.is_set():
                self.after(0, self.status_var.set, "Stopped")
            else:
                self.after(0, self.status_var.set, "Done")
        except Exception as exc:  # noqa: BLE001
            self.after(0, messagebox.showerror, "Run error", str(exc))
            self.after(0, self.status_var.set, "Error")
        finally:
            for env in envs:
                env.close()
            self.after(0, self._set_running_state, False)

    def _run_episodes(
        self,
        use_model: bool,
        episodes: int,
        max_steps: int,
        frame_delay_ms: int,
        use_2d: bool,
        seed: int,
        start_episode: int,
        model_path: str,
        env_id: str,
    ) -> None:
        actor_model = None
        env = None
        try:
            register_triple_pendulum_env()
            env_candidates = [env_id, TRIPLE_PENDULUM_ENV_ID, "InvertedDoublePendulum-v5", "Pendulum-v1"]
            # Preserve order while removing duplicates.
            env_candidates = list(dict.fromkeys(env_candidates))

            resolved_env_id = env_id
            upper_bound = None
            lower_bound = None
            load_error: Exception | None = None

            env_create_error: Exception | None = None

            for candidate_env_id in env_candidates:
                try:
                    if use_2d:
                        candidate_env = gym.make(candidate_env_id)
                    else:
                        try:
                            candidate_env = gym.make(candidate_env_id, render_mode="human", width=1280, height=720)
                        except TypeError:
                            candidate_env = gym.make(candidate_env_id, render_mode="human")
                except Exception as exc:  # noqa: BLE001
                    env_create_error = exc
                    continue
                if not isinstance(candidate_env.action_space, gym.spaces.Box):
                    candidate_env.close()
                    continue

                candidate_num_states = candidate_env.observation_space.shape[0]
                candidate_num_actions = candidate_env.action_space.shape[0]
                candidate_upper_bound = candidate_env.action_space.high.astype(np.float32)
                candidate_lower_bound = candidate_env.action_space.low.astype(np.float32)

                if use_model:
                    candidate_actor = get_actor(
                        num_states=candidate_num_states,
                        num_actions=candidate_num_actions,
                        upper_bound=candidate_upper_bound,
                    )
                    # Build model variables before loading weights.
                    candidate_actor(np.zeros((1, candidate_num_states), dtype=np.float32), training=False)
                    try:
                        candidate_actor.load_weights(model_path)
                    except Exception as exc:  # noqa: BLE001
                        load_error = exc
                        candidate_env.close()
                        continue
                    actor_model = candidate_actor

                env = candidate_env
                resolved_env_id = candidate_env_id
                upper_bound = candidate_upper_bound
                lower_bound = candidate_lower_bound
                break

            if env is None:
                if use_model and load_error is not None:
                    raise ValueError(
                        "Could not load this model for any supported environment shape "
                        f"({TRIPLE_PENDULUM_ENV_ID}, InvertedDoublePendulum-v5, or Pendulum-v1). "
                        f"Last error: {self._summarize_exception(load_error)}"
                    ) from load_error
                if env_create_error is not None:
                    raise ValueError(
                        "Could not create a supported continuous-control environment for viewer. "
                        f"Last environment error: {env_create_error}"
                    ) from env_create_error
                raise ValueError("Could not create a supported continuous-control environment for viewer.")

            if use_model and resolved_env_id != env_id:
                self.after(0, self.env_id_var.set, resolved_env_id)

            if use_2d:
                window_ready = threading.Event()

                def init_window() -> None:
                    self._open_2d_window(resolved_env_id)
                    window_ready.set()

                self.after(0, init_window)
                window_ready.wait(timeout=2)

            mode_text = "model" if use_model else "random policy"
            self.after(
                0,
                self.status_var.set,
                f"Running {episodes} episode(s) with {mode_text} in {resolved_env_id} from episode {start_episode}...",
            )

            for local_idx in range(episodes):
                if self.stop_event.is_set():
                    break

                episode_idx = start_episode + local_idx
                state, _ = env.reset(seed=seed + episode_idx - 1)
                episode_reward = 0.0

                if use_2d:
                    self.after(0, self._draw_2d_state, resolved_env_id, state.copy(), episode_idx, 0)

                for step_idx in range(max_steps):
                    if self.stop_event.is_set():
                        break

                    if actor_model is not None:
                        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
                        action_value = tf.squeeze(actor_model(state_tensor, training=False), axis=0).numpy()
                        action = np.clip(action_value, lower_bound, upper_bound).astype(np.float32)
                    else:
                        action = env.action_space.sample().astype(np.float32)

                    state, reward, terminated, truncated, _ = env.step(action)
                    if use_2d:
                        self.after(0, self._draw_2d_state, resolved_env_id, state.copy(), episode_idx, step_idx + 1)
                    if frame_delay_ms > 0:
                        time.sleep(frame_delay_ms / 1000.0)
                    episode_reward += float(reward)
                    if terminated or truncated:
                        # Continue showing motion until max_steps by resetting the env.
                        state, _ = env.reset(seed=seed + episode_idx * 100000 + step_idx + 1)
                        if use_2d:
                            self.after(0, self._draw_2d_state, resolved_env_id, state.copy(), episode_idx, step_idx + 1)

                self.after(0, self._append_result, episode_idx, episode_reward)

            if self.stop_event.is_set():
                self.after(0, self.status_var.set, "Stopped")
            else:
                self.after(0, self.status_var.set, "Done")
        except Exception as exc:  # noqa: BLE001
            self.after(0, messagebox.showerror, "Run error", str(exc))
            self.after(0, self.status_var.set, "Error")
        finally:
            if env is not None:
                env.close()
            self.after(0, self._set_running_state, False)


def main() -> None:
    app = PendulumViewerApp()
    app.mainloop()


if __name__ == "__main__":
    main()

import pathlib
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import gymnasium as gym
import numpy as np
import tensorflow as tf

from train import get_actor

tf.get_logger().setLevel("ERROR")


class PendulumViewerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Pendulum Model Viewer")
        self.geometry("760x520")

        self.stop_event = threading.Event()
        self.runner_thread: threading.Thread | None = None

        self.model_var = tk.StringVar()
        self.episodes_var = tk.StringVar(value="3")
        self.max_steps_var = tk.StringVar(value="200")
        self.seed_var = tk.StringVar(value="42")
        self.start_episode_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self.refresh_models()

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text="DDPG Pendulum Episode Viewer", font=("Segoe UI", 14, "bold"))
        title.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))

        ttk.Label(container, text="Actor model weights:").grid(row=1, column=0, sticky="w")
        self.model_combo = ttk.Combobox(container, textvariable=self.model_var, width=70, state="readonly")
        self.model_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 8))

        browse_btn = ttk.Button(container, text="Browse...", command=self.browse_model)
        browse_btn.grid(row=1, column=3, sticky="ew")

        refresh_btn = ttk.Button(container, text="Refresh Artifacts", command=self.refresh_models)
        refresh_btn.grid(row=2, column=3, sticky="ew", pady=(8, 0))

        ttk.Label(container, text="Episodes:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(container, textvariable=self.episodes_var, width=10).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(container, text="Max steps:").grid(row=2, column=1, sticky="e", pady=(8, 0))
        ttk.Entry(container, textvariable=self.max_steps_var, width=10).grid(row=2, column=2, sticky="w", pady=(8, 0))

        ttk.Label(container, text="Seed:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(container, textvariable=self.seed_var, width=10).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(container, text="Start episode:").grid(row=3, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(container, textvariable=self.start_episode_var, width=10).grid(row=3, column=3, sticky="w", pady=(8, 0))

        buttons = ttk.Frame(container)
        buttons.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(16, 8))
        buttons.columnconfigure((0, 1, 2), weight=1)

        self.run_model_btn = ttk.Button(buttons, text="Run Loaded Model", command=self.run_loaded_model)
        self.run_model_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.run_random_btn = ttk.Button(buttons, text="Run Random Policy", command=self.run_random_policy)
        self.run_random_btn.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop_run, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=2, sticky="ew")

        ttk.Label(container, text="Episode rewards:").grid(row=5, column=0, columnspan=4, sticky="w")

        self.results = tk.Listbox(container, height=12)
        self.results.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=(6, 8))

        status = ttk.Label(container, textvariable=self.status_var)
        status.grid(row=7, column=0, columnspan=4, sticky="w")

        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=1)
        container.rowconfigure(6, weight=1)

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
        if model_paths and self.model_var.get() not in model_paths:
            self.model_var.set(model_paths[0])
        if not model_paths:
            self.model_var.set("")
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

    def run_loaded_model(self) -> None:
        model_path = self.model_var.get().strip()
        if not model_path:
            messagebox.showerror("Missing model", "Select an actor weights file first.")
            return
        if not pathlib.Path(model_path).exists():
            messagebox.showerror("Missing model", "Selected model file does not exist.")
            return
        self.start_run(use_model=True)

    def run_random_policy(self) -> None:
        self.start_run(use_model=False)

    def start_run(self, use_model: bool) -> None:
        if self.runner_thread and self.runner_thread.is_alive():
            messagebox.showwarning("Already running", "An episode run is already in progress.")
            return

        try:
            episodes = int(self.episodes_var.get())
            max_steps = int(self.max_steps_var.get())
            seed = int(self.seed_var.get())
            start_episode = int(self.start_episode_var.get())
            if episodes <= 0 or max_steps <= 0 or start_episode <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Episodes, max steps, seed, and start episode must be valid integers. Episodes/max steps/start episode must be > 0.",
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
                "seed": seed,
                "start_episode": start_episode,
                "model_path": self.model_var.get().strip(),
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
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _append_result(self, episode_idx: int, reward: float) -> None:
        self.results.insert(tk.END, f"Episode {episode_idx:03d} | Reward: {reward:.2f}")
        self.results.see(tk.END)

    def _run_episodes(
        self,
        use_model: bool,
        episodes: int,
        max_steps: int,
        seed: int,
        start_episode: int,
        model_path: str,
    ) -> None:
        actor_model = None
        env = None
        try:
            env = gym.make("Pendulum-v1", render_mode="human")
            num_states = env.observation_space.shape[0]
            upper_bound = float(env.action_space.high[0])

            if use_model:
                actor_model = get_actor(num_states=num_states, upper_bound=upper_bound)
                # Build model variables before loading weights.
                actor_model(np.zeros((1, num_states), dtype=np.float32), training=False)
                actor_model.load_weights(model_path)

            mode_text = "model" if use_model else "random policy"
            self.after(0, self.status_var.set, f"Running {episodes} episode(s) with {mode_text} from episode {start_episode}...")

            for local_idx in range(episodes):
                if self.stop_event.is_set():
                    break

                episode_idx = start_episode + local_idx
                state, _ = env.reset(seed=seed + episode_idx - 1)
                episode_reward = 0.0

                for _ in range(max_steps):
                    if self.stop_event.is_set():
                        break

                    if actor_model is not None:
                        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
                        action_value = tf.squeeze(actor_model(state_tensor, training=False)).numpy()
                        action = np.array([np.squeeze(action_value)], dtype=np.float32)
                    else:
                        action = env.action_space.sample().astype(np.float32)

                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += float(reward)
                    if terminated or truncated:
                        break

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

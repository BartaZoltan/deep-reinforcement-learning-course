"""Shared helpers for Session 5 (n-step Returns and Eligibility Traces)."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import urllib.request

import matplotlib.pyplot as plt
import numpy as np


N_ROWS = 4
N_COLS = 12
START_STATE = (N_ROWS - 1) * N_COLS


def load_session4_utils():
    """Load Session 4 plotting helpers from local path or GitHub fallback."""
    candidates = [
        Path("notebooks/sessions/4_td_learning/utils.py"),
        Path("../4_td_learning/utils.py"),
        Path("sessions/4_td_learning/utils.py"),
        Path("/content/notebooks/sessions/4_td_learning/utils.py"),
    ]

    utils_path = next((p for p in candidates if p.exists()), None)

    if utils_path is None:
        utils_path = Path("notebooks/sessions/4_td_learning/utils.py")
        utils_path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://raw.githubusercontent.com/BartaZoltan/deep-reinforcement-learning-course/main/"
            "notebooks/sessions/4_td_learning/utils.py"
        )
        urllib.request.urlretrieve(url, utils_path)

    spec = importlib.util.spec_from_file_location("session4_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def plot_many_curves(
    curves: dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str = "Episode index",
    ylabel: str = "Value",
    hline: float | None = None,
    hline_label: str | None = None,
):
    """Plot several curves on one set of axes."""
    plt.figure(figsize=(7.2, 4.0))
    for label, values in curves.items():
        x = np.arange(1, len(values) + 1)
        plt.plot(x, values, linewidth=2.0, label=label)

    if hline is not None:
        plt.axhline(
            hline,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label=hline_label or "reference",
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_bars(metric_map: dict[str, float], *, title: str, ylabel: str):
    """Plot a compact bar chart for a small metric table."""
    labels = list(metric_map.keys())
    values = [metric_map[k] for k in labels]
    x = np.arange(len(labels))

    plt.figure(figsize=(6.8, 3.8))
    bars = plt.bar(x, values, color=["#60a5fa", "#f59e0b", "#34d399", "#f472b6"][: len(labels)])
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def summarize_last(values: np.ndarray, window: int = 50) -> float:
    """Return the trailing-window average of a curve."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values[-min(window, len(values)) :]))


def state_to_rc(state: int, n_cols: int = N_COLS):
    """Convert a flat CliffWalking state index to (row, col)."""
    return divmod(int(state), int(n_cols))


def safe_cliff_policy(state: int) -> int:
    """Hand-crafted policy that travels above the cliff before descending at the end."""
    row, col = state_to_rc(state)

    if row == N_ROWS - 1:
        return 0
    if row < N_ROWS - 2:
        return 2
    if row == N_ROWS - 2 and col < N_COLS - 1:
        return 1
    if row == N_ROWS - 2 and col == N_COLS - 1:
        return 2
    return 1


def is_cliff_state(state: int) -> bool:
    """Return whether a state index lies in the cliff region."""
    row, col = state_to_rc(state)
    return row == N_ROWS - 1 and 0 < col < N_COLS - 1


def valid_cliff_start_states(n_states: int) -> np.ndarray:
    """Return non-terminal, non-cliff states for exploring starts."""
    return np.array(
        [s for s in range(n_states) if (not is_cliff_state(s)) and s != (N_ROWS * N_COLS - 1)],
        dtype=int,
    )


def reset_cliff_with_exploring_start(env, rng: np.random.Generator, valid_states: np.ndarray) -> int:
    """Reset CliffWalking, then replace the start state with a sampled non-terminal state."""
    state, _ = env.reset(seed=int(rng.integers(0, 10_000_000)))
    del state
    sampled_state = int(rng.choice(valid_states))
    env.unwrapped.s = sampled_state
    return sampled_state


def generate_policy_episode(
    env,
    policy_fn,
    *,
    rng: np.random.Generator,
    max_steps: int = 200,
    exploring_starts: bool = False,
    valid_start_states: np.ndarray | None = None,
):
    """Generate one state-reward trajectory under a fixed policy."""
    if exploring_starts and env.spec is not None and env.spec.id == "CliffWalking-v0":
        if valid_start_states is None:
            valid_start_states = valid_cliff_start_states(env.observation_space.n)
        state = reset_cliff_with_exploring_start(env, rng, valid_start_states)
    else:
        state, _ = env.reset(seed=int(rng.integers(0, 10_000_000)))
        state = int(state)

    states = [int(state)]
    rewards = [0.0]

    for _ in range(max_steps):
        action = int(policy_fn(state))
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = int(next_state)
        rewards.append(float(reward))
        states.append(next_state)
        state = next_state
        if terminated or truncated:
            break

    return states, rewards


def greedy_policy_from_q(Q: np.ndarray) -> np.ndarray:
    """Extract a deterministic greedy policy from a tabular Q function."""
    return np.argmax(Q, axis=1)


def evaluate_greedy_return(
    env,
    greedy_actions: np.ndarray,
    *,
    n_eval_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 42,
):
    """Evaluate a deterministic greedy policy by mean episodic return."""
    rng = np.random.default_rng(seed)
    returns = []

    for _ in range(n_eval_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 10_000_000)))
        state = int(state)
        episode_return = 0.0

        for _ in range(max_steps):
            action = int(greedy_actions[state])
            state, reward, terminated, truncated, _ = env.step(action)
            state = int(state)
            episode_return += float(reward)
            if terminated or truncated:
                break

        returns.append(episode_return)

    returns = np.asarray(returns, dtype=float)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }


def export_cliffwalking_policy_rollout_gif(
    greedy_actions: np.ndarray,
    output_dir: str | Path,
    *,
    reset_seed: int = 42,
    max_steps: int = 40,
    gif_name: str = "policy_rollout.gif",
    frame_prefix: str = "rollout",
    fps: float = 1.5,
):
    """Roll out a deterministic CliffWalking policy and export an env-render GIF."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required to export CliffWalking policy GIFs.") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to export CliffWalking policy GIFs.") from exc

    env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state, _ = env.reset(seed=int(reset_seed))
    state = int(state)
    total_return = 0.0
    frames = []

    initial_frame = env.render()
    frames.append(Image.fromarray(np.asarray(initial_frame)).convert("P", palette=Image.ADAPTIVE))

    for step_idx in range(max_steps):
        action = int(greedy_actions[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = int(next_state)
        total_return += float(reward)

        frame = env.render()
        frame_path = output_dir / f"{frame_prefix}_{step_idx + 1:02d}.png"
        Image.fromarray(np.asarray(frame)).save(frame_path)
        frames.append(Image.open(frame_path).convert("P", palette=Image.ADAPTIVE))

        if terminated or truncated:
            break

    env.close()

    initial_path = output_dir / f"{frame_prefix}_00.png"
    Image.fromarray(np.asarray(initial_frame)).save(initial_path)

    gif_path = output_dir / gif_name
    duration_ms = int(round(1000 / fps)) if fps > 0 else 700
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    return {
        "gif_path": str(gif_path),
        "steps_executed": len(frames) - 1,
        "total_return": total_return,
    }

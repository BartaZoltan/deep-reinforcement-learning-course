"""Shared helpers for Session 5 (n-step Returns and Eligibility Traces)."""

from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.colors import ListedColormap, TwoSlopeNorm


N_ROWS = 4
N_COLS = 12
START_STATE = (N_ROWS - 1) * N_COLS
CLIFF_ACTION_SYMBOLS = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←",
}

S3_COLOR_S = "#A7D3F5"
S3_COLOR_F = "#FFFFFF"
S3_COLOR_H = "#F8B4B4"
S3_COLOR_G = "#B7E4C7"


def set_seed(seed: int = 42) -> np.random.Generator:
    """Set Python/NumPy seeds and return a default RNG."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def running_mean(values, window: int = 20) -> np.ndarray:
    """Simple moving average with a growing prefix."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()

    out = np.zeros_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = np.mean(values[start : i + 1])
    return out


def _cell_label(r: int, c: int, n_rows: int, n_cols: int) -> str | None:
    """Return a special label for start, cliff, and goal cells."""
    if (r, c) == (n_rows - 1, 0):
        return "S"
    if (r, c) == (n_rows - 1, n_cols - 1):
        return "G"
    if r == n_rows - 1 and 0 < c < n_cols - 1:
        return "H"
    return None


def _terrain_label(r: int, c: int, n_rows: int, n_cols: int) -> str:
    """Return terrain label for a CliffWalking cell."""
    special = _cell_label(r, c, n_rows, n_cols)
    if special is not None:
        return special
    return "F"


def _base_grid(n_rows: int, n_cols: int):
    """Create a base CliffWalking grid figure."""
    grid = np.ones((n_rows, n_cols), dtype=float)
    cmap = ListedColormap(["#f7f7f7"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    return fig, ax


def plot_cliffwalking_map(
    n_rows: int = 4,
    n_cols: int = 12,
    title: str = "CliffWalking layout",
):
    """Plot a colored CliffWalking grid with S/H/F/G terrain labels."""
    grid = np.zeros((n_rows, n_cols), dtype=int)
    grid[n_rows - 1, 1 : n_cols - 1] = 1
    grid[n_rows - 1, 0] = 2
    grid[n_rows - 1, n_cols - 1] = 3

    cmap = ListedColormap([S3_COLOR_F, S3_COLOR_H, S3_COLOR_S, S3_COLOR_G])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)
    ax.set_title(title)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(n_rows):
        for c in range(n_cols):
            label = _terrain_label(r, c, n_rows, n_cols)
            ax.text(c, r, label, ha="center", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_cliffwalking_values(
    values: np.ndarray,
    n_rows: int = 4,
    n_cols: int = 12,
    title: str = "CliffWalking state values",
):
    """Plot a CliffWalking value heatmap."""
    grid = np.asarray(values, dtype=float).reshape(n_rows, n_cols)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(grid, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for s in range(min(len(values), n_rows * n_cols)):
        r, c = divmod(s, n_cols)
        special = _cell_label(r, c, n_rows, n_cols)
        text = special if special is not None else f"{values[s]:.1f}"
        color = "white" if special is None else "black"
        ax.text(c, r, text, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="V(s)")
    plt.tight_layout()
    plt.show()


def plot_cliffwalking_value_difference(
    values: np.ndarray,
    reference: np.ndarray,
    n_rows: int = 4,
    n_cols: int = 12,
    title: str = "CliffWalking value difference",
):
    """Plot a signed CliffWalking value difference heatmap."""
    diff = np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)
    grid = diff.reshape(n_rows, n_cols)
    vmax = float(np.max(np.abs(grid)))
    vmax = max(vmax, 1e-8)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(grid, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_title(title)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for s in range(min(len(diff), n_rows * n_cols)):
        r, c = divmod(s, n_cols)
        special = _cell_label(r, c, n_rows, n_cols)
        text = special if special is not None else f"{diff[s]:+.1f}"
        ax.text(c, r, text, ha="center", va="center", fontsize=9, color="black", fontweight="bold")

    fig.colorbar(im, ax=ax, label="V(s) - reference")
    plt.tight_layout()
    plt.show()


def plot_cliffwalking_policy(
    actions: np.ndarray,
    n_rows: int = 4,
    n_cols: int = 12,
    title: str = "CliffWalking greedy policy",
):
    """Plot a CliffWalking policy grid."""
    fig, ax = _base_grid(n_rows, n_cols)
    ax.set_title(title)

    for s in range(min(len(actions), n_rows * n_cols)):
        r, c = divmod(s, n_cols)
        special = _cell_label(r, c, n_rows, n_cols)
        label = special if special is not None else CLIFF_ACTION_SYMBOLS.get(int(actions[s]), "?")
        ax.text(c, r, label, ha="center", va="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_epsilon_greedy_action_probs(
    epsilon: float,
    greedy_action: int = 1,
    n_actions: int = 4,
    title: str = "Epsilon-greedy action probabilities",
):
    """Plot epsilon-greedy action probabilities for one state."""
    probs = np.full(n_actions, float(epsilon) / float(n_actions), dtype=float)
    greedy_action = int(greedy_action)
    probs[greedy_action] += 1.0 - float(epsilon)

    labels = [f"{CLIFF_ACTION_SYMBOLS.get(a, str(a))} ({a})" for a in range(n_actions)]
    colors = ["#d1d5db"] * n_actions
    colors[greedy_action] = "#60a5fa"

    plt.figure(figsize=(6.2, 3.5))
    plt.bar(np.arange(n_actions), probs, color=colors, edgecolor="black")
    plt.xticks(np.arange(n_actions), labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Probability")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)

    for i, p in enumerate(probs):
        plt.text(i, p + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


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


def render_side_by_side_gifs(
    items: list[tuple[str, str | Path]],
    *,
    width: int = 360,
) -> HTML:
    """Return an HTML block that shows several GIFs next to each other."""
    blocks: list[str] = []
    for label, path in items:
        src = Path(path).as_posix()
        blocks.append(
            (
                '<div style="flex:0 0 auto;">'
                f"<p><strong>{label}</strong></p>"
                f"<img src=\"{src}\" width=\"{width}\">"
                "</div>"
            )
        )

    html = (
        '<div style="display:flex; gap:24px; flex-wrap:nowrap; overflow-x:auto; align-items:flex-start;">'
        + "".join(blocks)
        + "</div>"
    )
    return HTML(html)


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
    if exploring_starts and env.spec is not None and env.spec.id == "CliffWalking-v1":
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

    env = gym.make("CliffWalking-v1", render_mode="rgb_array")
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

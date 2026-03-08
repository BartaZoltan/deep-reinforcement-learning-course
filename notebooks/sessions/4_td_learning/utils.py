"""Shared helpers for Session 4 (Temporal-Difference Learning)."""

from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


CLIFF_ACTION_SYMBOLS = {
    0: "↑",  # UP
    1: "→",  # RIGHT
    2: "↓",  # DOWN
    3: "←",  # LEFT
}

# Match Session 3 visual palette.
S3_COLOR_S = "#A7D3F5"  # Start
S3_COLOR_F = "#FFFFFF"  # Free/Frozen
S3_COLOR_H = "#F8B4B4"  # Hole/Cliff
S3_COLOR_G = "#B7E4C7"  # Goal
AGENT_COLOR = "#93C5FD"  # Agent overlay in trajectory frames


def set_seed(seed: int = 42) -> np.random.Generator:
    """Set Python/NumPy seeds and return a default RNG."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def _render_cliffwalking_frame(
    state: int,
    n_rows: int,
    n_cols: int,
    title: str = "CliffWalking trajectory",
):
    """Render one CliffWalking state frame with Matplotlib."""
    start_state = (n_rows - 1) * n_cols
    goal_state = n_rows * n_cols - 1

    grid = np.zeros((n_rows, n_cols), dtype=int)
    grid[n_rows - 1, 1 : n_cols - 1] = 1  # cliff
    grid.flat[start_state] = 2
    grid.flat[goal_state] = 3
    grid.flat[int(state)] = 4

    cmap = ListedColormap(
        [
            S3_COLOR_F,  # normal/free
            S3_COLOR_H,  # cliff/hole
            S3_COLOR_S,  # start
            S3_COLOR_G,  # goal
            AGENT_COLOR,  # agent
        ]
    )

    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4)
    ax.set_title(title)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(n_rows):
        for c in range(n_cols):
            s = r * n_cols + c
            label = None
            if s == int(state):
                label = "A"
            else:
                label = _terrain_label(r, c, n_rows, n_cols)
            ax.text(c, r, label, ha="center", va="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_cliffwalking_map(
    n_rows: int = 4,
    n_cols: int = 12,
    title: str = "CliffWalking layout",
):
    """Plot a colored CliffWalking grid with S/H/F/G terrain labels."""
    grid = np.zeros((n_rows, n_cols), dtype=int)
    grid[n_rows - 1, 1 : n_cols - 1] = 1  # cliff/hole
    grid[n_rows - 1, 0] = 2  # start
    grid[n_rows - 1, n_cols - 1] = 3  # goal

    cmap = ListedColormap(
        [
            S3_COLOR_F,  # F
            S3_COLOR_H,  # H
            S3_COLOR_S,  # S
            S3_COLOR_G,  # G
        ]
    )

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
    """Plot a simple CliffWalking value heatmap."""
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


def plot_cliffwalking_policy(
    actions: np.ndarray,
    n_rows: int = 4,
    n_cols: int = 12,
    title: str = "CliffWalking greedy policy",
):
    """Plot a simple CliffWalking policy grid."""
    fig, ax = _base_grid(n_rows, n_cols)
    ax.set_title(title)

    for s in range(min(len(actions), n_rows * n_cols)):
        r, c = divmod(s, n_cols)
        special = _cell_label(r, c, n_rows, n_cols)
        label = special if special is not None else CLIFF_ACTION_SYMBOLS.get(int(actions[s]), "?")

        ax.text(c, r, label, ha="center", va="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_cliffwalking_cell_slip(
    slip_prob: float = 0.20,
    n_rows: int = 4,
    n_cols: int = 12,
    title: str = "Cell-wise slip probability",
):
    """Plot a CliffWalking grid with per-cell slip probability labels."""
    grid = np.zeros((n_rows, n_cols), dtype=int)
    grid[n_rows - 1, 1 : n_cols - 1] = 1  # cliff/hole
    grid[n_rows - 1, 0] = 2  # start
    grid[n_rows - 1, n_cols - 1] = 3  # goal

    cmap = ListedColormap(
        [
            S3_COLOR_F,  # F
            S3_COLOR_H,  # H
            S3_COLOR_S,  # S
            S3_COLOR_G,  # G
        ]
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)
    ax.set_title(title)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    p = float(slip_prob)
    for r in range(n_rows):
        for c in range(n_cols):
            terrain = _terrain_label(r, c, n_rows, n_cols)
            if terrain == "H":
                p_txt = "p=N/A"
            else:
                p_txt = f"p={p:.0%}"

            ax.text(c, r - 0.10, terrain, ha="center", va="center", fontsize=10.0, fontweight="bold")
            ax.text(c, r + 0.18, p_txt, ha="center", va="center", fontsize=7.8, fontweight="normal")

    plt.tight_layout()
    plt.show()


def plot_policy_comparison(
    actions_a: np.ndarray,
    actions_b: np.ndarray,
    title_a: str,
    title_b: str,
    suptitle: str,
    n_rows: int = 4,
    n_cols: int = 12,
):
    """Plot two CliffWalking greedy policies side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(suptitle)

    for ax, actions, title in zip(axes, [actions_a, actions_b], [title_a, title_b]):
        grid = np.ones((n_rows, n_cols), dtype=float)
        cmap = ListedColormap(["#f7f7f7"])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for s in range(min(len(actions), n_rows * n_cols)):
            r, c = divmod(s, n_cols)
            special = _cell_label(r, c, n_rows, n_cols)
            label = special if special is not None else CLIFF_ACTION_SYMBOLS.get(int(actions[s]), "?")
            ax.text(c, r, label, ha="center", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_return_curve(
    values,
    title: str = "Running mean episodic return",
    xlabel: str = "Episode index",
    ylabel: str = "Return",
):
    """Plot a single return curve."""
    x = np.arange(1, len(values) + 1)
    plt.figure(figsize=(6.5, 3.8))
    plt.plot(x, values, linewidth=2.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_curve_pair(
    x,
    y1,
    y2,
    label1: str,
    label2: str,
    title: str,
    xlabel: str,
    ylabel: str,
    show: bool = True,
):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(x, y1, marker="o", label=label1)
    ax.plot(x, y2, marker="o", label=label2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return ax



def plot_curves(
    x,
    curves,
    labels,
    title: str,
    xlabel: str,
    ylabel: str,
    marker: str | None = None,
):
    """Plot multiple curves on the same axes."""
    if len(curves) != len(labels):
        raise ValueError("`curves` and `labels` must have the same length.")

    plt.figure(figsize=(7.2, 4.0))
    for y, label in zip(curves, labels):
        plot_kwargs = {"label": label, "linewidth": 2.0}
        if marker is not None:
            plot_kwargs["marker"] = marker
        plt.plot(x, y, **plot_kwargs)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bar_pair(
    labels,
    vals_a,
    vals_b,
    name_a: str,
    name_b: str,
    title: str,
    ylabel: str,
):
    """Plot a simple side-by-side bar comparison."""
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(7.0, 3.8))
    plt.bar(x - width / 2, vals_a, width=width, label=name_a)
    plt.bar(x + width / 2, vals_b, width=width, label=name_b)
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_box_comparison(
    values_a,
    values_b,
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
):
    """Plot a two-method boxplot comparison."""
    plt.figure(figsize=(6.8, 3.8))
    bp = plt.boxplot(
        [values_a, values_b],
        labels=[label_a, label_b],
        patch_artist=True,
        widths=0.55,
    )

    colors = ["#60a5fa", "#f59e0b"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
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


def export_cliffwalking_manual_trajectory(
    actions,
    output_dir: str | Path,
    *,
    reset_seed: int = 0,
    gif_name: str = "cliffwalking_manual_trajectory.gif",
    frame_prefix: str = "frame",
    fps: float = 1.0,
    n_rows: int = 4,
    n_cols: int = 12,
):
    """Replay manual CliffWalking actions and save PNG frames plus a GIF.

    Parameters
    ----------
    actions : Sequence[int]
        Manual action sequence (0:UP, 1:RIGHT, 2:DOWN, 3:LEFT).
    output_dir : str | Path
        Directory for output PNG frames and GIF.
    reset_seed : int, default=0
        Environment reset seed.
    gif_name : str, default='cliffwalking_manual_trajectory.gif'
        GIF filename.
    frame_prefix : str, default='frame'
        Prefix for frame image filenames.
    fps : float, default=1.0
        GIF playback speed.
    n_rows : int, default=4
        Grid rows for visualization.
    n_cols : int, default=12
        Grid columns for visualization.

    Returns
    -------
    dict[str, object]
        Saved paths, rewards, and trajectory summary.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required to export CliffWalking trajectory assets.") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to save CliffWalking trajectory GIF assets.") from exc

    output_dir = ensure_dir(output_dir)
    env = gym.make("CliffWalking-v0")

    saved_frames: list[str] = []
    pil_frames = []
    rewards: list[float] = []
    states: list[int] = []
    actions_done: list[int] = []

    def _capture(step_idx: int, suffix: str) -> None:
        fig = _render_cliffwalking_frame(
            state=states[-1],
            n_rows=n_rows,
            n_cols=n_cols,
            title="CliffWalking manual trajectory",
        )
        frame_path = output_dir / f"{frame_prefix}_{step_idx:02d}_{suffix}.png"
        fig.savefig(frame_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved_frames.append(str(frame_path))
        pil_frames.append(Image.open(frame_path).copy())

    try:
        obs, _ = env.reset(seed=int(reset_seed))
        states.append(int(obs))
        _capture(0, f"s{int(obs)}")

        terminated = False
        truncated = False

        for step_idx, action in enumerate(actions, start=1):
            action = int(action)
            if action < 0 or action >= env.action_space.n:
                raise ValueError(
                    f"Invalid CliffWalking action {action}. Expected an integer in [0, {env.action_space.n - 1}]."
                )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            actions_done.append(action)
            states.append(int(next_obs))
            _capture(step_idx, f"a{action}_s{int(next_obs)}")

            if terminated or truncated:
                break

        if not pil_frames:
            raise RuntimeError("No frames were captured for the CliffWalking trajectory.")

        gif_path = output_dir / gif_name
        frame_duration_ms = max(int(round(1000.0 / max(float(fps), 1e-6))), 1)
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )

        return {
            "gif_path": str(gif_path),
            "frame_paths": saved_frames,
            "states": states,
            "actions": actions_done,
            "rewards": rewards,
            "steps_executed": len(rewards),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_state": int(states[-1]),
            "total_return": float(sum(rewards)),
        }
    finally:
        env.close()


def export_cliffwalking_random_trajectory_rgb(
    output_dir: str | Path,
    *,
    reset_seed: int = 0,
    max_steps: int = 30,
    gif_name: str = "cliffwalking_random_walk.gif",
    frame_prefix: str = "rand",
    fps: float = 2.0,
):
    """Generate one random CliffWalking trajectory using native RGB render.

    Parameters
    ----------
    output_dir : str | Path
        Directory for output PNG frames and GIF.
    reset_seed : int, default=0
        Environment reset seed.
    max_steps : int, default=30
        Maximum random-walk steps.
    gif_name : str, default='cliffwalking_random_walk.gif'
        GIF filename.
    frame_prefix : str, default='rand'
        Prefix for frame image filenames.
    fps : float, default=2.0
        GIF playback speed.

    Returns
    -------
    dict[str, object]
        Saved paths and trajectory summary.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required to export CliffWalking trajectory assets.") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to save CliffWalking trajectory GIF assets.") from exc

    output_dir = ensure_dir(output_dir)
    env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    rng = np.random.default_rng(int(reset_seed))

    saved_frames: list[str] = []
    pil_frames = []
    rewards: list[float] = []
    states: list[int] = []
    actions_done: list[int] = []

    def _capture(step_idx: int, suffix: str) -> None:
        frame = np.asarray(env.render(), dtype=np.uint8)
        img = Image.fromarray(frame)
        frame_path = output_dir / f"{frame_prefix}_{step_idx:02d}_{suffix}.png"
        img.save(frame_path)
        saved_frames.append(str(frame_path))
        pil_frames.append(img)

    try:
        obs, _ = env.reset(seed=int(reset_seed))
        states.append(int(obs))
        _capture(0, f"s{int(obs)}")

        terminated = False
        truncated = False

        for step_idx in range(1, int(max_steps) + 1):
            action = int(rng.integers(0, env.action_space.n))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            actions_done.append(action)
            rewards.append(float(reward))
            states.append(int(next_obs))
            _capture(step_idx, f"a{action}_s{int(next_obs)}")

            if terminated or truncated:
                break

        if not pil_frames:
            raise RuntimeError("No frames were captured for the CliffWalking trajectory.")

        gif_path = output_dir / gif_name
        frame_duration_ms = max(int(round(1000.0 / max(float(fps), 1e-6))), 1)
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )

        return {
            "gif_path": str(gif_path),
            "frame_paths": saved_frames,
            "states": states,
            "actions": actions_done,
            "rewards": rewards,
            "steps_executed": len(rewards),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_state": int(states[-1]),
            "total_return": float(sum(rewards)),
        }
    finally:
        env.close()


def export_cliffwalking_greedy_trajectory(
    greedy_actions: np.ndarray,
    output_dir: str | Path,
    *,
    reset_seed: int = 0,
    max_steps: int = 30,
    gif_name: str = "cliffwalking_greedy_policy.gif",
    frame_prefix: str = "greedy",
    fps: float = 2.0,
):
    """Replay a greedy CliffWalking policy and save PNG frames plus a GIF."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required to export CliffWalking trajectory assets.") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to save CliffWalking trajectory GIF assets.") from exc

    output_dir = ensure_dir(output_dir)
    env = gym.make("CliffWalking-v0", render_mode="rgb_array")

    saved_frames: list[str] = []
    pil_frames = []
    rewards: list[float] = []
    states: list[int] = []
    actions_done: list[int] = []

    def _capture(step_idx: int, suffix: str) -> None:
        frame = np.asarray(env.render(), dtype=np.uint8)
        img = Image.fromarray(frame)
        frame_path = output_dir / f"{frame_prefix}_{step_idx:02d}_{suffix}.png"
        img.save(frame_path)
        saved_frames.append(str(frame_path))
        pil_frames.append(img)

    try:
        obs, _ = env.reset(seed=int(reset_seed))
        states.append(int(obs))
        _capture(0, f"s{int(obs)}")

        terminated = False
        truncated = False

        for step_idx in range(1, int(max_steps) + 1):
            action = int(greedy_actions[int(states[-1])])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            actions_done.append(action)
            rewards.append(float(reward))
            states.append(int(next_obs))
            _capture(step_idx, f"a{action}_s{int(next_obs)}")

            if terminated or truncated:
                break

        if not pil_frames:
            raise RuntimeError("No frames were captured for the CliffWalking trajectory.")

        gif_path = output_dir / gif_name
        frame_duration_ms = max(int(round(1000.0 / max(float(fps), 1e-6))), 1)
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )

        return {
            "gif_path": str(gif_path),
            "frame_paths": saved_frames,
            "states": states,
            "actions": actions_done,
            "rewards": rewards,
            "steps_executed": len(rewards),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_state": int(states[-1]),
            "total_return": float(sum(rewards)),
        }
    finally:
        env.close()


def export_cliffwalking_policy_comparison_gif(
    greedy_actions_left: np.ndarray,
    greedy_actions_right: np.ndarray,
    output_dir: str | Path,
    *,
    label_left: str = "SARSA",
    label_right: str = "Q-learning",
    reset_seed: int = 0,
    max_steps: int = 30,
    gif_name: str = "cliffwalking_policy_comparison.gif",
    frame_prefix: str = "compare",
    fps: float = 2.0,
):
    """Replay two greedy CliffWalking policies side by side and save a GIF."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required to export CliffWalking trajectory assets.") from exc

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required to save CliffWalking trajectory GIF assets.") from exc

    output_dir = ensure_dir(output_dir)
    env_left = gym.make("CliffWalking-v0", render_mode="rgb_array")
    env_right = gym.make("CliffWalking-v0", render_mode="rgb_array")

    saved_frames: list[str] = []
    pil_frames = []
    rewards_left: list[float] = []
    rewards_right: list[float] = []
    states_left: list[int] = []
    states_right: list[int] = []

    def _compose_frame(step_idx: int, suffix: str) -> None:
        frame_left = Image.fromarray(np.asarray(env_left.render(), dtype=np.uint8))
        frame_right = Image.fromarray(np.asarray(env_right.render(), dtype=np.uint8))

        title_h = 28
        pad = 8
        width = frame_left.width + frame_right.width + 3 * pad
        height = max(frame_left.height, frame_right.height) + title_h + 2 * pad
        canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
        draw = ImageDraw.Draw(canvas)

        canvas.paste(frame_left, (pad, title_h + pad))
        canvas.paste(frame_right, (frame_left.width + 2 * pad, title_h + pad))
        draw.text((pad, 6), label_left, fill=(15, 23, 42))
        draw.text((frame_left.width + 2 * pad, 6), label_right, fill=(15, 23, 42))

        frame_path = output_dir / f"{frame_prefix}_{step_idx:02d}_{suffix}.png"
        canvas.save(frame_path)
        saved_frames.append(str(frame_path))
        pil_frames.append(canvas)

    try:
        obs_left, _ = env_left.reset(seed=int(reset_seed))
        obs_right, _ = env_right.reset(seed=int(reset_seed))
        states_left.append(int(obs_left))
        states_right.append(int(obs_right))
        _compose_frame(0, f"s{int(obs_left)}_s{int(obs_right)}")

        terminated_left = False
        truncated_left = False
        terminated_right = False
        truncated_right = False

        for step_idx in range(1, int(max_steps) + 1):
            if not (terminated_left or truncated_left):
                action_left = int(greedy_actions_left[int(states_left[-1])])
                next_obs_left, reward_left, terminated_left, truncated_left, _ = env_left.step(action_left)
                rewards_left.append(float(reward_left))
                states_left.append(int(next_obs_left))

            if not (terminated_right or truncated_right):
                action_right = int(greedy_actions_right[int(states_right[-1])])
                next_obs_right, reward_right, terminated_right, truncated_right, _ = env_right.step(action_right)
                rewards_right.append(float(reward_right))
                states_right.append(int(next_obs_right))

            _compose_frame(step_idx, f"s{states_left[-1]}_s{states_right[-1]}")

            if (terminated_left or truncated_left) and (terminated_right or truncated_right):
                break

        if not pil_frames:
            raise RuntimeError("No frames were captured for the CliffWalking comparison trajectory.")

        gif_path = output_dir / gif_name
        frame_duration_ms = max(int(round(1000.0 / max(float(fps), 1e-6))), 1)
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )

        return {
            "gif_path": str(gif_path),
            "frame_paths": saved_frames,
            "states_left": states_left,
            "states_right": states_right,
            "rewards_left": rewards_left,
            "rewards_right": rewards_right,
            "steps_executed": max(len(rewards_left), len(rewards_right)),
            "left_return": float(sum(rewards_left)),
            "right_return": float(sum(rewards_right)),
        }
    finally:
        env_left.close()
        env_right.close()


def running_mean(values, window: int = 20) -> np.ndarray:
    """Compute a simple running mean with a fixed trailing window."""
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = values[start : i + 1].mean()
    return out

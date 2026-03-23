"""Shared helpers for Session 6 (Function Approximation)."""

from __future__ import annotations

import base64
import math
import random
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


ACTION_NAMES = {
    0: "Push left",
    1: "No push",
    2: "Push right",
}


def _show_and_close(fig=None):
    """Render a Matplotlib figure without leaking Agg backend warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
            category=UserWarning,
        )
        plt.show()
    if fig is not None:
        plt.close(fig)


def set_seed(seed: int = 42) -> np.random.Generator:
    """Set Python/NumPy seeds and return a default RNG."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def resolve_asset_root(session_dir: str | Path) -> Path:
    """Resolve the asset output directory both locally and inside Colab-like layouts."""
    session_dir = Path(session_dir)
    asset_root = session_dir / "assets" / "cell_outputs"
    if not session_dir.exists():
        asset_root = Path("assets/cell_outputs")
    asset_root.mkdir(parents=True, exist_ok=True)
    return asset_root


def _resolve_cell_outputs_dir(path: str | Path) -> Path:
    """Flatten nested output requests back to the shared cell_outputs directory."""
    path = Path(path)
    for candidate in (path, *path.parents):
        if candidate.name == "cell_outputs":
            return candidate
    return path


def _sanitize_gif_name(name: str) -> str:
    """Turn a requested GIF filename into a safe single-file name."""
    raw = str(name).replace("\\", "_").replace("/", "_")
    stem = raw[:-4] if raw.lower().endswith(".gif") else raw
    stem = stem or "rollout"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "rollout"
    return f"{safe_stem}.gif"


def moving_average(values, window: int = 10) -> np.ndarray:
    """Simple moving average with a growing prefix."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()

    out = np.zeros_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = np.mean(values[start : i + 1])
    return out


def plot_mountaincar_state_space(env, n_samples: int = 400, seed: int = 42):
    """Plot sampled states from the MountainCar observation space."""
    rng = np.random.default_rng(seed)
    low = env.observation_space.low
    high = env.observation_space.high
    samples = rng.uniform(low=low, high=high, size=(n_samples, 2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(samples[:, 0], samples[:, 1], s=12, alpha=0.35, color="#4C78A8")
    ax.axvline(
        env.unwrapped.goal_position,
        color="#E45756",
        linestyle="--",
        linewidth=2,
        label="Goal position",
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("MountainCar state space")
    ax.legend(loc="upper left")
    plt.tight_layout()
    _show_and_close(fig)


def plot_mountaincar_observation_space(
    env,
    example_states: np.ndarray | None = None,
):
    """Plot the MountainCar observation space as a bounded position-velocity box."""
    low = np.asarray(env.observation_space.low, dtype=float)
    high = np.asarray(env.observation_space.high, dtype=float)

    if example_states is None:
        example_states = np.array(
            [
                [-1.05, 0.00],
                [-0.55, 0.03],
                [0.10, -0.02],
            ],
            dtype=float,
        )
    else:
        example_states = np.asarray(example_states, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.fill_between(
        [low[0], high[0]],
        low[1],
        high[1],
        color="#DCEBFA",
        alpha=0.85,
        label="Valid state region",
    )

    rect = plt.Rectangle(
        (low[0], low[1]),
        high[0] - low[0],
        high[1] - low[1],
        fill=False,
        edgecolor="#2C6FB7",
        linewidth=2.5,
    )
    ax.add_patch(rect)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.5)

    ax.scatter(
        example_states[:, 0],
        example_states[:, 1],
        color="#D1495B",
        s=70,
        zorder=3,
        label="Example states",
    )

    for idx, (x, y) in enumerate(example_states):
        ax.text(x + 0.02, y + 0.003, f"$s_{idx}$", fontsize=11)

    ax.text(low[0], low[1] - 0.006, f"low = ({low[0]:.2f}, {low[1]:.2f})", fontsize=10, ha="left")
    ax.text(high[0], high[1] + 0.003, f"high = ({high[0]:.2f}, {high[1]:.2f})", fontsize=10, ha="right")

    ax.set_xlim(low[0] - 0.08, high[0] + 0.08)
    ax.set_ylim(low[1] - 0.015, high[1] + 0.015)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("MountainCar observation space")
    ax.legend(loc="upper left")
    plt.tight_layout()
    _show_and_close(fig)


def plot_many_curves(curves: dict[str, np.ndarray], title: str, ylabel: str, window: int = 10):
    """Plot raw and smoothed learning curves."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, values in curves.items():
        values = np.asarray(values, dtype=float)
        raw_line, = ax.plot(values, alpha=0.20)
        ax.plot(
            moving_average(values, window=window),
            linewidth=2,
            color=raw_line.get_color(),
            label=f"{label} (moving average)",
        )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    _show_and_close(fig)


def plot_series_vs_x(
    x_values: np.ndarray,
    series: dict[str, np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
):
    """Plot one or more series against an explicit shared x-axis."""
    x_values = np.asarray(x_values, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, values in series.items():
        values = np.asarray(values, dtype=float)
        ax.plot(x_values, values, marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    _show_and_close(fig)


def plot_tile_coder_activation(tile_coder, state: np.ndarray):
    """Visualize the active tile in each tiling for a 2D state."""
    coords = tile_coder.tile_coordinates(state)
    n_cols = math.ceil(tile_coder.num_tilings / 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.0 * n_cols, 5.5))
    axes = np.asarray(axes).reshape(-1)

    for tiling in range(tile_coder.num_tilings):
        ax = axes[tiling]
        grid = np.zeros((tile_coder.tiles[1], tile_coder.tiles[0]), dtype=int)
        pos_idx, vel_idx = coords[tiling]
        grid[vel_idx, pos_idx] = 1
        ax.imshow(
            grid,
            origin="lower",
            cmap=ListedColormap(["#F7F7F7", "#4C78A8"]),
            aspect="auto",
        )
        ax.set_title(f"Tiling {tiling}")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[tile_coder.num_tilings :]:
        ax.axis("off")

    fig.suptitle(f"Active tiles for state = ({state[0]:.3f}, {state[1]:.3f})", y=1.02)
    plt.tight_layout()
    _show_and_close(fig)


def plot_tile_coder_activation_detailed(tile_coder, state: np.ndarray, action: int = 0):
    """Visualize active tiles per tiling together with their feature indices."""
    state = np.asarray(state, dtype=float)
    coords = tile_coder.tile_coordinates(state)
    active_features = tile_coder.encode(state, action)

    low = tile_coder.low
    high = tile_coder.high
    width = tile_coder.state_range / tile_coder.tiles

    n_cols = math.ceil(tile_coder.num_tilings / 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 7.4))
    axes = np.asarray(axes).reshape(-1)

    for tiling in range(tile_coder.num_tilings):
        ax = axes[tiling]
        shift = tile_coder.offsets[tiling] * tile_coder.state_range

        ax.add_patch(
            Rectangle(
                (low[0], low[1]),
                high[0] - low[0],
                high[1] - low[1],
                facecolor="#EAF3FB",
                edgecolor="#2C6FB7",
                linewidth=2,
                zorder=0,
            )
        )

        for j in range(tile_coder.tiles[0] + 1):
            x = low[0] - shift[0] + j * width[0]
            ax.plot([x, x], [low[1], high[1]], color="gray", linewidth=0.9, alpha=0.7, zorder=1)

        for j in range(tile_coder.tiles[1] + 1):
            y = low[1] - shift[1] + j * width[1]
            ax.plot([low[0], high[0]], [y, y], color="gray", linewidth=0.9, alpha=0.7, zorder=1)

        pos_idx, vel_idx = coords[tiling]
        active_lower_left = low - shift + np.array([pos_idx * width[0], vel_idx * width[1]])

        ax.add_patch(
            Rectangle(
                active_lower_left,
                width[0],
                width[1],
                facecolor="#D1495B",
                edgecolor="#8C1C13",
                alpha=0.55,
                linewidth=2,
                zorder=2,
            )
        )

        ax.scatter(state[0], state[1], color="black", s=35, zorder=3)
        ax.text(state[0] + 0.02, state[1] + 0.003, "$s$", fontsize=11)

        ax.text(
            0.03,
            0.93,
            f"active feature: {active_features[tiling]}",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
        )

        ax.set_xlim(low[0] - 0.03, high[0] + 0.03)
        ax.set_ylim(low[1] - 0.01, high[1] + 0.01)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title(f"Tiling {tiling}")

    for ax in axes[tile_coder.num_tilings :]:
        ax.axis("off")

    fig.suptitle(
        f"Tile coding: one active tile per tiling for state $s$ and action {action}",
        y=1.02,
        fontsize=14,
    )
    plt.tight_layout()
    _show_and_close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.add_patch(
        Rectangle(
            (low[0], low[1]),
            high[0] - low[0],
            high[1] - low[1],
            facecolor="#EAF3FB",
            edgecolor="#2C6FB7",
            linewidth=2,
        )
    )

    colors = plt.cm.tab10(np.linspace(0, 1, tile_coder.num_tilings))

    for tiling in range(tile_coder.num_tilings):
        shift = tile_coder.offsets[tiling] * tile_coder.state_range
        pos_idx, vel_idx = coords[tiling]
        active_lower_left = low - shift + np.array([pos_idx * width[0], vel_idx * width[1]])

        ax.add_patch(
            Rectangle(
                active_lower_left,
                width[0],
                width[1],
                facecolor=colors[tiling],
                edgecolor=colors[tiling],
                alpha=0.20,
                linewidth=2,
            )
        )

    ax.scatter(state[0], state[1], color="black", s=45, zorder=3)
    ax.text(state[0] + 0.02, state[1] + 0.003, "$s$", fontsize=11)

    ax.set_xlim(low[0] - 0.03, high[0] + 0.03)
    ax.set_ylim(low[1] - 0.01, high[1] + 0.01)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Overlapping active tiles for state $s$ and action {action}")
    plt.tight_layout()
    _show_and_close(fig)


def plot_cost_to_go_surface(
    cost_grid: np.ndarray,
    position_values: np.ndarray,
    velocity_values: np.ndarray,
    title: str,
):
    """Plot the MountainCar cost-to-go surface."""
    X, Y = np.meshgrid(position_values, velocity_values)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, cost_grid, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Cost-to-go")
    ax.set_title(title)
    plt.tight_layout()
    _show_and_close(fig)


def plot_greedy_action_map(
    action_grid: np.ndarray,
    position_values: np.ndarray,
    velocity_values: np.ndarray,
    title: str,
):
    """Plot the greedy action over the MountainCar state space."""
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = ListedColormap(["#4C78A8", "#B9B9B9", "#F58518"])
    im = ax.imshow(
        action_grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[position_values[0], position_values[-1], velocity_values[0], velocity_values[-1]],
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([ACTION_NAMES[0], ACTION_NAMES[1], ACTION_NAMES[2]])
    plt.tight_layout()
    _show_and_close(fig)


def run_greedy_episode(
    env,
    weights: np.ndarray,
    tile_coder,
    q_values_fn,
    max_steps: int,
    seed: int = 42,
):
    """Run one greedy episode under the current approximate action values."""
    state, _ = env.reset(seed=int(seed))
    state = np.asarray(state, dtype=float)
    states = [state.copy()]
    actions = []
    rewards = []

    for _ in range(max_steps):
        action = int(np.argmax(q_values_fn(weights, tile_coder, state)))
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.asarray(next_state, dtype=float)

        actions.append(action)
        rewards.append(float(reward))
        states.append(next_state.copy())
        state = next_state

        if terminated or truncated:
            break

    return {
        "states": np.asarray(states, dtype=float),
        "actions": np.asarray(actions, dtype=int),
        "rewards": np.asarray(rewards, dtype=float),
        "total_return": float(np.sum(rewards)),
        "steps": int(len(actions)),
        "terminated": bool(terminated) if len(actions) > 0 else False,
        "truncated": bool(truncated) if len(actions) > 0 else False,
        "seed": int(seed),
    }


def select_representative_greedy_rollout(
    env,
    weights: np.ndarray,
    tile_coder,
    q_values_fn,
    *,
    objective: str = "max_return",
    n_candidates: int = 12,
    max_steps: int = 200,
    seed: int = 42,
):
    """Select a representative greedy rollout by searching over several seeds."""
    if objective not in {"max_return", "min_steps_success", "max_steps"}:
        raise ValueError(f"Unsupported objective: {objective}")

    rollouts = [
        run_greedy_episode(
            env,
            weights,
            tile_coder,
            q_values_fn=q_values_fn,
            max_steps=max_steps,
            seed=seed + offset,
        )
        for offset in range(n_candidates)
    ]

    if objective == "max_return":
        return max(rollouts, key=lambda r: (r["total_return"], -r["steps"]))

    if objective == "max_steps":
        return max(rollouts, key=lambda r: (r["steps"], r["total_return"]))

    successful = [r for r in rollouts if r["terminated"]]
    if successful:
        return min(successful, key=lambda r: (r["steps"], -r["total_return"]))
    return max(rollouts, key=lambda r: (r["total_return"], -r["steps"]))


def evaluate_greedy_policy(
    env,
    weights: np.ndarray,
    tile_coder,
    q_values_fn,
    n_episodes: int = 20,
    max_steps: int = 200,
    seed: int = 42,
):
    """Evaluate the greedy policy induced by approximate action values."""
    returns = []
    lengths = []
    for episode_idx in range(n_episodes):
        rollout = run_greedy_episode(
            env,
            weights,
            tile_coder,
            q_values_fn=q_values_fn,
            max_steps=max_steps,
            seed=seed + episode_idx,
        )
        returns.append(rollout["total_return"])
        lengths.append(rollout["steps"])
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
    }


def plot_mountaincar_rollout(rollout: dict, goal_position: float):
    """Plot a greedy MountainCar rollout as position/velocity/action over time."""
    states = rollout["states"]
    actions = rollout["actions"]
    steps = np.arange(len(actions))

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(np.arange(len(states)), states[:, 0], color="#4C78A8", linewidth=2)
    axes[0].axhline(goal_position, color="#E45756", linestyle="--", linewidth=2, label="Goal position")
    axes[0].set_ylabel("Position")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Greedy MountainCar rollout diagnostics")

    axes[1].plot(np.arange(len(states)), states[:, 1], color="#54A24B", linewidth=2)
    axes[1].set_ylabel("Velocity")

    axes[2].step(steps, actions, where="post", color="#F58518", linewidth=2)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels([ACTION_NAMES[0], ACTION_NAMES[1], ACTION_NAMES[2]])
    axes[2].set_ylabel("Action")
    axes[2].set_xlabel("Step")

    reward_text = f"Return = {rollout['total_return']:.1f}, steps = {rollout['steps']}"
    axes[2].text(0.01, -0.45, reward_text, transform=axes[2].transAxes)
    plt.tight_layout()
    _show_and_close(fig)


def export_greedy_rollout_gif(
    env_id: str,
    weights: np.ndarray,
    tile_coder,
    q_values_fn,
    output_dir: str | Path,
    *,
    gif_name: str = "rollout.gif",
    max_steps: int = 200,
    seed: int = 42,
    fps: float = 20.0,
    objective: str = "max_return",
    n_candidates: int = 1,
):
    """Export an env-rendered GIF of one greedy rollout."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required to export rollout GIFs.") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to export rollout GIFs.") from exc

    env = gym.make(env_id, render_mode="rgb_array")
    output_dir = _resolve_cell_outputs_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_name = _sanitize_gif_name(gif_name)

    rollout = select_representative_greedy_rollout(
        env,
        weights,
        tile_coder,
        q_values_fn=q_values_fn,
        objective=objective,
        n_candidates=n_candidates,
        max_steps=max_steps,
        seed=seed,
    )

    state, _ = env.reset(seed=int(rollout["seed"]))
    state = np.asarray(state, dtype=float)
    total_return = 0.0
    steps = 0
    frames = []

    first_frame = env.render()
    frames.append(Image.fromarray(np.asarray(first_frame)).convert("P", palette=Image.ADAPTIVE))

    for step_idx in range(max_steps):
        action = int(np.argmax(q_values_fn(weights, tile_coder, state)))
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = np.asarray(next_state, dtype=float)
        total_return += float(reward)
        steps = step_idx + 1

        frame = env.render()
        frames.append(Image.fromarray(np.asarray(frame)).convert("P", palette=Image.ADAPTIVE))

        if terminated or truncated:
            break

    gif_path = output_dir / gif_name
    duration_ms = max(1, int(1000 / fps))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    env.close()
    return {
        "gif_path": str(gif_path),
        "total_return": total_return,
        "steps_executed": steps,
        "seed": int(rollout["seed"]),
        "terminated": bool(rollout["terminated"]),
        "truncated": bool(rollout["truncated"]),
    }


def render_side_by_side_gifs(items: list[dict[str, str]], width: int = 360):
    """Return HTML that renders one or more GIFs side by side."""
    blocks = []
    for item in items:
        gif_bytes = Path(item["gif_path"]).read_bytes()
        encoded = base64.b64encode(gif_bytes).decode("ascii")
        title = item.get("title", "Rollout")
        blocks.append(
            f"<div><p><strong>{title}</strong></p><img src='data:image/gif;base64,{encoded}' width='{width}'></div>"
        )

    html = (
        "<div style='display:flex; gap:24px; flex-wrap:nowrap; overflow-x:auto; align-items:flex-start;'>"
        + "".join(blocks)
        + "</div>"
    )
    return HTML(html)


def compare_sarsa_vs_q_learning(
    env_id: str,
    *,
    title_prefix: str,
    tile_coder_cls,
    semi_gradient_sarsa_fn,
    semi_gradient_q_learning_fn,
    q_values_fn,
    asset_root: str | Path,
    seed: int = 42,
    num_tilings: int,
    tiles: tuple[int, ...],
    alpha: float,
    epsilon: float,
    n_episodes: int,
    max_steps: int,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
    gif_objective: str = "max_return",
    gif_candidates: int = 16,
    gif_steps: int | None = None,
    gif_fps: float = 20.0,
    asset_name: str | None = None,
):
    """Run and visualize a SARSA vs Q-learning comparison on one environment."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required for environment comparisons.") from exc

    try:
        env_cmp = gym.make(env_id)
    except Exception as exc:
        print(f"Skipping {env_id}: {exc}")
        return None

    if low is None or high is None:
        low = np.asarray(env_cmp.observation_space.low, dtype=float)
        high = np.asarray(env_cmp.observation_space.high, dtype=float)
    else:
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)

    coder_sarsa = tile_coder_cls(
        low=low,
        high=high,
        num_tilings=num_tilings,
        tiles=tiles,
        num_actions=env_cmp.action_space.n,
    )
    coder_q = tile_coder_cls(
        low=low,
        high=high,
        num_tilings=num_tilings,
        tiles=tiles,
        num_actions=env_cmp.action_space.n,
    )

    sarsa_run = semi_gradient_sarsa_fn(
        env_cmp,
        coder_sarsa,
        n_episodes=n_episodes,
        alpha=alpha,
        gamma=1.0,
        epsilon=epsilon,
        max_steps=max_steps,
        seed=seed,
    )
    q_run = semi_gradient_q_learning_fn(
        env_cmp,
        coder_q,
        n_episodes=n_episodes,
        alpha=alpha,
        gamma=1.0,
        epsilon=epsilon,
        max_steps=max_steps,
        seed=seed,
    )

    sarsa_eval = evaluate_greedy_policy(
        env_cmp,
        sarsa_run["weights"],
        coder_sarsa,
        q_values_fn=q_values_fn,
        n_episodes=20,
        max_steps=max_steps,
        seed=seed,
    )
    q_eval = evaluate_greedy_policy(
        env_cmp,
        q_run["weights"],
        coder_q,
        q_values_fn=q_values_fn,
        n_episodes=20,
        max_steps=max_steps,
        seed=seed,
    )

    print(f"Greedy policy evaluation for {title_prefix}:")
    print("  SARSA     =", sarsa_eval)
    print("  Q-learning=", q_eval)

    plot_many_curves(
        {"Semi-gradient SARSA": sarsa_run["returns"], "Semi-gradient Q-learning": q_run["returns"]},
        title=f"{title_prefix}: return comparison",
        ylabel="Episode return",
        window=10,
    )
    plot_many_curves(
        {"Semi-gradient SARSA": sarsa_run["lengths"], "Semi-gradient Q-learning": q_run["lengths"]},
        title=f"{title_prefix}: episode-length comparison",
        ylabel="Steps per episode",
        window=10,
    )

    gif_steps = gif_steps or max_steps
    asset_name = asset_name or env_id.lower().replace("-", "_")
    asset_root = Path(asset_root)

    try:
        sarsa_gif = export_greedy_rollout_gif(
            env_id,
            sarsa_run["weights"],
            coder_sarsa,
            q_values_fn=q_values_fn,
            output_dir=asset_root,
            gif_name=f"{asset_name}_sarsa.gif",
            max_steps=gif_steps,
            seed=seed,
            fps=gif_fps,
            objective=gif_objective,
            n_candidates=gif_candidates,
        )
        q_gif = export_greedy_rollout_gif(
            env_id,
            q_run["weights"],
            coder_q,
            q_values_fn=q_values_fn,
            output_dir=asset_root,
            gif_name=f"{asset_name}_q_learning.gif",
            max_steps=gif_steps,
            seed=seed,
            fps=gif_fps,
            objective=gif_objective,
            n_candidates=gif_candidates,
        )
        print("SARSA rollout:", sarsa_gif)
        print("Q-learning rollout:", q_gif)
        display(
            render_side_by_side_gifs(
                [
                    {"title": "Semi-gradient SARSA", "gif_path": sarsa_gif["gif_path"]},
                    {"title": "Semi-gradient Q-learning", "gif_path": q_gif["gif_path"]},
                ],
                width=400,
            )
        )
    except Exception as exc:
        print(f"GIF export skipped for {title_prefix}: {exc}")
        sarsa_gif = None
        q_gif = None

    env_cmp.close()
    return {
        "sarsa": sarsa_run,
        "q_learning": q_run,
        "coder_sarsa": coder_sarsa,
        "coder_q": coder_q,
        "sarsa_eval": sarsa_eval,
        "q_eval": q_eval,
        "sarsa_gif": sarsa_gif,
        "q_gif": q_gif,
    }


def cartpole_feature_bounds():
    """Return clipped feature bounds for a simple linear CartPole representation."""
    low = np.array([-2.4, -3.0, -0.2095, -3.5], dtype=float)
    high = np.array([2.4, 3.0, 0.2095, 3.5], dtype=float)
    return low, high


def lunarlander_feature_bounds():
    """Return clipped feature bounds for a simple linear LunarLander representation."""
    low = np.array([-1.5, -0.5, -2.0, -2.0, -np.pi, -4.0, 0.0, 0.0], dtype=float)
    high = np.array([1.5, 1.5, 2.0, 2.0, np.pi, 4.0, 1.0, 1.0], dtype=float)
    return low, high

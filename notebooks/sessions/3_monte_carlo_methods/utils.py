"""Shared helpers for Session 3 (Monte Carlo Methods)."""

from __future__ import annotations

from pathlib import Path
import random
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ACTION_SYMBOLS = {
    0: "←",  # LEFT in FrozenLake
    1: "↓",  # DOWN
    2: "→",  # RIGHT
    3: "↑",  # UP
}


REPO_RAW_BASE = (
    "https://raw.githubusercontent.com/BartaZoltan/deep-reinforcement-learning-course/main"
)


def set_seed(seed: int = 42) -> np.random.Generator:
    """Set Python/NumPy seeds and return a default RNG."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_figure(fig, save_path: str | Path | None) -> None:
    """Save a Matplotlib figure when a path is provided."""
    if save_path is None:
        return
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")


def download_if_missing(local_path: str | Path, raw_repo_path: str) -> Path:
    """Download a file from GitHub raw URL only if it does not exist locally."""
    target = Path(local_path)
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{REPO_RAW_BASE}/{raw_repo_path.lstrip('/')}"
    urllib.request.urlretrieve(url, target)
    return target


def frozenlake_desc_to_grid(desc) -> np.ndarray:
    """Convert Gymnasium FrozenLake desc to a 2D char array."""
    arr = np.asarray(desc)
    if arr.dtype.kind in {"S", "O"}:  # bytes in classic gym envs
        arr = np.vectorize(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x))(arr)
    return arr.astype("<U1")


def state_to_rc(state: int, ncol: int) -> tuple[int, int]:
    return divmod(int(state), int(ncol))


def values_to_grid(V: np.ndarray, nrow: int, ncol: int) -> np.ndarray:
    grid = np.zeros((nrow, ncol), dtype=float)
    for s in range(nrow * ncol):
        r, c = state_to_rc(s, ncol)
        grid[r, c] = float(V[s])
    return grid


def _draw_cell_grid(ax, H: int, W: int) -> None:
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def plot_frozenlake_map(desc, title: str = "FrozenLake map", ax=None):
    grid = frozenlake_desc_to_grid(desc)
    H, W = grid.shape

    code_map = {"S": 0, "F": 1, "H": 2, "G": 3}
    arr = np.vectorize(lambda ch: code_map.get(ch, 1))(grid)
    cmap = ListedColormap(["#A7D3F5", "#FFFFFF", "#F8B4B4", "#B7E4C7"])

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.imshow(arr, cmap=cmap, vmin=0, vmax=3)
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    _draw_cell_grid(ax, H, W)

    for r in range(H):
        for c in range(W):
            ax.text(c, r, grid[r, c], ha="center", va="center", color="black", fontweight="bold")


def plot_value_grid(V: np.ndarray, desc, title: str = "State-value function", ax=None):
    grid_chars = frozenlake_desc_to_grid(desc)
    H, W = grid_chars.shape
    grid_vals = values_to_grid(V, H, W)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    im = ax.imshow(grid_vals, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    _draw_cell_grid(ax, H, W)

    vmin = float(np.min(grid_vals))
    vmax = float(np.max(grid_vals))
    den = (vmax - vmin) if vmax > vmin else 1.0

    for r in range(H):
        for c in range(W):
            ch = grid_chars[r, c]
            val = grid_vals[r, c]
            norm = (val - vmin) / den
            color = "white" if norm < 0.55 else "black"
            if ch in {"H", "G", "S"}:
                ax.text(c, r - 0.15, ch, ha="center", va="center", color="black", fontweight="bold", fontsize=9)
                ax.text(c, r + 0.18, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
            else:
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    return im


def plot_q_grid(Q: np.ndarray, desc, title: str = "Action-value function", ax=None):
    grid_chars = frozenlake_desc_to_grid(desc)
    H, W = grid_chars.shape
    nS = H * W

    # Use the mean action value as a simple background summary.
    bg_vals = values_to_grid(np.mean(Q, axis=1), H, W)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2))

    im = ax.imshow(bg_vals, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    _draw_cell_grid(ax, H, W)

    vmin = float(np.min(bg_vals))
    vmax = float(np.max(bg_vals))
    den = (vmax - vmin) if vmax > vmin else 1.0

    for s in range(nS):
        r, c = state_to_rc(s, W)
        ch = grid_chars[r, c]
        norm = (bg_vals[r, c] - vmin) / den
        color = "white" if norm < 0.55 else "black"

        if ch in {"H", "G", "S"}:
            ax.text(
                c,
                r,
                ch,
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=11,
            )
            continue

        q = Q[s]
        ax.text(c, r - 0.28, f"U:{q[3]:.2f}", ha="center", va="center", color=color, fontsize=6.5)
        ax.text(c - 0.28, r, f"L:{q[0]:.2f}", ha="center", va="center", color=color, fontsize=6.5)
        ax.text(c + 0.28, r, f"R:{q[2]:.2f}", ha="center", va="center", color=color, fontsize=6.5)
        ax.text(c, r + 0.28, f"D:{q[1]:.2f}", ha="center", va="center", color=color, fontsize=6.5)

    return im


def plot_value_comparison(
    V_a: np.ndarray,
    V_b: np.ndarray,
    desc,
    title_a: str,
    title_b: str,
    suptitle: str,
    save_path: str | Path | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if V_a.ndim == 2 and V_b.ndim == 2:
        im1 = plot_q_grid(V_a, desc, title=title_a, ax=axes[0])
        im2 = plot_q_grid(V_b, desc, title=title_b, ax=axes[1])
        color_label = "mean Q(s, a)"
    else:
        im1 = plot_value_grid(V_a, desc, title=title_a, ax=axes[0])
        im2 = plot_value_grid(V_b, desc, title=title_b, ax=axes[1])
        color_label = "V(s)"
    fig.suptitle(suptitle)
    fig.subplots_adjust(right=0.88, top=0.86, wspace=0.25)
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.62])
    fig.colorbar(im2, cax=cax, label=color_label)
    _save_figure(fig, save_path)
    plt.show()


def plot_policy_grid(actions: np.ndarray, desc, title: str = "Greedy policy", ax=None):
    grid_chars = frozenlake_desc_to_grid(desc)
    H, W = grid_chars.shape

    bg = np.ones((H, W), dtype=float)
    cmap = ListedColormap(["#f7f7f7"])
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.imshow(bg, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    _draw_cell_grid(ax, H, W)

    for s in range(H * W):
        r, c = state_to_rc(s, W)
        ch = grid_chars[r, c]
        if ch in {"S", "G", "H"}:
            txt = ch
        else:
            txt = ACTION_SYMBOLS.get(int(actions[s]), "?")
        ax.text(c, r, txt, ha="center", va="center", color="black", fontsize=11, fontweight="bold")


def plot_policy_comparison(
    actions_a: np.ndarray,
    actions_b: np.ndarray,
    desc,
    title_a: str,
    title_b: str,
    suptitle: str,
    save_path: str | Path | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_policy_grid(actions_a, desc, title=title_a, ax=axes[0])
    plot_policy_grid(actions_b, desc, title=title_b, ax=axes[1])
    fig.suptitle(suptitle)
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def plot_bar_pair(
    labels,
    vals_a,
    vals_b,
    name_a: str,
    name_b: str,
    title: str,
    ylabel: str,
    save_path: str | Path | None = None,
):
    x = np.arange(len(labels))
    w = 0.36
    fig = plt.figure(figsize=(7.0, 3.8))
    plt.bar(x - w / 2, vals_a, width=w, label=name_a)
    plt.bar(x + w / 2, vals_b, width=w, label=name_b)
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _save_figure(fig, save_path)
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
    save_path: str | Path | None = None,
):
    fig = plt.figure(figsize=(6.5, 3.8))
    plt.plot(x, y1, marker="o", label=label1)
    plt.plot(x, y2, marker="o", label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def export_frozenlake_manual_trajectory(
    actions,
    output_dir: str | Path,
    *,
    map_name: str = "4x4",
    desc=None,
    is_slippery: bool = False,
    reset_seed: int = 0,
    gif_name: str = "frozenlake_manual_trajectory.gif",
    frame_prefix: str = "frame",
    fps: float = 1.0,
):
    """Replay manual actions on FrozenLake and save each frame plus a GIF."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required to export FrozenLake trajectory assets.") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to save FrozenLake trajectory GIF assets.") from exc

    output_dir = ensure_dir(output_dir)
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        desc=desc,
        is_slippery=is_slippery,
        render_mode="rgb_array",
    )

    saved_frames: list[str] = []
    pil_frames = []
    rewards: list[float] = []

    def _capture_frame(step_idx: int, label: str) -> None:
        frame = np.asarray(env.render(), dtype=np.uint8)
        image = Image.fromarray(frame)
        frame_path = output_dir / f"{frame_prefix}_{step_idx:02d}_{label}.png"
        image.save(frame_path)
        saved_frames.append(str(frame_path))
        pil_frames.append(image)

    try:
        obs, _ = env.reset(seed=int(reset_seed))
        _capture_frame(0, f"s{int(obs)}")

        terminated = False
        truncated = False

        for step_idx, action in enumerate(actions, start=1):
            action = int(action)
            if action < 0 or action >= env.action_space.n:
                raise ValueError(
                    f"Invalid FrozenLake action {action}. Expected an integer in [0, {env.action_space.n - 1}]."
                )

            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            _capture_frame(step_idx, f"a{action}_s{int(obs)}")

            if terminated or truncated:
                break

        if not pil_frames:
            raise RuntimeError("No frames were captured for the FrozenLake trajectory.")

        gif_path = output_dir / gif_name
        frame_duration_ms = max(int(round(1000.0 / max(float(fps), 1e-6))), 1)
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )

        total_return = 0.0
        discounted_return = 0.0
        gamma = 1.0
        for reward in rewards:
            total_return += reward
            discounted_return += gamma * reward

        return {
            "gif_path": str(gif_path),
            "frame_paths": saved_frames,
            "steps_executed": len(rewards),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_state": int(obs),
            "total_return": float(total_return),
            "discounted_return_gamma_1": float(discounted_return),
        }
    finally:
        env.close()


def export_greedy_episode_gif(
    env_id: str,
    greedy_actions: np.ndarray,
    output_dir: str | Path,
    gif_name: str,
    max_steps: int = 100,
    fps: int = 3,
):
    """Render one greedy rollout and save PNG frames plus a GIF.

    Parameters
    ----------
    env_id : str
        Gymnasium environment id.
    greedy_actions : np.ndarray
        Greedy action index for each state.
    output_dir : str | Path
        Directory for saved frames and the GIF.
    gif_name : str
        Output GIF filename.
    max_steps : int, default=100
        Maximum number of steps in the rollout.
    fps : int, default=3
        GIF playback speed.

    Returns
    -------
    dict[str, object]
        Output paths, episode return, and episode length.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required to export rollout GIFs.") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to save rollout GIFs.") from exc

    output_path = ensure_dir(output_dir)
    env = gym.make(env_id, render_mode="rgb_array")

    obs, _ = env.reset()
    frames = [Image.fromarray(env.render())]
    episode_return = 0.0
    episode_length = 0

    try:
        for _ in range(max_steps):
            action = int(greedy_actions[int(obs)])
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            frames.append(Image.fromarray(env.render()))
            if terminated or truncated:
                break

        frame_paths: list[str] = []
        for i, frame in enumerate(frames):
            frame_path = output_path / f"frame_{i:02d}.png"
            frame.save(frame_path)
            frame_paths.append(str(frame_path))

        gif_path = output_path / gif_name
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / max(fps, 1)),
            loop=0,
        )

        return {
            "gif_path": str(gif_path),
            "frame_paths": frame_paths,
            "episode_return": episode_return,
            "episode_length": episode_length,
        }
    finally:
        env.close()

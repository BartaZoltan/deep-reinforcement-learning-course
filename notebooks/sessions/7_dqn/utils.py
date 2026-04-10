"""Shared helpers for Session 7 (DQN)."""

from __future__ import annotations

import base64
import random
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML


def set_seed(seed: int = 42) -> np.random.Generator:
    """Seed Python and NumPy, then return a reusable RNG."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def resolve_asset_root(session_dir: str | Path) -> Path:
    """Resolve the output directory for saved images and GIFs."""
    session_dir = Path(session_dir)
    asset_root = session_dir / "assets" / "cell_outputs"
    if not session_dir.exists():
        asset_root = Path("assets/cell_outputs")
    asset_root.mkdir(parents=True, exist_ok=True)
    return asset_root


def _show_and_close(fig=None):
    """Display a Matplotlib figure without leaking backend warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
            category=UserWarning,
        )
        plt.show()
    if fig is not None:
        plt.close(fig)


def moving_average(values, window: int = 10) -> np.ndarray:
    """Moving average with a growing prefix."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()

    out = np.zeros_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = np.mean(values[start : i + 1])
    return out


def plot_many_curves(curves: dict[str, np.ndarray], title: str, ylabel: str, window: int = 10):
    """Plot raw and smoothed curves in a shared color."""
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
    ax.set_xlabel("Update / Episode")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    _show_and_close(fig)


def plot_evaluation_curves(
    curves: dict[str, dict[str, np.ndarray]],
    title: str,
    ylabel: str = "Mean return",
):
    """Plot evaluation means with a same-color standard-deviation band."""
    fig, ax = plt.subplots(figsize=(9, 4))
    plotted_any = False

    for label, stats in curves.items():
        steps = np.asarray(stats.get("steps", []), dtype=float)
        means = np.asarray(stats.get("means", []), dtype=float)
        stds = np.asarray(stats.get("stds", []), dtype=float)
        if len(steps) == 0 or len(means) == 0:
            continue

        plotted_any = True
        line, = ax.plot(steps, means, linewidth=2, label=label)
        if len(stds) == len(means):
            ax.fill_between(
                steps,
                means - stds,
                means + stds,
                color=line.get_color(),
                alpha=0.18,
            )

    ax.set_title(title)
    ax.set_xlabel("Environment step")
    ax.set_ylabel(ylabel)
    if plotted_any:
        ax.legend()
    plt.tight_layout()
    _show_and_close(fig)


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    """Linear interpolation schedule clipped at the end value."""
    if duration <= 0:
        return float(end)
    mix = min(max(step / duration, 0.0), 1.0)
    return float(start + mix * (end - start))


def select_device():
    """Return the preferred torch device as a string."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def register_atari_envs():
    """Register Atari environments with Gymnasium."""
    import gymnasium as gym
    import ale_py

    gym.register_envs(ale_py)


def make_raw_pong_env(*, render_mode: str | None = None, seed: int | None = None):
    """Create the unwrapped Pong environment without preprocessing."""
    import gymnasium as gym

    register_atari_envs()
    env = gym.make(
        "ALE/Pong-v5",
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def _fire_reset_wrapper(env):
    """Press FIRE once after reset for Atari games that require it to start play."""
    import gymnasium as gym

    class FireResetWrapper(gym.Wrapper):
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            try:
                action_meanings = list(self.unwrapped.get_action_meanings())
            except Exception:
                return obs, info

            if "FIRE" not in action_meanings:
                return obs, info

            fire_action = int(action_meanings.index("FIRE"))
            obs, _, terminated, truncated, info = self.env.step(fire_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
            return obs, info

    return FireResetWrapper(env)


def export_sample_pong_gameplay_gif(
    *,
    output_dir: str | Path,
    gif_name: str,
    seed: int = 42,
    max_steps: int = 300,
    fps: float = 20.0,
):
    """Export one short raw-RGB Pong gameplay GIF using a simple stochastic policy."""
    from PIL import Image

    env = make_raw_pong_env(render_mode="rgb_array", seed=seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_name = _sanitize_gif_name(gif_name)

    rng = np.random.default_rng(seed)
    action_meanings = list(env.unwrapped.get_action_meanings())
    fire_action = action_meanings.index("FIRE") if "FIRE" in action_meanings else 0
    movement_actions = [
        idx for idx, meaning in enumerate(action_meanings) if meaning not in {"NOOP", "FIRE"}
    ] or list(range(env.action_space.n))

    env.reset(seed=seed)
    frames = [Image.fromarray(np.asarray(env.render())).convert("P", palette=Image.ADAPTIVE)]
    total_reward = 0.0
    steps = 0

    last_action = fire_action
    for step_idx in range(max_steps):
        if step_idx < 2:
            action = fire_action
        elif step_idx % 6 == 0:
            action = int(rng.choice(movement_actions))
        else:
            action = last_action

        last_action = action
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps = step_idx + 1

        frames.append(
            Image.fromarray(np.asarray(env.render())).convert("P", palette=Image.ADAPTIVE)
        )

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
        "total_return": float(total_reward),
        "steps_executed": int(steps),
    }


def make_pong_env(
    *,
    render_mode: str | None = None,
    seed: int | None = None,
    frame_skip: int = 4,
    stack_size: int = 4,
    screen_size: int = 84,
    fire_reset: bool = True,
):
    """Create a Pong environment with standard DQN-style preprocessing."""
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    register_atari_envs()
    env = gym.make(
        "ALE/Pong-v5",
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    env = AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        screen_size=screen_size,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False,
    )
    if fire_reset:
        env = _fire_reset_wrapper(env)
    env = FrameStackObservation(env, stack_size)
    if seed is not None:
        env.reset(seed=seed)
    return env


def show_pong_preprocessing_pipeline(
    *,
    seed: int = 42,
    frame_skip: int = 4,
    stack_size: int = 4,
    screen_size: int = 84,
    demo_steps: int = 6,
):
    """Visualize the raw Atari frame, the processed frame, and the final stacked observation."""
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    rng = np.random.default_rng(seed)

    raw_env = make_raw_pong_env(render_mode="rgb_array", seed=seed)

    processed_env = make_raw_pong_env(render_mode=None, seed=seed)
    processed_env = AtariPreprocessing(
        processed_env,
        frame_skip=frame_skip,
        screen_size=screen_size,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False,
    )

    stacked_env = make_raw_pong_env(render_mode=None, seed=seed)
    stacked_env = AtariPreprocessing(
        stacked_env,
        frame_skip=frame_skip,
        screen_size=screen_size,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False,
    )
    stacked_env = FrameStackObservation(stacked_env, stack_size)

    raw_env.reset(seed=seed)
    processed_obs, _ = processed_env.reset(seed=seed)
    stacked_obs, _ = stacked_env.reset(seed=seed)

    action_meanings = list(raw_env.unwrapped.get_action_meanings())
    # Use a fixed seeded action sequence so the stacked frames are not identical.
    valid_actions = [idx for idx, _ in enumerate(action_meanings)]
    action_sequence = [int(rng.choice(valid_actions)) for _ in range(demo_steps)]

    for action in action_sequence:
        _, _, raw_terminated, raw_truncated, _ = raw_env.step(action)
        processed_obs, _, processed_terminated, processed_truncated, _ = processed_env.step(action)
        stacked_obs, _, stacked_terminated, stacked_truncated, _ = stacked_env.step(action)

        if raw_terminated or raw_truncated:
            raw_env.reset(seed=seed)
        if processed_terminated or processed_truncated:
            processed_obs, _ = processed_env.reset(seed=seed)
        if stacked_terminated or stacked_truncated:
            stacked_obs, _ = stacked_env.reset(seed=seed)

    raw_frame = np.asarray(raw_env.render())
    processed_obs = np.asarray(processed_obs, dtype=np.uint8)
    stacked_obs = np.asarray(stacked_obs, dtype=np.uint8)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.6))
    axes[0].imshow(raw_frame)
    axes[0].set_title(f"Raw RGB frame\n{raw_frame.shape}")
    axes[0].axis("off")

    axes[1].imshow(processed_obs, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(
        f"After preprocessing\ncrop + grayscale + resize\n{processed_obs.shape}"
    )
    axes[1].axis("off")

    for idx in range(stack_size):
        axes[idx + 2].imshow(stacked_obs[idx], cmap="gray", vmin=0, vmax=255)
        axes[idx + 2].set_title(f"Stack frame t-{stack_size - idx - 1}")
        axes[idx + 2].axis("off")

    fig.suptitle("Pong preprocessing pipeline: from raw frame to final stacked observation")
    plt.tight_layout()
    _show_and_close(fig)

    raw_env.close()
    processed_env.close()
    stacked_env.close()

    return {
        "frame_skip": int(frame_skip),
        "screen_size": int(screen_size),
        "stack_size": int(stack_size),
        "demo_steps": int(demo_steps),
        "action_sequence": [action_meanings[a] for a in action_sequence],
        "raw_shape": tuple(raw_frame.shape),
        "processed_shape": tuple(processed_obs.shape),
        "stacked_shape": tuple(stacked_obs.shape),
        "processed_frame": processed_obs.copy(),
        "stacked_obs": stacked_obs.copy(),
    }


def show_pong_preprocessing_step_by_step(
    *,
    seed: int = 42,
    frame_skip: int = 4,
    stack_size: int = 4,
    screen_size: int = 84,
    warmup_steps: int = 8,
):
    """Visualize the preprocessing stages used before DQN sees Pong observations."""
    import cv2

    rng = np.random.default_rng(seed)
    raw_env = make_raw_pong_env(render_mode="rgb_array", seed=seed)
    action_meanings = list(raw_env.unwrapped.get_action_meanings())
    fire_action = action_meanings.index("FIRE") if "FIRE" in action_meanings else 0
    movement_actions = [
        idx for idx, meaning in enumerate(action_meanings) if meaning not in {"NOOP", "FIRE"}
    ] or list(range(raw_env.action_space.n))

    raw_env.reset(seed=seed)

    for step_idx in range(warmup_steps):
        if step_idx < 2:
            action = fire_action
        else:
            action = int(rng.choice(movement_actions))
        _, _, terminated, truncated, _ = raw_env.step(action)
        if terminated or truncated:
            raw_env.reset(seed=seed)

    repeated_action = int(rng.choice(movement_actions))
    repeated_action_name = action_meanings[repeated_action]
    raw_frames = []
    grayscale_frames = []

    for _ in range(frame_skip):
        _, _, terminated, truncated, _ = raw_env.step(repeated_action)
        raw_frames.append(np.asarray(raw_env.render()).copy())
        grayscale = np.empty(raw_env.observation_space.shape[:2], dtype=np.uint8)
        raw_env.unwrapped.ale.getScreenGrayscale(grayscale)
        grayscale_frames.append(grayscale.copy())
        if terminated or truncated:
            raw_env.reset(seed=seed)

    max_pooled = np.maximum(grayscale_frames[-2], grayscale_frames[-1])
    resized = cv2.resize(max_pooled, (screen_size, screen_size), interpolation=cv2.INTER_AREA)

    stacked_env = make_pong_env(
        seed=seed,
        frame_skip=frame_skip,
        stack_size=stack_size,
        screen_size=screen_size,
    )
    stacked_obs, _ = stacked_env.reset(seed=seed)
    stacked_obs = np.asarray(stacked_obs, dtype=np.uint8)
    stack_actions = [fire_action, fire_action] + [int(rng.choice(movement_actions)) for _ in range(stack_size + 1)]
    for action in stack_actions:
        next_obs, _, terminated, truncated, _ = stacked_env.step(action)
        stacked_obs = np.asarray(next_obs, dtype=np.uint8)
        if terminated or truncated:
            stacked_obs, _ = stacked_env.reset(seed=seed)
            stacked_obs = np.asarray(stacked_obs, dtype=np.uint8)

    fig, axes = plt.subplots(1, frame_skip, figsize=(3.4 * frame_skip, 3.0))
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.imshow(raw_frames[idx])
        ax.set_title(f"Repeated action frame {idx + 1}")
        ax.axis("off")
    fig.suptitle(
        f"Frame skipping: one chosen action (`{repeated_action_name}`) drives {frame_skip} emulator frames"
    )
    plt.tight_layout()
    _show_and_close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = np.asarray(axes)
    axes[0, 0].imshow(grayscale_frames[-2], cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Grayscale frame t-1")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(grayscale_frames[-1], cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Grayscale frame t")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(max_pooled, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Max-pooled frame")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(resized, cmap="gray", vmin=0, vmax=255)
    axes[0, 3].set_title(f"Resized to {screen_size}x{screen_size}")
    axes[0, 3].axis("off")

    for idx in range(stack_size):
        axes[1, idx].imshow(stacked_obs[idx], cmap="gray", vmin=0, vmax=255)
        axes[1, idx].set_title(f"Stack frame t-{stack_size - idx - 1}")
        axes[1, idx].axis("off")

    fig.suptitle("Preprocessing after frame skip: grayscale, max-pooling, resize, then frame stacking")
    plt.tight_layout()
    _show_and_close(fig)

    raw_env.close()
    stacked_env.close()

    return {
        "frame_skip": int(frame_skip),
        "stack_size": int(stack_size),
        "screen_size": int(screen_size),
        "repeated_action": repeated_action_name,
        "raw_frames": [frame.copy() for frame in raw_frames],
        "grayscale_frames": [frame.copy() for frame in grayscale_frames],
        "max_pooled_frame": max_pooled.copy(),
        "resized_frame": resized.copy(),
        "stacked_obs": stacked_obs.copy(),
        "stacked_shape": tuple(stacked_obs.shape),
    }


def prepare_pong_preprocessing_reference(
    *,
    seed: int = 42,
    frame_skip: int = 4,
    stack_size: int = 4,
    screen_size: int = 84,
    warmup_steps: int = 8,
    crop_rows: tuple[int, int] = (34, 194),
    crop_cols: tuple[int, int] = (0, 160),
):
    """Capture one consistent Pong preprocessing reference for stage-by-stage visualization."""
    import cv2

    rng = np.random.default_rng(seed)
    env = make_raw_pong_env(render_mode="rgb_array", seed=seed)
    action_meanings = list(env.unwrapped.get_action_meanings())
    fire_action = action_meanings.index("FIRE") if "FIRE" in action_meanings else 0
    movement_actions = [
        idx for idx, meaning in enumerate(action_meanings) if meaning not in {"NOOP", "FIRE"}
    ] or list(range(env.action_space.n))

    def _warmup(raw_env):
        raw_env.reset(seed=seed)
        for step_idx in range(warmup_steps):
            if step_idx < 2:
                action = fire_action
            else:
                action = int(rng.choice(movement_actions))
            _, _, terminated, truncated, _ = raw_env.step(action)
            if terminated or truncated:
                raw_env.reset(seed=seed)

    def _process_rgb(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        r0, r1 = crop_rows
        c0, c1 = crop_cols
        cropped = grayscale[r0:r1, c0:c1]
        resized = cv2.resize(cropped, (screen_size, screen_size), interpolation=cv2.INTER_AREA)
        return grayscale, cropped, resized

    _warmup(env)

    repeated_action = int(rng.choice(movement_actions))
    repeated_action_name = action_meanings[repeated_action]
    raw_skip_frames = []
    for _ in range(frame_skip):
        _, _, terminated, truncated, _ = env.step(repeated_action)
        raw_skip_frames.append(np.asarray(env.render()).copy())
        if terminated or truncated:
            _warmup(env)

    raw_frame = raw_skip_frames[-1]
    grayscale_frame, cropped_frame, resized_frame = _process_rgb(raw_frame)

    _warmup(env)
    stacked_frames = []
    stack_actions = [fire_action, fire_action] + [int(rng.choice(movement_actions)) for _ in range(stack_size + 2)]
    for action in stack_actions:
        _, _, terminated, truncated, _ = env.step(action)
        frame = np.asarray(env.render()).copy()
        _, _, resized = _process_rgb(frame)
        stacked_frames.append(resized.copy())
        if terminated or truncated:
            _warmup(env)

    env.close()

    stacked_frames = stacked_frames[-stack_size:]
    stacked_obs = np.stack(stacked_frames, axis=0)
    return {
        "frame_skip": int(frame_skip),
        "stack_size": int(stack_size),
        "screen_size": int(screen_size),
        "crop_rows": tuple(int(x) for x in crop_rows),
        "crop_cols": tuple(int(x) for x in crop_cols),
        "repeated_action": repeated_action_name,
        "raw_skip_frames": [frame.copy() for frame in raw_skip_frames],
        "raw_frame": raw_frame.copy(),
        "grayscale_frame": grayscale_frame.copy(),
        "cropped_frame": cropped_frame.copy(),
        "resized_frame": resized_frame.copy(),
        "stacked_frames": [frame.copy() for frame in stacked_frames],
        "stacked_obs": stacked_obs.copy(),
    }


def plot_pong_frame_skip_showcase(reference: dict):
    """Show the raw emulator frames produced while one action is being repeated."""
    frames = reference["raw_skip_frames"]
    repeated_action = reference["repeated_action"]
    fig, axes = plt.subplots(1, len(frames), figsize=(3.4 * len(frames), 3.0))
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.imshow(frames[idx])
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")
    fig.suptitle(f"Frame skipping demo: repeating `{repeated_action}` over consecutive raw frames")
    plt.tight_layout()
    _show_and_close(fig)


def plot_pong_grayscale_showcase(reference: dict):
    """Compare one raw RGB frame with its grayscale version."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    axes[0].imshow(reference["raw_frame"])
    axes[0].set_title(f"Raw RGB frame\n{reference['raw_frame'].shape}")
    axes[0].axis("off")

    axes[1].imshow(reference["grayscale_frame"], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Grayscale frame\n{reference['grayscale_frame'].shape}")
    axes[1].axis("off")

    fig.suptitle("Color is removed so the network focuses on geometry and motion cues")
    plt.tight_layout()
    _show_and_close(fig)


def plot_pong_crop_showcase(reference: dict):
    """Show the crop region and the resulting cropped frame."""
    from matplotlib.patches import Rectangle

    grayscale = reference["grayscale_frame"]
    cropped = reference["cropped_frame"]
    r0, r1 = reference["crop_rows"]
    c0, c1 = reference["crop_cols"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].imshow(grayscale, cmap="gray", vmin=0, vmax=255)
    axes[0].add_patch(
        Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor="#D1495B", linewidth=2.0)
    )
    axes[0].set_title("Crop region on the grayscale frame")
    axes[0].axis("off")

    axes[1].imshow(cropped, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Cropped play area\n{cropped.shape}")
    axes[1].axis("off")

    fig.suptitle("Cropping removes borders and keeps the visually important game area")
    plt.tight_layout()
    _show_and_close(fig)


def plot_pong_resize_showcase(reference: dict):
    """Compare the cropped frame with the final resized 84x84 frame."""
    cropped = reference["cropped_frame"]
    resized = reference["resized_frame"]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].imshow(cropped, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Cropped frame\n{cropped.shape}")
    axes[0].axis("off")

    axes[1].imshow(resized, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Resized frame\n{resized.shape}")
    axes[1].axis("off")

    fig.suptitle("Resizing produces the compact input resolution used by DQN")
    plt.tight_layout()
    _show_and_close(fig)


def plot_pong_stack_showcase(reference: dict):
    """Show the final stacked processed frames used as the network input."""
    plot_frame_stack(np.asarray(reference["stacked_obs"]), title="The final stacked Pong observation")


def plot_frame_stack(obs: np.ndarray, title: str = "Stacked Pong frames"):
    """Show the stacked grayscale frames of one Atari observation."""
    obs = np.asarray(obs)
    n_frames = obs.shape[0]
    fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 3))
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.imshow(obs[idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Frame t-{n_frames - idx - 1}")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    _show_and_close(fig)


def summarize_pong_env(env) -> dict:
    """Return a small summary dictionary for the wrapped Pong environment."""
    action_meanings = []
    try:
        action_meanings = list(env.unwrapped.get_action_meanings())
    except Exception:
        pass

    return {
        "action_space": str(env.action_space),
        "observation_space": str(env.observation_space),
        "observation_shape": tuple(np.asarray(env.observation_space.shape).tolist()),
        "action_meanings": action_meanings,
    }


def _sanitize_gif_name(name: str) -> str:
    raw = str(name).replace("\\", "_").replace("/", "_")
    stem = raw[:-4] if raw.lower().endswith(".gif") else raw
    stem = stem or "rollout"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "rollout"
    return f"{safe_stem}.gif"


def export_atari_policy_gif(
    q_network,
    *,
    device,
    output_dir: str | Path,
    gif_name: str,
    seed: int = 42,
    max_steps: int = 4000,
    fps: float = 20.0,
):
    """Render one greedy Pong rollout from a trained Q-network."""
    from PIL import Image
    import torch

    env = make_pong_env(render_mode="rgb_array", seed=seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_name = _sanitize_gif_name(gif_name)

    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.uint8)
    frames = []
    total_return = 0.0
    steps = 0
    q_network.eval()

    frame = env.render()
    frames.append(Image.fromarray(np.asarray(frame)).convert("P", palette=Image.ADAPTIVE))

    with torch.no_grad():
        for step_idx in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            action = int(torch.argmax(q_network(obs_tensor), dim=1).item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.asarray(next_obs, dtype=np.uint8)
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
        "total_return": float(total_return),
        "steps_executed": int(steps),
    }


def render_side_by_side_gifs(items: list[dict[str, str]], width: int = 360):
    """Render one or more GIFs side by side."""
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

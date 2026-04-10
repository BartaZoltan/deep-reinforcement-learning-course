"""Shared helpers for Session 8 (Rainbow DQN components)."""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_seed(seed: int = 42) -> np.random.Generator:
    """Seed Python and NumPy, then return a reusable RNG."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def resolve_asset_root(session_dir: str | Path) -> Path:
    """Resolve the output directory for saved figures."""
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


def select_device():
    """Return the preferred torch device as a string."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_prioritized_replay_demo(td_errors, alpha: float = 0.6, beta: float = 0.4):
    """Show replay probabilities and importance weights from toy TD errors."""
    td_errors = np.asarray(td_errors, dtype=float)
    priorities = np.abs(td_errors) + 1e-6
    probs = priorities**alpha
    probs /= probs.sum()
    weights = (len(td_errors) * probs) ** (-beta)
    weights /= weights.max()

    labels = [f"t{i}" for i in range(len(td_errors))]
    x = np.arange(len(td_errors))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(x, probs, color="#2C6FB7")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Sampling probabilities")
    axes[0].set_ylabel("Probability")

    axes[1].bar(x, weights, color="#D1495B")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Importance-sampling weights")
    axes[1].set_ylabel("Normalized weight")

    fig.suptitle(f"Prioritized replay toy example (alpha={alpha}, beta={beta})")
    plt.tight_layout()
    _show_and_close(fig)

    return {"priorities": priorities, "probabilities": probs, "weights": weights}


def plot_dueling_decomposition(
    value: float,
    advantages,
    action_labels: list[str] | None = None,
):
    """Visualize the value and centered advantages used by a dueling head."""
    advantages = np.asarray(advantages, dtype=float)
    centered = advantages - advantages.mean()
    q_values = value + centered
    action_labels = action_labels or [f"a{i}" for i in range(len(advantages))]
    x = np.arange(len(advantages))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].bar([0], [value], color="#4C956C")
    axes[0].set_xticks([0], ["state"])
    axes[0].set_title("Value stream V(s)")
    axes[0].set_ylabel("Score")

    axes[1].bar(x, centered, color="#D1495B")
    axes[1].axhline(0.0, color="gray", linewidth=1)
    axes[1].set_xticks(x, action_labels)
    axes[1].set_title("Centered advantage stream")

    axes[2].bar(x, q_values, color="#2C6FB7")
    axes[2].set_xticks(x, action_labels)
    axes[2].set_title("Combined Q(s, a)")

    fig.suptitle("Dueling decomposition: value plus centered advantages")
    plt.tight_layout()
    _show_and_close(fig)

    return {"centered_advantages": centered, "q_values": q_values}


def plot_n_step_targets(rewards, gamma: float, bootstrap_value: float, n_values=(1, 3, 5)):
    """Compare a few n-step targets on one toy reward sequence."""
    rewards = np.asarray(rewards, dtype=float)
    targets = []
    for n in n_values:
        horizon = min(n, len(rewards))
        discounted = sum((gamma**k) * rewards[k] for k in range(horizon))
        if n <= len(rewards):
            discounted += (gamma**n) * bootstrap_value
        targets.append(discounted)

    x = np.arange(len(n_values))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x, targets, color="#2C6FB7")
    ax.set_xticks(x, [f"{n}-step" for n in n_values])
    ax.set_ylabel("Target value")
    ax.set_title("How the backup horizon changes the target")
    plt.tight_layout()
    _show_and_close(fig)

    return {"targets": np.asarray(targets, dtype=float)}


def plot_noisy_action_value_samples(samples, action_labels: list[str] | None = None):
    """Plot repeated action-value predictions under noisy layers."""
    samples = np.asarray(samples, dtype=float)
    action_labels = action_labels or [f"a{i}" for i in range(samples.shape[1])]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for action_idx, label in enumerate(action_labels):
        ax.plot(samples[:, action_idx], marker="o", linewidth=1.5, label=label)

    ax.set_title("Noisy layer outputs across repeated forward passes")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Predicted action value")
    ax.legend(ncol=min(3, len(action_labels)))
    plt.tight_layout()
    _show_and_close(fig)


def plot_categorical_distribution(support, probs, title: str):
    """Show one categorical value distribution on the C51 support."""
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)
    expected = float(np.sum(support * probs))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(support, probs, width=(support[1] - support[0]) * 0.9, color="#2C6FB7")
    ax.axvline(expected, color="#D1495B", linestyle="--", linewidth=2, label=f"Expectation = {expected:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Return atom")
    ax.set_ylabel("Probability")
    ax.legend()
    plt.tight_layout()
    _show_and_close(fig)

    return {"expected_value": expected}


def plot_distribution_projection(old_support, old_probs, new_support, new_probs):
    """Compare a pre-projection and post-projection categorical distribution."""
    old_support = np.asarray(old_support, dtype=float)
    old_probs = np.asarray(old_probs, dtype=float)
    new_support = np.asarray(new_support, dtype=float)
    new_probs = np.asarray(new_probs, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].bar(old_support, old_probs, width=(old_support[1] - old_support[0]) * 0.9, color="#4C956C")
    axes[0].set_title("Shifted target distribution")
    axes[0].set_xlabel("Return atom")
    axes[0].set_ylabel("Probability")

    axes[1].bar(new_support, new_probs, width=(new_support[1] - new_support[0]) * 0.9, color="#2C6FB7")
    axes[1].set_title("Projected back onto fixed support")
    axes[1].set_xlabel("Return atom")

    fig.suptitle("Distributional projection in C51")
    plt.tight_layout()
    _show_and_close(fig)


def print_rainbow_component_table():
    """Print a compact overview of the Rainbow ingredients used in this notebook."""
    components = [
        ("Double DQN", "Reduces overestimation by separating action selection and evaluation"),
        ("Prioritized Replay", "Samples more informative transitions more often"),
        ("Dueling Network", "Separates state value from action-specific advantage"),
        ("n-step Returns", "Propagates rewards faster through the target"),
        ("Noisy Layers", "Replaces hand-designed epsilon schedules with learned exploration"),
        ("Distributional RL", "Predicts a return distribution instead of only an expectation"),
    ]
    for name, description in components:
        print(f"- {name}: {description}")

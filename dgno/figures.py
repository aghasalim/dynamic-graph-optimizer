"""Generate the figures used in the README.

``python -m dgno.figures`` writes everything into ``docs/``.  Kept separate from
``visualize`` because that module animates an episode, while this one produces
the static comparisons the write-up refers to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import csv

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from .baselines import BackpressurePolicy, ShortestPathPolicy
from .env import DynamicRoutingEnv
from .transfer import run_transfer
from .visualize import draw_network_state, record_episode

__all__ = ["policy_comparison", "learning_curve", "transfer_scaling"]

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "ppo-gnn-4m"
DOCS = Path(__file__).resolve().parents[1] / "docs"


def policy_comparison(out: Path, step: int = 95, seed: int = 0) -> Path:
    """Draw three policies at the same timestep of the same episode, side by side."""
    agent = PPO.load(str(CHECKPOINT), device="cpu")
    env = DynamicRoutingEnv(seed=seed)
    panels = [
        ("shortest-path", ShortestPathPolicy(env)),
        ("backpressure", BackpressurePolicy(env)),
        ("PPO + GNN (4M)", agent),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (label, policy) in zip(axes, panels, strict=True):
        recording = record_episode(env, policy, seed=seed, max_steps=step + 1)
        frame = recording.frames[step]
        draw_network_state(
            ax,
            env.sim.net,
            frame,
            env.config.edge_capacity,
            title=(
                f"{label}\nthroughput {recording.throughput[step]:.1f}   "
                f"peak backlog {frame['queues'].max():.2f}"
            ),
            labels=False,
        )
    figure.suptitle(
        f"same episode, same seed, t={step}. node colour is backlog, "
        "edge width is flow, edge colour is the routing offset",
        fontsize=10,
        y=0.04,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def learning_curve(out: Path) -> Path:
    """Plot training reward against the deterministic evaluations of the same run.

    Two separate points: the rollout curve rises steadily, and it sits well below
    what the deterministic policy actually achieves, because rollouts are collected
    under a Gaussian with std ~0.35 and the exploration noise costs reward.
    """
    steps, rewards = [], []
    with open(DOCS / "curve-long4M.csv") as handle:
        for row in csv.DictReader(handle):
            if row.get("rollout/ep_rew_mean"):
                steps.append(float(row["time/total_timesteps"]) / 1e6)
                rewards.append(float(row["rollout/ep_rew_mean"]))
    x, y = np.asarray(steps), np.asarray(rewards)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    left.plot(x, y, lw=0.8, color="#2166ac", alpha=0.55, label="rollout ep_rew_mean")
    settled = x > 0.2
    fit = np.poly1d(np.polyfit(x[settled], y[settled], 1))
    left.plot(x[settled], fit(x[settled]), lw=1.8, color="#b2182b",
              label=f"trend, {fit.coefficients[0]:+.1f} per M steps")
    left.axhline(272.5, ls="--", lw=1.0, color="#1a9850",
                 label="deterministic return at 4M")
    left.set_title(
        "training reward rises, and sits far below\nwhat the greedy policy scores"
    )
    left.set_xlabel("million steps")
    left.set_ylabel("episode return")
    left.legend(fontsize=8, frameon=False, loc="lower right")
    left.spines[["top", "right"]].set_visible(False)

    right.plot([0.4, 4.0], [0.891, 0.929], "o-", color="#b2182b",
               label="PPO (deterministic, 2 checkpoints)")
    right.axhline(0.901, ls="--", lw=1.0, color="#1a9850", label="backpressure")
    right.axhline(0.869, ls=":", lw=1.0, color="0.4", label="shortest-path")
    right.set_title("served fraction: crosses backpressure\nbetween the two checkpoints")
    right.set_xlabel("million steps")
    right.set_ylabel("served fraction")
    right.set_xlim(0, 4.4)
    right.legend(fontsize=8, frameon=False, loc="lower right")
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def transfer_scaling(out: Path, episodes: int = 10) -> Path:
    """Plot served fraction and peak backlog against grid size, without retraining."""
    grids = [(3, 4), (4, 5), (5, 6), (6, 8), (8, 10)]
    results = run_transfer(str(CHECKPOINT), grids, episodes=episodes)
    x = [r.num_nodes for r in results]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    series = [
        ("shortest-path", "shortest_path", "0.4", ":"),
        ("backpressure", "backpressure", "#1a9850", "--"),
        ("PPO, trained on 4x5", "transferred", "#b2182b", "-"),
    ]
    for label, attr, colour, style in series:
        left.plot(
            x, [getattr(r, attr)["served_fraction"] for r in results],
            style, color=colour, marker="o", label=label,
        )
        right.plot(
            x, [getattr(r, attr)["mean_peak_queue"] for r in results],
            style, color=colour, marker="o", label=label,
        )
    left.axvline(20, lw=0.8, color="0.75")
    right.axvline(20, lw=0.8, color="0.75")
    left.set_title("served fraction (higher is better)")
    right.set_title("peak backlog (lower is better)")
    for ax in (left, right):
        ax.set_xlabel("nodes  (grey line = the grid it trained on)")
        ax.spines[["top", "right"]].set_visible(False)
    left.legend(fontsize=8, frameon=False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    DOCS.mkdir(exist_ok=True)
    for path in (
        policy_comparison(DOCS / "policy-comparison.png"),
        learning_curve(DOCS / "learning-curve.png"),
        transfer_scaling(DOCS / "transfer-scaling.png", episodes=args.episodes),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

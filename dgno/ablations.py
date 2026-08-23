"""Re-derive the ablation table from the committed checkpoints.

Every agent is scored under the same reward and the same seeds, including the one
trained with ``bottleneck_mode=absolute``.  That makes the ``return`` column
comparable across rows, which it is not in the individual ``runs/*/evaluation.txt``
files -- those each use the reward the agent was trained on.  ``served``,
``peak_q`` and ``churn`` are reward-independent and identical either way.

``python -m dgno.ablations``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from .baselines import BackpressurePolicy, ShortestPathPolicy, compare_policies
from .env import DynamicRoutingEnv

__all__ = ["CHECKPOINTS", "build_policy_table"]

ROOT = Path(__file__).resolve().parents[1] / "checkpoints"

#: Label -> checkpoint, in the order the README discusses them.
CHECKPOINTS = {
    "ppo 4M": ROOT / "ppo-gnn-4m.zip",
    "ppo 400k (baseline)": ROOT / "ablations" / "shaping-400k.zip",
    "ppo 400k gain 1.0": ROOT / "ablations" / "gain-1.0-400k.zip",
    "ppo 400k no churn": ROOT / "ablations" / "no-churn-400k.zip",
    "ppo 400k both": ROOT / "ablations" / "gain-and-no-churn-400k.zip",
    "ppo 400k absolute": ROOT / "ablations" / "absolute-400k.zip",
    "ppo 4M entropy": ROOT / "ablations" / "entropy-4m.zip",
}


def build_policy_table(env: DynamicRoutingEnv) -> dict:
    """Baselines plus every committed checkpoint, ready for ``compare_policies``."""
    policies: dict = {
        "shortest-path": ShortestPathPolicy(env),
        "backpressure": BackpressurePolicy(env),
    }
    for label, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint for '{label}': {path}")
        policies[label] = PPO.load(str(path.with_suffix("")), device="cpu")
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=str, default="docs/ablations.txt")
    args = parser.parse_args()

    env = DynamicRoutingEnv(seed=0)
    table = compare_policies(build_policy_table(env), env, episodes=args.episodes)
    header = (
        f"all policies scored under the same reward and seeds, "
        f"{args.episodes} episodes, 4x5 grid"
    )
    print(f"{header}\n\n{table}")
    Path(args.out).write_text(f"{header}\n\n{table}\n")


if __name__ == "__main__":
    main()

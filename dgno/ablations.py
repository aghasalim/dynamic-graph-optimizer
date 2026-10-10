"""Re-derive the ablation table from the committed checkpoints.

Every agent is scored under the same reward and the same seeds, including the one
trained with ``bottleneck_mode=absolute``.  That makes the ``return`` column
comparable across rows, which it is not in the individual ``runs/*/evaluation.txt``
files, those each use the reward the agent was trained on.  ``served``,
``peak_q`` and ``churn`` are reward-independent and identical either way.

``python -m dgno.ablations``
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO

from .baselines import (
    BackpressurePolicy,
    ShortestPathPolicy,
    compare_policies,
    paired_difference,
)
from .env import DynamicRoutingEnv

__all__ = ["CHECKPOINTS", "build_policy_table", "paired_table"]

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


EPISODES_CSV = ROOT.parent / "verify" / "data" / "episode-metrics.csv"


def paired_table(path: Path = EPISODES_CSV, against: str = "backpressure") -> str:
    """Return minus one policy's return, paired by episode.

    Read from the per episode metrics behind docs/ablations.txt.
    """
    returns: dict[str, dict[int, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            returns.setdefault(row["policy"], {})[int(row["episode"])] = float(row["ret"])
    base = returns[against]
    episodes = sorted(base)
    lines = [
        f"return minus {against}, paired by episode over the same {len(episodes)} seeds,",
        "mean and 95% bootstrap interval",
        "",
        f"{'policy':<22}{'diff':>9}{'low':>9}{'high':>9}",
        "-" * 49,
    ]
    for name, rets in returns.items():
        if name == against:
            continue
        mean, lo, hi = paired_difference(
            [rets[e] for e in episodes], [base[e] for e in episodes]
        )
        lines.append(f"{name:<22}{mean:>9.2f}{lo:>9.2f}{hi:>9.2f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=str, default="docs/ablations.txt")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="write docs/paired-returns.txt from verify/data/episode-metrics.csv instead",
    )
    args = parser.parse_args()

    if args.paired:
        text = paired_table()
        print(text, end="")
        Path("docs/paired-returns.txt").write_text(text)
        return

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

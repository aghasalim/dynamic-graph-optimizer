"""Move a trained policy onto a different grid size without retraining.

Everything in the encoder is shared across nodes and edges, so the only
size-dependent tensors in the policy are the ``edge_index`` buffer and the
Gaussian ``log_std`` (one entry per action).  Rebuilding the policy for a new
topology and copying every shape-compatible parameter therefore transfers the
learned control rule intact, which is the concrete payoff of keeping the
architecture permutation-equivariant.

``python -m dgno.transfer --model runs/long4M/agent --grids 3x4,4x5,6x8,8x10``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from stable_baselines3 import PPO

from .baselines import BackpressurePolicy, ShortestPathPolicy, evaluate_policy
from .env import DynamicRoutingEnv, RewardConfig
from .models import GraphActorCriticPolicy, GraphFeaturesExtractor
from .simulator import NetworkConfig
from .train import TrainConfig

__all__ = ["TransferResult", "rehost_policy", "run_transfer"]


@dataclass
class TransferResult:
    """One grid's outcome, with the baselines it was scored against."""

    grid: str
    num_nodes: int
    num_edges: int
    transferred: dict[str, float]
    shortest_path: dict[str, float]
    backpressure: dict[str, float]


def rehost_policy(
    source: PPO, env: DynamicRoutingEnv, config: TrainConfig
) -> GraphActorCriticPolicy:
    """Rebuild the policy for ``env``'s topology and copy the trained weights.

    Returns the new policy.  Any tensor whose shape depends on the graph size is
    left at its fresh initialisation; for deterministic evaluation that is only
    ``log_std``, which is unused.
    """
    policy = GraphActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: config.learning_rate,
        features_extractor_class=GraphFeaturesExtractor,
        features_extractor_kwargs={
            "graph_spec": env.spec_graph,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "heads": config.heads,
            "edge_latent_dim": config.edge_latent_dim,
        },
        vf_hidden=config.vf_hidden,
        action_head_gain=config.action_head_gain,
        log_std_init=config.log_std_init,
    )

    trained = source.policy.state_dict()
    target = policy.state_dict()
    copied, skipped = [], []
    for name, tensor in target.items():
        if name in trained and trained[name].shape == tensor.shape:
            target[name] = trained[name].clone()
            copied.append(name)
        else:
            skipped.append(name)
    policy.load_state_dict(target)
    policy.set_training_mode(False)
    policy._transfer_report = (copied, skipped)  # noqa: SLF001
    return policy


def run_transfer(
    model_path: str,
    grids: list[tuple[int, int]],
    episodes: int = 10,
    seed: int = 12345,
    config: TrainConfig | None = None,
) -> list[TransferResult]:
    """Score the transferred policy against both baselines on each grid."""
    config = config or TrainConfig()
    source = PPO.load(model_path, device="cpu")
    reward = RewardConfig(gamma=config.gamma)

    results = []
    for rows, cols in grids:
        env = DynamicRoutingEnv(
            config=NetworkConfig(rows=rows, cols=cols), reward=reward, seed=seed
        )
        policy = rehost_policy(source, env, config)
        with torch.no_grad():
            results.append(
                TransferResult(
                    grid=f"{rows}x{cols}",
                    num_nodes=env.spec_graph.num_nodes,
                    num_edges=env.spec_graph.num_edges,
                    transferred=evaluate_policy(policy, env, episodes, seed),
                    shortest_path=evaluate_policy(
                        ShortestPathPolicy(env), env, episodes, seed
                    ),
                    backpressure=evaluate_policy(
                        BackpressurePolicy(env), env, episodes, seed
                    ),
                )
            )
    return results


def format_results(results: list[TransferResult]) -> str:
    """Render the transfer table."""
    header = (
        f"{'grid':>6} {'nodes':>6} {'edges':>6}  "
        f"{'served: sp':>11} {'bp':>7} {'ppo':>7}   "
        f"{'peak_q: sp':>11} {'bp':>7} {'ppo':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.grid:>6} {r.num_nodes:>6} {r.num_edges:>6}  "
            f"{r.shortest_path['served_fraction']:>11.3f} "
            f"{r.backpressure['served_fraction']:>7.3f} "
            f"{r.transferred['served_fraction']:>7.3f}   "
            f"{r.shortest_path['mean_peak_queue']:>11.3f} "
            f"{r.backpressure['mean_peak_queue']:>7.3f} "
            f"{r.transferred['mean_peak_queue']:>7.3f}"
        )
    return "\n".join(lines)


def _parse_grids(text: str) -> list[tuple[int, int]]:
    grids = []
    for chunk in text.split(","):
        rows, _, cols = chunk.strip().partition("x")
        grids.append((int(rows), int(cols)))
    return grids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="runs/long4M/agent")
    parser.add_argument("--grids", type=str, default="3x4,4x5,6x8,8x10")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=str, default="docs/transfer.txt")
    args = parser.parse_args()

    results = run_transfer(args.model, _parse_grids(args.grids), episodes=args.episodes)
    table = format_results(results)
    print(f"trained on 4x5, evaluated with no retraining\n\n{table}")
    with open(args.out, "w") as handle:
        handle.write(f"trained on 4x5, evaluated with no retraining\n\n{table}\n")


if __name__ == "__main__":
    main()

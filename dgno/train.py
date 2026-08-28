"""PPO training loop over the GNN state representation.

Run ``python -m dgno.train --timesteps 300000`` to train, then evaluate against
the classical baselines on identical seeds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .baselines import BackpressurePolicy, ShortestPathPolicy, compare_policies
from .env import DynamicRoutingEnv, RewardConfig
from .models import GraphActorCriticPolicy, GraphFeaturesExtractor
from .simulator import NetworkConfig

__all__ = ["TrainConfig", "build_vec_env", "build_agent", "train"]


@dataclass
class TrainConfig:
    """PPO and architecture hyper-parameters.

    ``n_steps`` is deliberately a fraction of the 300-step horizon: rush-hour
    demand has a 120-step period, so a rollout that spans several periods keeps
    the advantage estimates from being dominated by whichever phase of the cycle
    the batch happened to land in.

    ``target_kl`` and the modest ``n_epochs`` are not decoration: with a 62-dim
    continuous action the default 10 epochs at 3e-4 drives ``clip_fraction`` past
    0.3, which means most of each minibatch is being clipped and the effective
    update is both large and biased.  ``log_std_init = -1`` starts the Gaussian at
    sigma ~ 0.37 rather than 1.0, at sigma = 1 nearly every sampled action
    saturates the [-1, 1] box, so the agent spends its early samples exploring
    routing configurations it can never actually execute.
    """

    total_timesteps: int = 300_000
    num_envs: int = 8
    n_steps: int = 256
    batch_size: int = 512
    n_epochs: int = 5
    learning_rate: float = 2e-4
    target_kl: float | None = 0.02
    log_std_init: float = -1.0
    action_head_gain: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    num_layers: int = 3
    heads: int = 4
    edge_latent_dim: int = 16
    vf_hidden: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        rollout = self.num_envs * self.n_steps
        if rollout % self.batch_size != 0:
            raise ValueError(
                f"batch_size {self.batch_size} must divide the rollout size {rollout}"
            )


def make_env_fn(
    network: NetworkConfig, reward: RewardConfig, seed: int, rank: int = 0
):
    """Build a factory that constructs one monitored environment instance."""

    def _init() -> Monitor:
        env = DynamicRoutingEnv(config=network, reward=reward, seed=seed + rank)
        return Monitor(env)

    return _init


def build_vec_env(
    config: TrainConfig,
    network: NetworkConfig,
    reward: RewardConfig,
    subprocess: bool = False,
) -> VecNormalize:
    """Vectorised training env.

    Observations are already normalised by construction, so only the reward is
    running-normalised, reward scale acts as an implicit multiplier on the value
    loss, and the throughput and shaping terms differ by an order of magnitude.
    """
    fns = [
        make_env_fn(network, reward, config.seed, rank) for rank in range(config.num_envs)
    ]
    vec_cls = SubprocVecEnv if subprocess and config.num_envs > 1 else DummyVecEnv
    return VecNormalize(vec_cls(fns), norm_obs=False, norm_reward=True, gamma=config.gamma)


def build_agent(venv: VecNormalize, config: TrainConfig, probe: DynamicRoutingEnv) -> PPO:
    """Assemble PPO with the graph feature extractor and equivariant policy."""
    policy_kwargs = {
        "features_extractor_class": GraphFeaturesExtractor,
        "features_extractor_kwargs": {
            "graph_spec": probe.spec_graph,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "heads": config.heads,
            "edge_latent_dim": config.edge_latent_dim,
        },
        "vf_hidden": config.vf_hidden,
        "log_std_init": config.log_std_init,
        "action_head_gain": config.action_head_gain,
    }
    return PPO(
        GraphActorCriticPolicy,
        venv,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        verbose=1,
    )


def train(
    config: TrainConfig | None = None,
    network: NetworkConfig | None = None,
    reward: RewardConfig | None = None,
    output_dir: str | Path = "runs/ppo",
    subprocess: bool = False,
    episodes: int = 10,
) -> tuple[PPO, str]:
    """Train the agent and return it alongside the baseline comparison table."""
    config = config or TrainConfig()
    network = network or NetworkConfig()
    reward = reward or RewardConfig(gamma=config.gamma)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    venv = build_vec_env(config, network, reward, subprocess=subprocess)
    probe = DynamicRoutingEnv(config=network, reward=reward, seed=config.seed)
    agent = build_agent(venv, config, probe)
    # CSV alongside stdout so the learning curve survives the run; `ep_rew_mean`
    # comes off the Monitor wrapper, i.e. raw episode reward before VecNormalize.
    agent.set_logger(configure(str(out), ["stdout", "csv"]))
    agent.learn(total_timesteps=config.total_timesteps, progress_bar=False)

    agent.save(out / "agent")
    venv.save(str(out / "vecnormalize.pkl"))
    venv.close()

    table = compare_policies(
        {
            "shortest-path": ShortestPathPolicy(probe),
            "backpressure": BackpressurePolicy(probe),
            "ppo-gnn": agent,
        },
        probe,
        episodes=episodes,
    )
    (out / "evaluation.txt").write_text(table + "\n")
    return agent, table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=TrainConfig.total_timesteps)
    parser.add_argument("--num-envs", type=int, default=TrainConfig.num_envs)
    parser.add_argument("--rows", type=int, default=NetworkConfig.rows)
    parser.add_argument("--cols", type=int, default=NetworkConfig.cols)
    parser.add_argument("--layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--ent-coef", type=float, default=TrainConfig.ent_coef)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--bottleneck-mode",
        choices=("shaping", "absolute"),
        default=RewardConfig.bottleneck_mode,
    )
    parser.add_argument(
        "--action-head-gain", type=float, default=TrainConfig.action_head_gain
    )
    parser.add_argument(
        "--smoothness-weight", type=float, default=RewardConfig.smoothness_weight
    )
    parser.add_argument("--out", type=str, default="runs/ppo")
    parser.add_argument("--subprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TrainConfig(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        num_layers=args.layers,
        seed=args.seed,
        action_head_gain=args.action_head_gain,
        ent_coef=args.ent_coef,
    )
    network = NetworkConfig(rows=args.rows, cols=args.cols)
    reward = RewardConfig(
        bottleneck_mode=args.bottleneck_mode,
        smoothness_weight=args.smoothness_weight,
        gamma=config.gamma,
    )

    print("train config:", asdict(config))
    _, table = train(
        config=config,
        network=network,
        reward=reward,
        output_dir=args.out,
        subprocess=args.subprocess,
        episodes=args.episodes,
    )
    print("\nEvaluation on shared seeds\n")
    print(table)


if __name__ == "__main__":
    main()

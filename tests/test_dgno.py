"""Self-checks for the simulator invariants and the GNN batching path.

Runs under pytest or as a plain script: ``python tests/test_dgno.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgno.baselines import BackpressurePolicy, ShortestPathPolicy, evaluate_policy
from dgno.env import DynamicRoutingEnv, RewardConfig, smooth_max
from dgno.models import GraphFeaturesExtractor
from dgno.simulator import NetworkConfig, NetworkSimulator
from dgno.train import TrainConfig, build_agent, build_vec_env


def test_routing_split_is_a_distribution_per_node() -> None:
    sim = NetworkSimulator(NetworkConfig(), np.random.default_rng(0))
    rng = np.random.default_rng(1)
    for _ in range(20):
        split = sim.routing_split(rng.uniform(-1.0, 1.0, sim.net.num_edges))
        totals = np.add.reduceat(split, sim.net.out_start)
        assert np.allclose(totals, 1.0), "out-edge probabilities must sum to one"
        assert np.all(split >= 0.0)


def test_mass_is_conserved() -> None:
    """Accepted demand must equal delivered throughput plus backlog still queued.

    This is the invariant that spillback rationing is most likely to break: if the
    single-pass scale factor ever over- or under-corrects, mass appears or vanishes.
    """
    sim = NetworkSimulator(NetworkConfig(), np.random.default_rng(7))
    rng = np.random.default_rng(8)
    accepted = delivered = 0.0
    for _ in range(400):
        measurements = sim.step(rng.uniform(-1.0, 1.0, sim.net.num_edges))
        accepted += measurements.offered - measurements.dropped
        delivered += measurements.throughput
        assert np.all(sim.queues >= -1e-9), "backlog went negative"
        assert np.all(sim.queues <= sim.config.queue_capacity + 1e-9), "backlog overflowed"
    residual = accepted - (delivered + float(np.sum(sim.queues)))
    assert abs(residual) < 1e-6 * max(accepted, 1.0), f"mass leak of {residual}"


def test_smooth_max_interpolates_between_mean_and_max() -> None:
    """The surrogate is sandwiched by mean and max, and approaches max as tau -> 0.

    The gap to the true maximum closes at rate ``tau * log(N)``, not instantly, so
    the small-tau check is a tolerance rather than an equality.
    """
    values = np.array([0.1, 0.4, 0.95, 0.2])
    gap = 1e-4 * np.log(values.size)
    assert abs(smooth_max(values, 1e-4) - float(np.max(values))) <= gap + 1e-9
    assert abs(smooth_max(values, 1e3) - float(np.mean(values))) < 1e-3
    for tau in (0.05, 0.25, 1.0, 5.0):
        assert float(np.mean(values)) <= smooth_max(values, tau) <= float(np.max(values))


def test_shaping_reward_telescopes_over_a_rollout() -> None:
    """The congestion term must telescope, which is what makes it policy-invariant.

    With gamma=1 and every other weight zeroed, the summed reward over any rollout
    collapses to ``Phi(start) - Phi(end)`` regardless of the actions taken.  If it
    does not, the agent can farm the shaping term in a cycle and the invariance
    argument in the README is void.
    """
    reward = RewardConfig(
        gamma=1.0,
        throughput_weight=0.0,
        drop_weight=0.0,
        smoothness_weight=0.0,
        bottleneck_weight=1.0,
    )
    env = DynamicRoutingEnv(reward=reward, seed=0)
    env.reset(seed=0)
    start = env._potential()
    total = 0.0
    for _ in range(80):
        _, step_reward, terminated, truncated, _ = env.step(env.action_space.sample())
        total += step_reward
        if terminated or truncated:
            break
    assert abs(total - (start - env._potential())) < 1e-9, "shaping term does not telescope"


def test_observation_stays_inside_the_declared_space() -> None:
    env = DynamicRoutingEnv(seed=3)
    observation, _ = env.reset(seed=3)
    assert env.observation_space.contains(observation)
    for _ in range(120):
        observation, reward, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        if terminated or truncated:
            break


def test_graph_extractor_batching_matches_single_samples() -> None:
    """Batched forward must equal per-sample forwards.

    The batch is built by offsetting a replicated ``edge_index`` by ``b * num_nodes``.
    An off-by-one there silently wires batch elements together and still trains,
    just badly -- so this is the one bug worth a dedicated check.
    """
    env = DynamicRoutingEnv(seed=5)
    extractor = GraphFeaturesExtractor(
        env.observation_space, env.spec_graph, hidden_dim=32, num_layers=2, heads=4
    ).eval()

    samples = []
    observation, _ = env.reset(seed=5)
    for _ in range(6):
        samples.append(observation)
        observation, *_ = env.step(env.action_space.sample())
    batch = torch.as_tensor(np.stack(samples), dtype=torch.float32)

    with torch.no_grad():
        together = extractor(batch)
        apart = torch.cat([extractor(batch[i : i + 1]) for i in range(batch.shape[0])])
    assert torch.allclose(together, apart, atol=1e-5), "batch elements are leaking"


def test_policy_action_head_is_permutation_equivariant() -> None:
    """SB3's dense ``action_net`` must have been replaced by the shared decoder."""
    env = DynamicRoutingEnv(seed=0)
    config = TrainConfig(num_envs=2, n_steps=32, batch_size=64, total_timesteps=64)
    venv = build_vec_env(config, env.config, env.reward_config)
    agent = build_agent(venv, config, env)
    try:
        assert isinstance(agent.policy.action_net, torch.nn.Identity)
        assert agent.policy.mlp_extractor.latent_dim_pi == env.action_space.shape[0]
        assert agent.policy.mlp_extractor.actor_head.weight.shape[0] == 1
        optimised = {id(p) for group in agent.policy.optimizer.param_groups
                     for p in group["params"]}
        missing = [n for n, p in agent.policy.named_parameters() if id(p) not in optimised]
        assert not missing, f"parameters excluded from the optimizer: {missing}"
        agent.learn(total_timesteps=64)
    finally:
        venv.close()


def test_backpressure_beats_shortest_path_under_load() -> None:
    """Sanity floor: the classical policy must actually help, or the sim is trivial."""
    env = DynamicRoutingEnv(seed=0)
    naive = evaluate_policy(ShortestPathPolicy(env), env, episodes=4, seed=99)
    smart = evaluate_policy(BackpressurePolicy(env), env, episodes=4, seed=99)
    assert smart["served_fraction"] > naive["served_fraction"]
    assert smart["mean_peak_queue"] < naive["mean_peak_queue"]


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as error:
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
        else:
            print(f"pass {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

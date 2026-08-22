"""Classical control baselines and the shared evaluation harness.

Without these the training curve is unfalsifiable: a rising episode return says
nothing about whether the agent beat the routing rule you would have shipped
anyway.  ``BackpressurePolicy`` is the strong one -- max-weight scheduling is
throughput-optimal for this class of queueing network -- so the interesting
question is whether PPO beats it by exploiting the demand clock it can see and
backpressure cannot.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .env import DynamicRoutingEnv

__all__ = [
    "Policy",
    "ShortestPathPolicy",
    "BackpressurePolicy",
    "evaluate_policy",
    "compare_policies",
]

_EDGE_CAPACITY = 0
_EDGE_BACKPRESSURE = 4


class Policy(Protocol):
    """The subset of the SB3 predict API the evaluator relies on."""

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Return ``(action, state)`` for the given observation."""
        ...


class _EdgeFeaturePolicy:
    """Base for hand-written policies that only need the edge feature block."""

    def __init__(self, env: DynamicRoutingEnv) -> None:
        self.spec = env.spec_graph
        self._offset = self.spec.num_nodes * self.spec.node_dim

    def _edge_features(self, observation: np.ndarray) -> np.ndarray:
        batch = np.atleast_2d(observation)
        return batch[:, self._offset :].reshape(
            batch.shape[0], self.spec.num_edges, self.spec.edge_dim
        )

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Return ``(action, state)`` for a single observation or a batch of them."""
        action = self._act(self._edge_features(observation))
        if np.ndim(observation) == 1:
            action = action[0]
        return action.astype(np.float32), None

    def _act(self, edges: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class ShortestPathPolicy(_EdgeFeaturePolicy):
    """Zero action: the environment's built-in shortest-path routing prior."""

    def _act(self, edges: np.ndarray) -> np.ndarray:
        return np.zeros((edges.shape[0], self.spec.num_edges))


class BackpressurePolicy(_EdgeFeaturePolicy):
    """Max-weight routing: push along edges with high differential backlog.

    Weighting the backlog difference by live capacity is the standard max-weight
    form ``(q_i - q_j) * c_ij``; here it lands on a routing logit rather than a
    scheduling decision, so ``gain`` sets how sharply the softmax concentrates.
    """

    def __init__(self, env: DynamicRoutingEnv, gain: float = 4.0) -> None:
        super().__init__(env)
        self.gain = gain

    def _act(self, edges: np.ndarray) -> np.ndarray:
        pressure = edges[:, :, _EDGE_BACKPRESSURE]
        capacity = edges[:, :, _EDGE_CAPACITY]
        return np.clip(self.gain * pressure * capacity, -1.0, 1.0)


def evaluate_policy(
    policy: Policy,
    env: DynamicRoutingEnv,
    episodes: int = 10,
    seed: int = 12345,
) -> dict[str, float]:
    """Roll out ``policy`` on fixed seeds and summarise the routing outcome.

    Seeds are shared across policies so the comparison is paired: every policy
    faces the identical demand realisation and incident sequence.
    """
    returns, served, dropped, peaks, means, churn = [], [], [], [], [], []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = total_offered = total_served = total_dropped = 0.0
        episode_peaks, episode_means = [], []
        previous = np.zeros(env.action_space.shape, dtype=np.float64)
        episode_churn = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = policy.predict(observation, deterministic=True)
            action = np.asarray(action, dtype=np.float64).reshape(env.action_space.shape)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_offered += info["offered"]
            total_served += info["throughput"]
            total_dropped += info["dropped"]
            episode_peaks.append(info["max_queue"])
            episode_means.append(info["mean_queue"])
            episode_churn += float(np.mean((action - previous) ** 2))
            previous = action
            steps += 1
            done = terminated or truncated
        returns.append(total_reward)
        served.append(total_served / max(total_offered, 1e-8))
        dropped.append(total_dropped / max(total_offered, 1e-8))
        peaks.append(float(np.mean(episode_peaks)))
        means.append(float(np.mean(episode_means)))
        churn.append(episode_churn / max(steps, 1))

    return {
        "return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "served_fraction": float(np.mean(served)),
        "dropped_fraction": float(np.mean(dropped)),
        "mean_peak_queue": float(np.mean(peaks)),
        "mean_queue": float(np.mean(means)),
        "action_churn": float(np.mean(churn)),
    }


def compare_policies(
    policies: dict[str, Policy],
    env: DynamicRoutingEnv,
    episodes: int = 10,
    seed: int = 12345,
) -> str:
    """Evaluate every policy on the same seeds and format a comparison table."""
    rows = {
        name: evaluate_policy(policy, env, episodes=episodes, seed=seed)
        for name, policy in policies.items()
    }
    columns = [
        ("return", "return", "{:>9.2f}"),
        ("served", "served_fraction", "{:>9.3f}"),
        ("dropped", "dropped_fraction", "{:>9.3f}"),
        ("peak_q", "mean_peak_queue", "{:>9.3f}"),
        ("mean_q", "mean_queue", "{:>9.3f}"),
        ("churn", "action_churn", "{:>9.4f}"),
    ]
    width = max(len(name) for name in rows) + 2
    header = "policy".ljust(width) + "".join(f"{label:>10}" for label, _, _ in columns)
    lines = [header, "-" * len(header)]
    for name, metrics in rows.items():
        cells = "".join(fmt.format(metrics[key]) + " " for _, key, fmt in columns)
        lines.append(name.ljust(width) + cells)
    return "\n".join(lines)

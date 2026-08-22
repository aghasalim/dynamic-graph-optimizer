"""Gymnasium environment wrapping the queueing simulator.

The observation is a flat vector that packs a node-feature block and an
edge-feature block back to back; the topology is static, so the graph structure
itself lives in the model rather than in the observation.  :class:`GraphSpec`
tells the feature extractor how to unpack it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .simulator import NetworkConfig, NetworkSimulator

__all__ = ["RewardConfig", "GraphSpec", "DynamicRoutingEnv"]

NODE_FEATURES = 8
EDGE_FEATURES = 5
OBS_CLIP = 10.0


@dataclass(frozen=True)
class RewardConfig:
    """Weights and shape of the reward.  See ``README.md`` for the derivation."""

    throughput_weight: float = 1.0
    drop_weight: float = 0.5
    bottleneck_weight: float = 1.0
    smoothness_weight: float = 0.05
    bottleneck_temperature: float = 0.25
    #: ``"shaping"`` uses the potential-based form ``gamma * Phi' - Phi``, which is
    #: policy-invariant.  ``"absolute"`` penalises the level of congestion and does
    #: move the optimum, trading throughput for a flatter backlog profile.
    bottleneck_mode: str = "shaping"
    gamma: float = 0.99

    def __post_init__(self) -> None:
        """Reject reward settings that would silently produce a nonsense objective."""
        if self.bottleneck_mode not in ("shaping", "absolute"):
            raise ValueError("bottleneck_mode must be 'shaping' or 'absolute'")
        if self.bottleneck_temperature <= 0.0:
            raise ValueError("bottleneck_temperature must be positive")


@dataclass(frozen=True)
class GraphSpec:
    """Static shape metadata the GNN needs to rebuild a graph from a flat vector."""

    num_nodes: int
    num_edges: int
    node_dim: int
    edge_dim: int
    edge_index: np.ndarray

    @property
    def obs_dim(self) -> int:
        """Length of the flat observation vector."""
        return self.num_nodes * self.node_dim + self.num_edges * self.edge_dim


def smooth_max(values: np.ndarray, temperature: float) -> float:
    """Log-sum-exp bottleneck surrogate.

    Interpolates between ``mean(values)`` as ``temperature -> inf`` and
    ``max(values)`` as ``temperature -> 0``, and is differentiable everywhere, so
    every node receives gradient in proportion to how nearly it is the bottleneck.
    """
    scaled = values / temperature
    peak = float(np.max(scaled))
    return float(temperature * (peak + np.log(np.mean(np.exp(scaled - peak)))))


class DynamicRoutingEnv(gym.Env):
    """Route traffic across a spatial grid whose demand and capacity keep moving.

    Action: one routing-logit offset per directed edge, in ``[-1, 1]``.  A zero
    action reproduces shortest-path routing, so the agent is learning a correction
    to a working default.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: NetworkConfig | None = None,
        reward: RewardConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.reward_config = reward or RewardConfig()
        self.sim = NetworkSimulator(self.config, np.random.default_rng(seed))

        net = self.sim.net
        self.spec_graph = GraphSpec(
            num_nodes=net.num_nodes,
            num_edges=net.num_edges,
            node_dim=NODE_FEATURES,
            edge_dim=EDGE_FEATURES,
            edge_index=np.stack([net.src, net.dst]).astype(np.int64),
        )
        self.action_space = spaces.Box(-1.0, 1.0, (net.num_edges,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -OBS_CLIP, OBS_CLIP, (self.spec_graph.obs_dim,), dtype=np.float32
        )

        self._reference_throughput = float(self.config.base_demand * net.source_ids.size)
        self._hop_scale = max(float(np.max(net.hops_to_sink)), 1.0)
        self._prev_action = np.zeros(net.num_edges)
        self._prev_potential = 0.0

    # -- observation -------------------------------------------------------

    def _node_features(self) -> np.ndarray:
        sim, net, cfg = self.sim, self.sim.net, self.config
        scale = max(cfg.edge_capacity, 1e-8)
        phase = 2.0 * np.pi * sim.t / cfg.demand_period
        return np.stack(
            [
                sim.normalised_queues,
                sim.last_inflow / scale,
                sim.last_outflow / scale,
                net.is_source.astype(np.float64),
                net.is_sink.astype(np.float64),
                net.hops_to_sink / self._hop_scale,
                np.full(net.num_nodes, np.sin(phase)),
                np.full(net.num_nodes, np.cos(phase)),
            ],
            axis=1,
        )

    def _edge_features(self) -> np.ndarray:
        sim, net, cfg = self.sim, self.sim.net, self.config
        scale = max(cfg.edge_capacity, 1e-8)
        queues = sim.normalised_queues
        return np.stack(
            [
                sim.edge_capacity * sim.incident_mult / scale,
                sim.last_flow / scale,
                sim.utilisation,
                self._prev_action,
                queues[net.src] - queues[net.dst],
            ],
            axis=1,
        )

    def _observation(self) -> np.ndarray:
        flat = np.concatenate(
            [self._node_features().ravel(), self._edge_features().ravel()]
        )
        return np.clip(flat, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _potential(self) -> float:
        return smooth_max(
            self.sim.normalised_queues, self.reward_config.bottleneck_temperature
        )

    # -- gymnasium API -----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode and return the first observation."""
        super().reset(seed=seed)
        self.sim.reset(np.random.default_rng(seed) if seed is not None else None)
        self._prev_action = np.zeros(self.sim.net.num_edges)
        self._prev_potential = self._potential()
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one routing-logit adjustment and advance the network by one tick."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        measurements = self.sim.step(action)

        rc = self.reward_config
        potential = self._potential()
        if rc.bottleneck_mode == "shaping":
            # Ng-Harada-Russell potential-based shaping with Psi = -Phi: dense
            # credit for congestion without moving the optimal policy.
            congestion = rc.gamma * potential - self._prev_potential
        else:
            congestion = potential

        churn = float(np.mean((action - self._prev_action) ** 2))
        reward = (
            rc.throughput_weight * measurements.throughput / self._reference_throughput
            - rc.drop_weight * measurements.dropped / self._reference_throughput
            - rc.bottleneck_weight * congestion
            - rc.smoothness_weight * churn
        )

        self._prev_action = action
        self._prev_potential = potential
        truncated = self.sim.t >= self.config.horizon
        info = self._info(measurements)
        return self._observation(), float(reward), False, truncated, info

    def _info(self, measurements) -> dict[str, Any]:
        queues = self.sim.normalised_queues
        return {
            "throughput": measurements.throughput,
            "dropped": measurements.dropped,
            "offered": measurements.offered,
            "served_ratio": measurements.throughput / max(measurements.offered, 1e-8),
            "mean_queue": float(np.mean(queues)),
            "max_queue": float(np.max(queues)),
            "bottleneck": self._potential(),
            "mean_utilisation": float(np.mean(self.sim.utilisation)),
        }

    def snapshot(self) -> dict[str, np.ndarray]:
        """Copy of the renderable state, for the visualiser."""
        return {
            "queues": self.sim.normalised_queues.copy(),
            "flow": self.sim.last_flow.copy(),
            "utilisation": self.sim.utilisation.copy(),
            "incident": (self.sim.incident_mult < 1.0).copy(),
            "action": self._prev_action.copy(),
            "t": self.sim.t,
        }

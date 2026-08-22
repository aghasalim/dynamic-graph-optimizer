"""Discrete-time queueing simulator for a spatial routing network.

Nodes are intersections (or routers) holding a backlog ``q_i``; directed edges are
roads (or links) with a finite per-step capacity that incidents can degrade.  The
controller does not move packets directly: it shifts the *routing logits* of each
outgoing edge, and the split of a node's backlog across its out-edges is the
softmax of those logits.  With a zero action the network falls back to
shortest-path routing, which is the classical policy that congests under load.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

__all__ = ["NetworkConfig", "GridNetwork", "NetworkSimulator"]


@dataclass(frozen=True)
class NetworkConfig:
    """Physical and stochastic parameters of the simulated network."""

    rows: int = 4
    cols: int = 5
    queue_capacity: float = 40.0
    edge_capacity: float = 8.0
    capacity_jitter: float = 0.35
    base_demand: float = 7.0
    demand_period: int = 120
    demand_amplitude: float = 0.8
    demand_noise: float = 0.15
    incident_rate: float = 0.03
    incident_severity: float = 0.15
    incident_duration: tuple[int, int] = (8, 25)
    shortest_path_bias: float = 2.5
    action_gain: float = 3.0
    horizon: int = 300

    def __post_init__(self) -> None:
        if self.rows < 2 or self.cols < 3:
            raise ValueError("grid must be at least 2 x 3 for a left-to-right flow field")
        if not 0.0 < self.incident_severity < 1.0:
            raise ValueError("incident_severity is a capacity multiplier in (0, 1)")


class GridNetwork:
    """Static topology: a 4-connected grid with source and sink columns.

    Edges are stored grouped by source node so that the per-node routing softmax
    is a single segmented reduction rather than a Python loop over nodes.
    """

    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        rows, cols = config.rows, config.cols
        self.num_nodes = rows * cols

        self.positions = np.array(
            [(c, rows - 1 - r) for r in range(rows) for c in range(cols)], dtype=np.float64
        )

        src: list[int] = []
        dst: list[int] = []
        out_start = np.zeros(self.num_nodes, dtype=np.int64)
        out_count = np.zeros(self.num_nodes, dtype=np.int64)
        for r in range(rows):
            for c in range(cols):
                node = r * cols + c
                out_start[node] = len(src)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        src.append(node)
                        dst.append(nr * cols + nc)
                out_count[node] = len(src) - out_start[node]

        self.src = np.asarray(src, dtype=np.int64)
        self.dst = np.asarray(dst, dtype=np.int64)
        self.out_start = out_start
        self.out_count = out_count
        self.num_edges = self.src.size
        if np.any(out_count == 0):
            raise ValueError("segmented softmax requires every node to have an out-edge")

        self.is_source = np.zeros(self.num_nodes, dtype=bool)
        self.is_sink = np.zeros(self.num_nodes, dtype=bool)
        self.is_source[np.arange(rows) * cols] = True
        self.is_sink[np.arange(rows) * cols + (cols - 1)] = True

        self.source_ids = np.flatnonzero(self.is_source)
        self.sink_ids = np.flatnonzero(self.is_sink)
        self.src_is_sink = self.is_sink[self.src]
        self.reverse_edge = self._build_reverse_index()
        self.hops_to_sink = self._bfs_from_sinks()

    def _build_reverse_index(self) -> np.ndarray:
        """Map each directed edge to its opposite direction (roads are two-way)."""
        pairs = zip(self.src, self.dst, strict=True)
        lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(pairs)}
        return np.asarray(
            [lookup[(int(b), int(a))] for a, b in zip(self.src, self.dst, strict=True)],
            dtype=np.int64,
        )

    def _bfs_from_sinks(self) -> np.ndarray:
        """Hop distance to the nearest sink; the shortest-path routing prior."""
        dist = np.full(self.num_nodes, np.inf)
        queue: deque[int] = deque()
        for sink in self.sink_ids:
            dist[sink] = 0.0
            queue.append(int(sink))
        neighbours: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for a, b in zip(self.src, self.dst, strict=True):
            neighbours[int(a)].append(int(b))
        while queue:
            node = queue.popleft()
            for nxt in neighbours[node]:
                if dist[nxt] > dist[node] + 1:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        return dist

    def to_networkx(self):
        """Build the matching :mod:`networkx` DiGraph (visualisation only)."""
        import networkx as nx

        graph = nx.DiGraph()
        for node in range(self.num_nodes):
            graph.add_node(node, pos=tuple(self.positions[node]))
        for edge, (a, b) in enumerate(zip(self.src, self.dst, strict=True)):
            graph.add_edge(int(a), int(b), index=edge)
        return graph


@dataclass
class StepMeasurements:
    """Everything one simulator step exposes to the environment and the logger."""

    throughput: float
    dropped: float
    offered: float
    flow: np.ndarray
    node_inflow: np.ndarray
    node_outflow: np.ndarray
    split: np.ndarray = field(repr=False)


class NetworkSimulator:
    """Stateful queueing dynamics with time-varying demand and random incidents."""

    def __init__(
        self, config: NetworkConfig, rng: np.random.Generator | None = None
    ) -> None:
        self.config = config
        self.net = GridNetwork(config)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Start a fresh episode; edge capacities are resampled per episode."""
        if rng is not None:
            self.rng = rng
        cfg, net = self.config, self.net
        self.t = 0
        self.queues = np.zeros(net.num_nodes)
        self.edge_capacity = cfg.edge_capacity * (
            1.0 + cfg.capacity_jitter * self.rng.uniform(-1.0, 1.0, net.num_edges)
        )
        self.incident_mult = np.ones(net.num_edges)
        self.incident_timer = np.zeros(net.num_edges, dtype=np.int64)
        self.last_flow = np.zeros(net.num_edges)
        self.last_inflow = np.zeros(net.num_nodes)
        self.last_outflow = np.zeros(net.num_nodes)
        self.source_phase = self.rng.uniform(0.0, 2.0 * np.pi, net.source_ids.size)

    # -- per-step pieces ---------------------------------------------------

    def _advance_incidents(self) -> None:
        active = self.incident_timer > 0
        self.incident_timer[active] -= 1
        cleared = active & (self.incident_timer == 0)
        self.incident_mult[cleared] = 1.0
        if self.rng.random() < self.config.incident_rate:
            edge = int(self.rng.integers(self.net.num_edges))
            pair = np.array([edge, self.net.reverse_edge[edge]])
            duration = int(self.rng.integers(*self.config.incident_duration))
            self.incident_mult[pair] = self.config.incident_severity
            self.incident_timer[pair] = duration

    def demand(self) -> np.ndarray:
        """Offered load at each source: a per-source rush-hour cycle plus noise."""
        cfg = self.config
        phase = 2.0 * np.pi * self.t / cfg.demand_period + self.source_phase
        seasonal = 1.0 + cfg.demand_amplitude * np.sin(phase)
        noise = 1.0 + cfg.demand_noise * self.rng.standard_normal(self.source_phase.size)
        return np.maximum(cfg.base_demand * seasonal * noise, 0.0)

    def _inject(self, offered: np.ndarray) -> float:
        """Add offered load at the sources, dropping whatever exceeds capacity."""
        ids = self.net.source_ids
        room = self.config.queue_capacity - self.queues[ids]
        accepted = np.minimum(offered, np.maximum(room, 0.0))
        self.queues[ids] += accepted
        return float(np.sum(offered - accepted))

    def routing_split(self, action: np.ndarray) -> np.ndarray:
        """Segmented softmax over each node's out-edges.

        ``logit_e = -bias * hops(dst_e) + gain * action_e`` — a zero action leaves
        pure shortest-path routing, so the agent learns a *correction* to a sane
        default rather than a routing policy from scratch.
        """
        cfg, net = self.config, self.net
        prior = -cfg.shortest_path_bias * net.hops_to_sink[net.dst]
        logits = prior + cfg.action_gain * action
        starts, counts = net.out_start, net.out_count
        shifted = logits - np.repeat(np.maximum.reduceat(logits, starts), counts)
        weights = np.exp(shifted)
        return weights / np.repeat(np.add.reduceat(weights, starts), counts)

    def _apply_spillback(self, flow: np.ndarray) -> np.ndarray:
        """Ration inflow proportionally when a downstream queue has no headroom.

        Single-pass rationing: a node scaled down here may leave its own upstream
        slightly over-served within the same step.  The residual is bounded by one
        step of capacity and self-corrects on the next tick.
        """
        net = self.net
        arriving = np.bincount(net.dst, weights=flow, minlength=net.num_nodes)
        headroom = np.where(net.is_sink, np.inf, self.config.queue_capacity - self.queues)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(arriving > 0.0, np.minimum(1.0, headroom / arriving), 1.0)
        return flow * scale[net.dst]

    # -- main entry point --------------------------------------------------

    def step(self, action: np.ndarray) -> StepMeasurements:
        """Advance one tick under ``action`` and return the step's measurements."""
        net = self.net
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        self._advance_incidents()
        offered = self.demand()
        dropped = self._inject(offered)

        split = self.routing_split(action)
        capacity = self.edge_capacity * self.incident_mult
        flow = np.minimum(self.queues[net.src] * split, capacity)
        flow[net.src_is_sink] = 0.0
        flow = self._apply_spillback(flow)

        outflow = np.bincount(net.src, weights=flow, minlength=net.num_nodes)
        inflow = np.bincount(net.dst, weights=flow, minlength=net.num_nodes)
        throughput = float(np.sum(inflow[net.sink_ids]))

        self.queues = np.clip(
            self.queues - outflow + inflow, 0.0, self.config.queue_capacity
        )
        self.queues[net.sink_ids] = 0.0

        self.last_flow, self.last_inflow, self.last_outflow = flow, inflow, outflow
        self.t += 1
        return StepMeasurements(
            throughput=throughput,
            dropped=dropped,
            offered=float(np.sum(offered)),
            flow=flow,
            node_inflow=inflow,
            node_outflow=outflow,
            split=split,
        )

    # -- derived quantities ------------------------------------------------

    @property
    def utilisation(self) -> np.ndarray:
        """Fraction of each edge's live capacity consumed last step."""
        capacity = np.maximum(self.edge_capacity * self.incident_mult, 1e-8)
        return np.clip(self.last_flow / capacity, 0.0, 1.0)

    @property
    def normalised_queues(self) -> np.ndarray:
        """Backlog scaled to ``[0, 1]`` by the per-node queue capacity."""
        return self.queues / self.config.queue_capacity

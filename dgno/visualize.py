"""NetworkX + Matplotlib animation of one episode.

Renders the grid with node colour as backlog and edge width as flow, plus a
time-series panel underneath so a congestion wave is visible both spatially and
temporally.  Run ``python -m dgno.visualize --policy backpressure``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .baselines import BackpressurePolicy, ShortestPathPolicy
from .env import DynamicRoutingEnv, RewardConfig
from .simulator import NetworkConfig

__all__ = [
    "EpisodeRecording",
    "record_episode",
    "animate_episode",
    "draw_network_state",
]

QUEUE_CMAP = plt.get_cmap("YlOrRd")
ACTION_CMAP = plt.get_cmap("coolwarm")


@dataclass
class EpisodeRecording:
    """Per-step snapshots plus the scalar series drawn in the lower panel."""

    frames: list[dict] = field(default_factory=list)
    throughput: list[float] = field(default_factory=list)
    max_queue: list[float] = field(default_factory=list)
    dropped: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of recorded steps."""
        return len(self.frames)


def record_episode(
    env: DynamicRoutingEnv, policy, seed: int = 0, max_steps: int | None = None
) -> EpisodeRecording:
    """Roll out one episode, capturing a renderable snapshot per step."""
    observation, _ = env.reset(seed=seed)
    recording = EpisodeRecording()
    limit = max_steps or env.config.horizon
    for _ in range(limit):
        action, _ = policy.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(env.action_space.shape)
        observation, _, terminated, truncated, info = env.step(action)
        recording.frames.append(env.snapshot())
        recording.throughput.append(info["throughput"])
        recording.max_queue.append(info["max_queue"])
        recording.dropped.append(info["dropped"])
        if terminated or truncated:
            break
    return recording


def draw_network_state(
    ax,
    network,
    frame: dict,
    capacity_scale: float,
    title: str = "",
    labels: bool = True,
    graph=None,
    positions=None,
    edge_list=None,
) -> None:
    """Render one network snapshot onto ``ax``.

    Shared by the animator and the README figure script so both stay in step;
    ``graph``/``positions``/``edge_list`` can be passed in when the caller already
    built them, which the animator does once per episode rather than per frame.
    """
    if graph is None:
        graph = network.to_networkx()
    if positions is None:
        positions = {n: tuple(network.positions[n]) for n in graph.nodes}
    if edge_list is None:
        pairs = zip(network.src, network.dst, strict=True)
        edge_list = [(int(a), int(b)) for a, b in pairs]

    ax.clear()
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=edge_list,
        width=0.6 + 4.5 * frame["flow"] / capacity_scale,
        edge_color=[ACTION_CMAP(0.5 * (a + 1.0)) for a in frame["action"]],
        connectionstyle="arc3,rad=0.13",
        arrowsize=9,
        alpha=0.85,
        ax=ax,
    )
    flags = zip(edge_list, frame["incident"], strict=True)
    incident = [edge for edge, flag in flags if flag]
    if incident:
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=incident,
            width=2.4,
            edge_color="black",
            style="dashed",
            connectionstyle="arc3,rad=0.13",
            arrowsize=1,
            ax=ax,
        )

    borders = np.where(network.is_source, "#1a9850", "0.25")
    borders = np.where(network.is_sink, "#2166ac", borders)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[QUEUE_CMAP(q) for q in frame["queues"]],
        node_size=620 if labels else 260,
        edgecolors=list(borders),
        linewidths=np.where(network.is_source | network.is_sink, 2.6, 0.8),
        ax=ax,
    )
    if labels:
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={n: f"{frame['queues'][n]:.2f}" for n in graph.nodes},
            font_size=7,
            ax=ax,
        )
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    ax.margins(0.08)


class EpisodeAnimator:
    """Draws one :class:`EpisodeRecording` onto a two-panel figure."""

    def __init__(self, env: DynamicRoutingEnv, recording: EpisodeRecording) -> None:
        self.env = env
        self.recording = recording
        self.net = env.sim.net
        self.graph = self.net.to_networkx()
        self.positions = {n: tuple(self.net.positions[n]) for n in self.graph.nodes}
        self.edge_list = [
            (int(a), int(b)) for a, b in zip(self.net.src, self.net.dst, strict=True)
        ]
        self.capacity_scale = max(float(env.config.edge_capacity), 1e-8)

        self.figure, (self.ax_graph, self.ax_series) = plt.subplots(
            2, 1, figsize=(9.5, 8.5), height_ratios=[3, 1]
        )
        self.figure.subplots_adjust(hspace=0.22)
        self._setup_series_axis()

    def _setup_series_axis(self) -> None:
        steps = np.arange(len(self.recording))
        self.ax_series.plot(
            steps, self.recording.throughput, lw=1.2, color="#2166ac", label="throughput"
        )
        peak = max(max(self.recording.throughput, default=1.0), 1.0)
        self.ax_series.plot(
            steps,
            np.asarray(self.recording.max_queue) * peak,
            lw=1.2,
            color="#b2182b",
            label="peak backlog (scaled)",
        )
        self.ax_series.set_xlim(0, max(len(self.recording) - 1, 1))
        self.ax_series.set_xlabel("timestep")
        self.ax_series.legend(loc="lower right", fontsize=8, ncol=2, frameon=False)
        self.ax_series.spines[["top", "right"]].set_visible(False)
        self.cursor = self.ax_series.axvline(0, color="0.3", lw=1.0, ls="--")

    def _draw_frame(self, index: int) -> None:
        # ponytail: full redraw per frame -- ~300 frames renders in seconds.
        # Swap for LineCollection offset updates only if frame counts grow.
        frame = self.recording.frames[index]
        served = self.recording.throughput[index]
        draw_network_state(
            self.ax_graph,
            self.net,
            frame,
            self.capacity_scale,
            title=(
                f"t={frame['t']:3d}   throughput={served:6.2f}   "
                f"peak backlog={frame['queues'].max():.2f}   "
                f"incidents={int(np.sum(frame['incident']) // 2)}"
            ),
            graph=self.graph,
            positions=self.positions,
            edge_list=self.edge_list,
        )
        self.cursor.set_xdata([index, index])

    def animate(self, output: Path, fps: int = 12) -> Path:
        """Write the episode to an animated GIF."""
        animation = FuncAnimation(
            self.figure,
            self._draw_frame,
            frames=len(self.recording),
            interval=1000 // fps,
            blit=False,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(output), writer=PillowWriter(fps=fps))
        plt.close(self.figure)
        return output


def animate_episode(
    env: DynamicRoutingEnv,
    policy,
    output: str | Path = "renders/episode.gif",
    seed: int = 0,
    max_steps: int | None = None,
    fps: int = 12,
) -> Path:
    """Record an episode under ``policy`` and write the animation."""
    recording = record_episode(env, policy, seed=seed, max_steps=max_steps)
    return EpisodeAnimator(env, recording).animate(Path(output), fps=fps)


def _load_policy(name: str, env: DynamicRoutingEnv, model_path: str | None):
    if name == "shortest-path":
        return ShortestPathPolicy(env)
    if name == "backpressure":
        return BackpressurePolicy(env)
    if not model_path:
        raise SystemExit("--policy ppo requires --model pointing at a saved agent")
    from stable_baselines3 import PPO

    return PPO.load(model_path, device="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", choices=("shortest-path", "backpressure", "ppo"), default="backpressure"
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rows", type=int, default=NetworkConfig.rows)
    parser.add_argument("--cols", type=int, default=NetworkConfig.cols)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out", type=str, default="renders/episode.gif")
    args = parser.parse_args()

    env = DynamicRoutingEnv(
        config=NetworkConfig(rows=args.rows, cols=args.cols),
        reward=RewardConfig(),
        seed=args.seed,
    )
    policy = _load_policy(args.policy, env, args.model)
    path = animate_episode(
        env, policy, output=args.out, seed=args.seed, max_steps=args.steps, fps=args.fps
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

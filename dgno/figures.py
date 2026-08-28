"""Generate the figures used in the README.

``python -m dgno.figures`` writes everything into ``docs/``.  Kept separate from
``visualize`` because that module animates an episode, while this one produces
the static comparisons the write-up refers to.

Three of the figures and the animation read committed files only:
``docs/curve-long4M.csv``, ``docs/ablations.txt`` and ``docs/transfer.txt`` are
the measured output of runs that already happened, so a figure here cannot
disagree with a number quoted in the README.  The three that replay the
simulator sit behind ``--rollouts`` and are not redrawn by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from PIL import Image
from stable_baselines3 import PPO

from .baselines import BackpressurePolicy, ShortestPathPolicy
from .env import DynamicRoutingEnv
from .style import PALETTE, titled
from .visualize import ACTION_CMAP, QUEUE_CMAP, draw_network_state, record_episode

__all__ = ["learning_curve", "transfer_scaling", "ablation_table", "anim_learning",
           "network_state", "policy_comparison", "reward_anatomy"]

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "ppo-gnn-4m"
DOCS = Path(__file__).resolve().parents[1] / "docs"

# The README talks about these three policies by name in every table, so they
# keep one colour across every figure: grey for the naive default, green for the
# classical baseline, red for the learned policy.
SP, BP, PPO_COLOUR = PALETTE[5], PALETTE[2], PALETTE[1]
# In the ablation figure the split that matters is the training budget, not the
# reward variant, so the 400k runs share a colour and the 4M runs share another.
SHORT_BUDGET = PALETTE[0]

RETURN_AXIS = "episode return (reward units, 300-step episode)"
SERVED_AXIS = "served demand (fraction of offered)"
BACKLOG_AXIS = "mean peak backlog (fraction of node capacity)"

# 100 logged rows is about 0.2M steps at this rollout size.  Wide enough that the
# update-to-update noise stops dominating, narrow enough to still show the dip.
WINDOW = 100


def _curve(name: str = "curve-long4M.csv") -> tuple[np.ndarray, np.ndarray]:
    """Training steps in millions and rollout return, from a committed log."""
    steps, rewards = [], []
    with open(DOCS / name) as handle:
        for row in csv.DictReader(handle):
            if row.get("rollout/ep_rew_mean"):
                steps.append(float(row["time/total_timesteps"]) / 1e6)
                rewards.append(float(row["rollout/ep_rew_mean"]))
    return np.asarray(steps), np.asarray(rewards)


def _trailing(y: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Mean of the last ``window`` samples, expanding while there are fewer."""
    cumulative = np.cumsum(np.insert(y, 0, 0.0))
    hi = np.arange(len(y)) + 1
    lo = np.maximum(hi - window, 0)
    return (cumulative[hi] - cumulative[lo]) / (hi - lo)


def _ablation_rows() -> dict[str, dict[str, float]]:
    """Parse the committed ablation table into {policy: {metric: value}}.

    Policy names contain spaces and the six metric columns do not, so the split
    comes from the right.
    """
    columns = ["return", "served", "dropped", "peak_q", "mean_q", "churn"]
    rows: dict[str, dict[str, float]] = {}
    for line in (DOCS / "ablations.txt").read_text().splitlines():
        line = line.rstrip()
        parts = line.rsplit(None, len(columns))
        if len(parts) != len(columns) + 1:
            continue
        try:
            values = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        rows[parts[0]] = dict(zip(columns, values, strict=True))
    return rows


def _transfer_rows() -> list[dict[str, float]]:
    """Parse the committed transfer table, one dict per grid size."""
    keys = ["nodes", "edges", "served_sp", "served_bp", "served_ppo",
            "peak_sp", "peak_bp", "peak_ppo"]
    out = []
    for line in (DOCS / "transfer.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != len(keys) + 1 or "x" not in parts[0]:
            continue
        try:
            values = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        out.append(dict(zip(keys, values, strict=True)) | {"grid": parts[0]})
    return out


def _encoding_key(figure, y: float = 0.055, height: float = 0.028) -> None:
    """Two colour bars under the panels, saying what node and edge colour mean.

    The panels carry no numbers, so the scale has to live somewhere.  A caption
    strung across the bottom of the figure is the version that used to be here
    and it read as a caption, not as a key.
    """
    bars = [
        (0.15, QUEUE_CMAP, (0.0, 1.0),
         "node colour: backlog (fraction of node capacity)"),
        (0.56, ACTION_CMAP, (-1.0, 1.0),
         "edge colour: routing offset (edge width is flow)"),
    ]
    for x, cmap, limits, label in bars:
        cax = figure.add_axes((x, y, 0.29, height))
        cax.grid(False)
        bar = figure.colorbar(ScalarMappable(Normalize(*limits), cmap), cax=cax,
                              orientation="horizontal")
        bar.outline.set_visible(False)
        bar.set_label(label, fontsize=9, color="#5a5a5a")
        cax.tick_params(labelsize=8.5, length=3)


def _mark_columns(ax) -> None:
    """Name the source and sink columns in the colour their borders are drawn in."""
    ax.text(0.0, 1.0, "sources", transform=ax.transAxes, fontsize=9,
            color=PALETTE[2], ha="left", va="top")
    ax.text(1.0, 1.0, "sinks", transform=ax.transAxes, fontsize=9,
            color=PALETTE[0], ha="right", va="top")


def network_state(out: Path, step: int = 95, seed: int = 0) -> Path:
    """Draw the README hero: the trained agent mid-episode, above its own episode.

    Same checkpoint, seed and timestep as the third panel of
    :func:`policy_comparison`, so the two figures can be read against each other.
    The lower panel puts that one frame back in context: backlog is not drifting
    up, it is riding the demand cycle.
    """
    agent = PPO.load(str(CHECKPOINT), device="cpu")
    env = DynamicRoutingEnv(seed=seed)
    recording = record_episode(env, agent, seed=seed)
    frame = recording.frames[step]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12.5, 7.6),
                                         height_ratios=[2.7, 1.15])
    draw_network_state(top, env.sim.net, frame, env.config.edge_capacity, labels=True)
    titled(top,
           f"At t={step} no node is past {frame['queues'].max():.2f} of its capacity",
           f"PPO + GNN at 4M steps, seed {seed}; node label and colour are backlog, "
           "edge width is flow, edge colour is the routing offset")
    _mark_columns(top)

    timeline = np.arange(len(recording))
    bottom.plot(timeline, recording.throughput, lw=1.5, color=PALETTE[0])
    bottom.set_xlim(0, len(recording) - 1)
    bottom.set_xlabel("timestep")
    bottom.set_ylabel("throughput (units per step)", color=PALETTE[0])
    bottom.tick_params(axis="y", colors=PALETTE[0])

    backlog = bottom.twinx()
    backlog.plot(timeline, recording.max_queue, lw=1.5, color=PALETTE[1])
    backlog.set_ylabel("peak backlog (fraction of node capacity)", color=PALETTE[1])
    backlog.tick_params(axis="y", colors=PALETTE[1])
    backlog.spines[["top", "left"]].set_visible(False)
    backlog.spines["right"].set_visible(True)
    backlog.grid(False)

    bottom.axvline(step, lw=1.1, ls="--", color="#999999", zorder=1)
    bottom.text(step + 4, 0.03, " the frame above", fontsize=9, color="#777777",
                transform=bottom.get_xaxis_transform(), ha="left", va="bottom")
    titled(bottom, "Backlog rides the demand cycle rather than drifting up",
           f"the whole {len(recording)}-step episode, two series on their own axis; "
           "the sources share a 120-step rush-hour period")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def policy_comparison(out: Path, step: int = 95, seed: int = 0) -> Path:
    """Draw three policies at the same timestep of the same episode, side by side."""
    agent = PPO.load(str(CHECKPOINT), device="cpu")
    env = DynamicRoutingEnv(seed=seed)
    panels = [
        ("shortest-path", ShortestPathPolicy(env)),
        ("backpressure", BackpressurePolicy(env)),
        ("PPO + GNN, 4M steps", agent),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    for ax, (label, policy) in zip(axes, panels, strict=True):
        recording = record_episode(env, policy, seed=seed, max_steps=step + 1)
        frame = recording.frames[step]
        draw_network_state(ax, env.sim.net, frame, env.config.edge_capacity,
                           labels=False)
        titled(ax, label,
               f"t={step} on seed {seed}, throughput {recording.throughput[step]:.1f}, "
               f"peak backlog {frame['queues'].max():.2f}")
    _mark_columns(axes[0])

    figure.tight_layout(rect=(0, 0.16, 1, 1))
    _encoding_key(figure, y=0.07)
    figure.savefig(out)
    plt.close(figure)
    return out


def learning_curve(out: Path) -> Path:
    """Rollout return over training, next to what the same policy scores greedily.

    Two separate points: the rollout curve rises steadily, and it sits well below
    what the deterministic policy actually achieves, because rollouts are collected
    under a Gaussian with std ~0.35 and the exploration noise costs reward.
    """
    x, y = _curve()
    scores = _ablation_rows()
    greedy = scores["ppo 4M"]["return"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    left.plot(x, y, lw=0.7, color=SHORT_BUDGET, alpha=0.35,
              label="rollout return, one point per PPO update")
    left.plot(x, _trailing(y), lw=2.2, color=PPO_COLOUR,
              label=f"trailing mean over {WINDOW} updates")
    settled = x > 0.2
    fit = np.poly1d(np.polyfit(x[settled], y[settled], 1))
    left.plot(x[settled], fit(x[settled]), lw=1.6, ls="--", color=PALETTE[4],
              label=f"trend past 0.2M, {fit.coefficients[0]:+.1f} return per M steps")
    left.axhline(greedy, ls="-", lw=1.4, color=BP,
                 label=f"same policy evaluated greedily, {greedy:.1f}")

    low = int(np.argmin(y))
    left.annotate(f"dips to {y[low]:.0f} by {x[low] * 1000:.0f}k steps",
                  (x[low], y[low]), xytext=(11, -4), textcoords="offset points",
                  fontsize=9, color="#5a5a5a", va="top", ha="left")

    left.set_xlabel("training steps (millions)")
    left.set_ylabel(RETURN_AXIS)
    left.set_ylim(y.min() - 8, greedy + 6)
    titled(left, "Exploration noise hides 30 points of return",
           "one 4M-step run, logged every PPO update, straight from the committed log")
    left.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))

    budget = {0.4: "ppo 400k (baseline)", 4.0: "ppo 4M"}
    xs = list(budget)
    ys = [scores[k]["served"] for k in budget.values()]
    right.plot(xs, ys, "o", markersize=10, color=PPO_COLOUR, zorder=3)
    for step_m, value in zip(xs, ys, strict=True):
        right.annotate(f"{value:.3f} at {step_m:g}M", (step_m, value), xytext=(0, 12),
                       textcoords="offset points", fontsize=9.5, color=PPO_COLOUR,
                       ha="center")

    for label, colour, style in (("backpressure", BP, "-"), ("shortest-path", SP, ":")):
        value = scores[label]["served"]
        right.axhline(value, ls=style, lw=1.3, color=colour, zorder=1)
        right.text(0.995, value, f" {label} {value:.3f} ", fontsize=9, color=colour,
                   va="bottom", ha="right", transform=right.get_yaxis_transform())

    right.set_xlim(-0.1, 4.6)
    right.set_ylim(0.855, 0.945)
    right.set_xlabel("training steps (millions)")
    right.set_ylabel(SERVED_AXIS)
    titled(right, "Ten times the budget carried it past backpressure",
           "the only two committed checkpoints, deterministic, 10 shared seeds")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def transfer_scaling(out: Path) -> Path:
    """Plot served fraction and peak backlog against grid size, from the committed table."""
    rows = _transfer_rows()
    x = [r["nodes"] for r in rows]
    trained_on = 20.0

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    series = [
        ("shortest-path", "sp", SP, ":"),
        ("backpressure", "bp", BP, "-"),
        ("PPO + GNN, trained on 4x5 only", "ppo", PPO_COLOUR, "-"),
    ]
    for label, key, colour, style in series:
        left.plot(x, [r[f"served_{key}"] for r in rows], style, color=colour,
                  marker="o", label=label)
        right.plot(x, [r[f"peak_{key}"] for r in rows], style, color=colour, marker="o")

    for ax in (left, right):
        ax.axvline(trained_on, lw=1.0, ls="--", color="#aaaaaa", zorder=1)
        ax.text(trained_on, 0.02, " trained here (4x5)", transform=ax.get_xaxis_transform(),
                fontsize=9, color="#888888", ha="left", va="bottom")
        ax.set_xlabel("nodes in the grid (count)")

    left.set_ylabel(SERVED_AXIS)
    titled(left, "The lead survives a grid four times the training size",
           "no retraining anywhere on this axis, 10 episodes per grid on shared seeds")
    right.set_ylabel(BACKLOG_AXIS)
    titled(right, "Backlog stays under both baselines at every size",
           "same runs, lower is better")

    handles, labels = left.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out)
    plt.close(figure)
    return out


def ablation_table(out: Path) -> Path:
    """Every committed checkpoint on the same seeds, from ``docs/ablations.txt``.

    Dots rather than bars: the interesting spread is 0.869 to 0.929 served, so a
    bar would either start at zero and show nothing or start at 0.85 and lie
    about the ratio between rows.
    """
    rows = _ablation_rows()
    order = sorted(rows, key=lambda k: rows[k]["served"], reverse=True)

    def colour_of(label: str) -> str:
        if label == "backpressure":
            return BP
        if label == "shortest-path":
            return SP
        return PPO_COLOUR if "4M" in label else SHORT_BUDGET

    colours = [colour_of(label) for label in order]
    positions = np.arange(len(order))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    panels = [
        (left, [rows[k]["served"] * 100 for k in order],
         rows["backpressure"]["served"] * 100, "served demand (% of offered)", "{:.1f}"),
        (right, [rows[k]["peak_q"] for k in order],
         rows["backpressure"]["peak_q"], BACKLOG_AXIS, "{:.3f}"),
    ]
    for ax, values, reference, xlabel, fmt in panels:
        ax.axvline(reference, color=BP, ls="--", lw=1.2, zorder=1)
        ax.scatter(values, positions, color=colours, s=70, zorder=3,
                   edgecolors="white", linewidths=0.8)
        pad = (max(values) - min(values)) * 0.30
        ax.set_xlim(min(values) - pad, max(values) + pad)
        for position, value in zip(positions, values, strict=True):
            ax.annotate(fmt.format(value), (value, position), xytext=(9, 0),
                        textcoords="offset points", fontsize=8.8, color="#5a5a5a",
                        va="center")
        ax.set_xlabel(xlabel)

    left.set_yticks(positions)
    left.set_yticklabels(order)
    left.invert_yaxis()
    titled(left, "Only ten times the budget cleared backpressure",
           "every committed checkpoint, one reward, 10 shared seeds; dashed line is "
           "backpressure")
    titled(right, "Peak backlog ranks the runs almost the same way",
           "mean over episodes of the worst node's backlog, lower is better")

    legend = [
        Line2D([], [], ls="", marker="o", color=PPO_COLOUR, label="PPO, 4M steps"),
        Line2D([], [], ls="", marker="o", color=SHORT_BUDGET, label="PPO, 400k steps"),
        Line2D([], [], ls="", marker="o", color=BP, label="backpressure"),
        Line2D([], [], ls="", marker="o", color=SP, label="shortest-path"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out)
    plt.close(figure)
    return out


def reward_anatomy(out: Path, steps: int = 300, seed: int = 0) -> Path:
    """Break an episode's return into what each reward term contributed.

    The congestion term is written as potential-based shaping, which telescopes.
    That is the property that makes it policy-invariant, and it is also why its
    cumulative contribution collapses to nearly nothing across an episode, the
    thing that cost me the bottleneck objective until I noticed.

    Dots rather than bars on the right: the terms span three orders of magnitude,
    so the axis has to be logarithmic, and a bar drawn against a log axis has a
    length that means nothing at all.
    """
    env = DynamicRoutingEnv(seed=seed)
    observation, _ = env.reset(seed=seed)
    policy = BackpressurePolicy(env)
    rc = env.reward_config
    ref = env._reference_throughput

    throughput, dropped, congestion, churn = [], [], [], []
    previous = np.zeros(env.action_space.shape)
    potential = env._potential()
    for _ in range(steps):
        action, _ = policy.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(env.action_space.shape)
        observation, _, terminated, truncated, info = env.step(action)
        new_potential = env._potential()
        throughput.append(rc.throughput_weight * info["throughput"] / ref)
        dropped.append(-rc.drop_weight * info["dropped"] / ref)
        congestion.append(-rc.bottleneck_weight *
                          (rc.gamma * new_potential - potential))
        churn.append(-rc.smoothness_weight * float(np.mean((action - previous) ** 2)))
        potential, previous = new_potential, action
        if terminated or truncated:
            break

    series = [
        ("throughput", np.cumsum(throughput), PALETTE[0]),
        ("dropped demand", np.cumsum(dropped), PALETTE[1]),
        ("congestion (shaping)", np.cumsum(congestion), PALETTE[4]),
        ("control churn", np.cumsum(churn), PALETTE[3]),
    ]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    for label, values, colour in series:
        left.plot(values, color=colour, label=label)
    left.axhline(0, color="#888888", lw=0.9, zorder=1)
    left.annotate("the other three terms, all inside 7 points of zero",
                  (len(series[0][1]) - 1, 0), xytext=(-6, 10),
                  textcoords="offset points", fontsize=9, color="#5a5a5a",
                  ha="right", va="bottom")
    left.set_xlabel("timestep")
    left.set_ylabel("cumulative contribution to return (reward units)")
    titled(left, "Throughput is essentially the whole return",
           f"one {len(throughput)}-step episode under backpressure, seed {seed}, "
           "each term weighted as the reward weights it")

    ranked = sorted(series, key=lambda s: abs(s[1][-1]), reverse=True)
    totals = [abs(values[-1]) for _, values, _ in ranked]
    positions = np.arange(len(ranked))
    right.scatter(totals, positions, color=[colour for _, _, colour in ranked], s=90,
                  zorder=3, edgecolors="white", linewidths=0.8)
    for position, value in zip(positions, totals, strict=True):
        right.annotate(f"{value:.2f}", (value, position), xytext=(10, 0),
                       textcoords="offset points", fontsize=9, color="#5a5a5a",
                       va="center")
    right.set_xscale("log")
    right.set_xlim(min(totals) * 0.45, max(totals) * 3.2)
    right.set_yticks(positions)
    right.set_yticklabels([label for label, _, _ in ranked])
    right.set_ylim(len(ranked) - 0.5, -0.5)
    right.set_xlabel("absolute total over the episode (reward units, log scale)")
    titled(right, "The shaping term cancels itself out",
           f"{max(totals) / min(totals):.0f} to 1 from top to bottom, so the axis "
           "is log and these are dots")

    handles, labels = left.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(out)
    plt.close(figure)
    return out


def _shrink_gif(path: Path) -> Path:
    """Requantise every frame onto one shared palette.  Usually halves the file."""
    src = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(src.convert("RGB"))
            durations.append(src.info.get("duration", 62))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(64, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    quantised[0].save(path, save_all=True, append_images=quantised[1:], loop=0,
                      duration=durations, optimize=True)
    return path


def anim_learning(out: Path, frames: int = 90, hold: int = 16, fps: int = 15) -> Path:
    """Replay the committed training log in step order.

    The README describes reading eight evenly spaced points off this log and
    concluding the run was flat.  Watching it fill in shows what that missed: the
    first logged row is a spike from before the policy learned anything, the
    curve falls under it, and the climb back takes most of the 4M steps.  Reads
    ``docs/curve-long4M.csv`` and nothing else, so the GIF is the same every run.
    """
    x, y = _curve()
    trail = _trailing(y)
    settled = x > 0.2
    slope = np.polyfit(x[settled], y[settled], 1)[0]
    low = int(np.argmin(y))
    first = y[0]

    figure, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(-0.05, x.max() + 0.05)
    ax.set_ylim(y.min() - 8, y.max() + 10)
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel(RETURN_AXIS)
    titled(ax, "The curve dips under its own first reading before it climbs",
           "one 4M-step run replayed in step order, straight from the committed log")

    ax.axhline(first, ls="--", lw=1.1, color="#999999", zorder=1,
               label=f"first logged row, {first:.0f} at 4k steps")

    raw = ax.plot([], [], lw=0.7, color=SHORT_BUDGET, alpha=0.35, zorder=2,
                  label="rollout return, one point per update")[0]
    mean = ax.plot([], [], lw=2.4, color=PPO_COLOUR, zorder=3,
                   label=f"trailing mean over {WINDOW} updates")[0]
    head = ax.plot([], [], "o", markersize=6.5, color=PPO_COLOUR, zorder=4)[0]
    ax.legend(loc="lower right")

    readout = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9.5,
                      color="#555555", ha="left", va="top")
    dip = ax.plot([], [], "v", markersize=8, color="#333333", zorder=5)[0]
    dip_text = ax.text(x[low] + 0.07, y[low] - 1.6, "", fontsize=8.8, color="#333333",
                       ha="left", va="center")

    cuts = np.linspace(x[0], x.max(), frames)

    def draw(i):
        cut = cuts[min(i, frames - 1)]
        k = max(int(np.searchsorted(x, cut, side="right")), 1)
        raw.set_data(x[:k], y[:k])
        mean.set_data(x[:k], trail[:k])
        head.set_data(x[k - 1:k], trail[k - 1:k])
        tail = f"\n{slope:+.1f} return per M steps, the whole way" if i >= frames else ""
        readout.set_text(f"{cut:.2f}M steps    trailing mean {trail[k - 1]:.0f}{tail}")
        if k > low:
            dip.set_data([x[low]], [y[low] - 0.8])
            dip_text.set_text(f"dips to {y[low]:.0f} by {x[low] * 1000:.0f}k steps")
        return [raw, mean, head, readout, dip, dip_text]

    animation = FuncAnimation(figure, draw, frames=frames + hold,
                              interval=1000 // fps, blit=False)
    animation.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return _shrink_gif(out)


def main() -> None:
    """Redraw the figures.  Only ``--rollouts`` touches the simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", action="store_true",
                        help="also redraw the two figures that replay the simulator")
    args = parser.parse_args()
    DOCS.mkdir(exist_ok=True)

    written = [
        learning_curve(DOCS / "learning-curve.png"),
        transfer_scaling(DOCS / "transfer-scaling.png"),
        ablation_table(DOCS / "ablations.png"),
        anim_learning(DOCS / "learning-curve.gif"),
    ]
    if args.rollouts:
        written += [
            network_state(DOCS / "network-state.png"),
            policy_comparison(DOCS / "policy-comparison.png"),
            reward_anatomy(DOCS / "reward-anatomy.png"),
        ]
    for path in written:
        print(f"wrote {path.name} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

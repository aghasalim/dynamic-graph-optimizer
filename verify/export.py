"""Write the raw fixture that verify/ reads, from the same code the README used.

Everything the repository publishes is a mean over ten evaluation episodes, and
the only record of those episodes was the printed mean itself.  This dumps two
things that were never written down before:

  data/episode-metrics.csv   the per episode metrics behind every row of
                             docs/ablations.txt, at full precision
  data/draws-*.csv           the random inputs each episode was built from,
                             so an implementation in another language can
                             replay the same network without numpy
  data/transfer-tensors.csv  which policy tensors survive a change of grid,
                             the count the README quotes twice

The draw files are what make an independent simulator possible.  numpy's
generator is not this repository's code, so reproducing its bit stream in C
would test numpy rather than the queueing dynamics.  Recording the draws and
replaying them tests the dynamics, which is the part that could be wrong.

Run: python verify/export.py           rewrite the fixture
     python verify/export.py --check   regenerate it and compare, changing nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dgno.ablations import build_policy_table  # noqa: E402
from dgno.env import DynamicRoutingEnv  # noqa: E402

EPISODES = 10
SEED = 12345
#: docs/reward-anatomy.png is drawn from one extra episode on its own seed, and
#: the README quotes two numbers off it. It is recorded as episode 10 here so the
#: C implementation can replay it too.
ANATOMY_EPISODE = 10
ANATOMY_SEED = 0
#: --check requires the recorded draws back byte identical: they are numpy on a
#: fixed seed and PCG64 is the same stream everywhere. The episode metrics are
#: not held to that. Every step has a min and a clip in it, so a difference in
#: the last bits of an exp or a sum can flip a branch and then grow over 300
#: steps. Regenerating on the CI runner rather than the laptop the fixture was
#: written on moves a single episode by at most 7.4e-04 relative, measured. The
#: tolerance is set an order of magnitude above that, and the means are then
#: required to round to the same printed table, which is the thing anything
#: downstream actually reads.
METRIC_TOLERANCE = 5e-3
#: the precision compare_policies() prints each column with, dgno/baselines.py
METRIC_DIGITS = (2, 3, 3, 3, 3, 4)
#: the largest grid in docs/transfer.txt, where the tensor census is taken
TRANSFER_ROWS, TRANSFER_COLS = 8, 10
OUT = Path(__file__).resolve().parent / "data"


class RecordingRNG:
    """Pass through to a real Generator, keeping every step level draw."""

    def __init__(self, inner, sim):
        self._inner = inner
        self._sim = sim
        self.incidents: list[tuple[int, int, int]] = []
        self.noise: list[tuple[int, np.ndarray]] = []
        self._pending_edge: int | None = None

    def random(self, *args, **kwargs):
        """Pass through; the incident coin flip needs no replay of its own."""
        return self._inner.random(*args, **kwargs)

    def integers(self, *args, **kwargs):
        """Pass through, pairing the edge and duration draws into one incident."""
        value = self._inner.integers(*args, **kwargs)
        if self._pending_edge is None:
            self._pending_edge = int(value)
        else:
            self.incidents.append((self._sim.t, self._pending_edge, int(value)))
            self._pending_edge = None
        return value

    def standard_normal(self, *args, **kwargs):
        """Pass through, keeping the per source demand noise for this step."""
        value = self._inner.standard_normal(*args, **kwargs)
        self.noise.append((self._sim.t, np.asarray(value).copy()))
        return value

    def uniform(self, *args, **kwargs):
        """Pass through; capacities and phases are read off the simulator after reset."""
        return self._inner.uniform(*args, **kwargs)


def g17(x: float) -> str:
    """Shortest text that reads back as the identical double."""
    return repr(float(x))


def emit(name: str, text: str, check: bool) -> None:
    """Write the file, or under --check compare it against what is committed.

    Exact everywhere except episode-metrics.csv, whose accumulated floats only
    have to agree to ``METRIC_TOLERANCE``.
    """
    path = OUT / name
    if not check:
        path.write_text(text)
        return

    committed = path.read_text()
    if name != "episode-metrics.csv":
        if committed == text:
            print(f"  ok   {name:<24} byte identical")
            return
        _fail(f"{name} differs from the committed fixture")

    # The metrics are compared numerically even when the bytes match, so the
    # check against the published table runs on every platform, not only the
    # ones where the export happens to be bit reproducible.
    old = committed.strip().split("\n")
    new = text.strip().split("\n")
    if len(old) != len(new):
        _fail(f"{name}: {len(new)} rows regenerated against {len(old)} committed")
    if old[0] != new[0]:
        _fail(f"{name}: the header changed")
    worst = 0.0
    totals: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for a, b in zip(old[1:], new[1:], strict=True):
        fa, fb = a.split(","), b.split(",")
        if fa[:2] != fb[:2]:
            _fail(f"{name}: row for {fb[:2]} does not line up with {fa[:2]}")
        for x, y in zip(fa[2:], fb[2:], strict=True):
            u, v = float(x), float(y)
            worst = max(worst, abs(u - v) / max(abs(u), 1.0))
        policy = fb[0]
        running = totals.setdefault(policy, [0.0] * len(METRIC_DIGITS))
        for i, value in enumerate(fb[2:]):
            running[i] += float(value)
        counts[policy] = counts.get(policy, 0) + 1
    if worst > METRIC_TOLERANCE:
        _fail(f"{name}: worst relative disagreement {worst:.1e}, tolerance "
              f"{METRIC_TOLERANCE:.0e}")

    # The published table is the means, so require those to round the same way
    # even though the episodes behind them are allowed to move.
    published = _published_table()
    for policy, running in totals.items():
        if policy not in published:
            _fail(f"docs/ablations.txt has no row for {policy}")
        for i, digits in enumerate(METRIC_DIGITS):
            got = f"{running[i] / counts[policy]:.{digits}f}"
            want = f"{published[policy][i]:.{digits}f}"
            if got != want:
                _fail(f"{policy}: regenerated column {i} is {got}, "
                      f"docs/ablations.txt says {want}")
    where = "byte identical" if committed == text else f"within {worst:.1e}"
    print(f"  ok   {name:<24} every metric {where}, and all "
          f"{len(totals)} rows still round to docs/ablations.txt")


def _published_table() -> dict[str, list[float]]:
    """docs/ablations.txt as label -> its six printed numbers."""
    rows: dict[str, list[float]] = {}
    for line in (ROOT / "docs" / "ablations.txt").read_text().split("\n"):
        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            values = [float(v) for v in fields[-6:]]
        except ValueError:
            continue
        rows[" ".join(fields[:-6])] = values
    return rows


def _fail(message: str) -> None:
    print(f"  FAIL {message}")
    raise SystemExit(1)


def transfer_census() -> tuple[list[str], list[str]]:
    """Re-host the 4M policy on the largest transfer grid and report the split.

    ``rehost_policy`` copies every tensor whose shape does not depend on the
    graph, so the count of what survives is the concrete measure of the
    equivariance claim. It is recorded here rather than asserted in the README.
    """
    from stable_baselines3 import PPO

    from dgno.env import RewardConfig
    from dgno.simulator import NetworkConfig
    from dgno.train import TrainConfig
    from dgno.transfer import rehost_policy

    config = TrainConfig()
    source = PPO.load(str(ROOT / "checkpoints" / "ppo-gnn-4m"), device="cpu")
    target = DynamicRoutingEnv(
        config=NetworkConfig(rows=TRANSFER_ROWS, cols=TRANSFER_COLS),
        reward=RewardConfig(gamma=config.gamma),
        seed=SEED,
    )
    policy = rehost_policy(source, target, config)
    copied, skipped = policy._transfer_report  # noqa: SLF001
    return list(copied), list(skipped)


def main() -> None:
    """Write every fixture file verify/ reads, or check the committed ones."""
    parser = argparse.ArgumentParser(description="write or check the verify fixture")
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate and compare against the committed files, writing nothing",
    )
    check = parser.parse_args().check
    OUT.mkdir(exist_ok=True)
    env = DynamicRoutingEnv(seed=0)
    policies = build_policy_table(env)

    metrics_rows = ["policy,episode,ret,served,dropped,peak_q,mean_q,churn"]
    capacity_rows = ["episode,edge,capacity"]
    phase_rows = ["episode,source,phase"]
    step_rows: list[str] = []
    sources = env.sim.net.source_ids.size
    step_rows.append(
        "episode,t,incident_edge,incident_duration,"
        + ",".join(f"z{i}" for i in range(sources))
    )

    for name, policy in policies.items():
        record = name == "shortest-path"  # the draws are policy independent
        for episode in range(EPISODES):
            observation, _ = env.reset(seed=SEED + episode)
            rec = RecordingRNG(env.sim.rng, env.sim)
            env.sim.rng = rec
            if record:
                for edge, cap in enumerate(env.sim.edge_capacity):
                    capacity_rows.append(f"{episode},{edge},{g17(cap)}")
                for source, ph in enumerate(env.sim.source_phase):
                    phase_rows.append(f"{episode},{source},{g17(ph)}")

            total_reward = offered = served = dropped = 0.0
            peaks, means = [], []
            previous = np.zeros(env.action_space.shape, dtype=np.float64)
            churn = 0.0
            steps = 0
            done = False
            while not done:
                action, _ = policy.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float64).reshape(
                    env.action_space.shape
                )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                offered += info["offered"]
                served += info["throughput"]
                dropped += info["dropped"]
                peaks.append(info["max_queue"])
                means.append(info["mean_queue"])
                churn += float(np.mean((action - previous) ** 2))
                previous = action
                steps += 1
                done = terminated or truncated

            metrics_rows.append(
                ",".join(
                    [
                        name.replace(",", " "),
                        str(episode),
                        g17(total_reward),
                        g17(served / max(offered, 1e-8)),
                        g17(dropped / max(offered, 1e-8)),
                        g17(float(np.mean(peaks))),
                        g17(float(np.mean(means))),
                        g17(churn / max(steps, 1)),
                    ]
                )
            )

            if record:
                incident = {t: (e, d) for t, e, d in rec.incidents}
                for t, z in rec.noise:
                    edge, duration = incident.get(t, (-1, 0))
                    step_rows.append(
                        f"{episode},{t},{edge},{duration},"
                        + ",".join(g17(v) for v in z)
                    )

    # the reward-anatomy episode, dgno/figures.py: backpressure on seed 0
    from dgno.baselines import BackpressurePolicy

    observation, _ = env.reset(seed=ANATOMY_SEED)
    rec = RecordingRNG(env.sim.rng, env.sim)
    env.sim.rng = rec
    for edge, cap in enumerate(env.sim.edge_capacity):
        capacity_rows.append(f"{ANATOMY_EPISODE},{edge},{g17(cap)}")
    for source, ph in enumerate(env.sim.source_phase):
        phase_rows.append(f"{ANATOMY_EPISODE},{source},{g17(ph)}")

    policy = BackpressurePolicy(env)
    rc = env.reward_config
    ref = env._reference_throughput
    terms = {"throughput": 0.0, "dropped": 0.0, "congestion": 0.0, "churn": 0.0}
    previous = np.zeros(env.action_space.shape)
    potential = env._potential()
    for _ in range(env.config.horizon):
        action, _ = policy.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(env.action_space.shape)
        observation, _, terminated, truncated, info = env.step(action)
        new_potential = env._potential()
        terms["throughput"] += rc.throughput_weight * info["throughput"] / ref
        terms["dropped"] += -rc.drop_weight * info["dropped"] / ref
        terms["congestion"] += -rc.bottleneck_weight * (
            rc.gamma * new_potential - potential
        )
        terms["churn"] += -rc.smoothness_weight * float(
            np.mean((action - previous) ** 2)
        )
        potential, previous = new_potential, action
        if terminated or truncated:
            break

    incident = {t: (e, d) for t, e, d in rec.incidents}
    for t, z in rec.noise:
        edge, duration = incident.get(t, (-1, 0))
        step_rows.append(
            f"{ANATOMY_EPISODE},{t},{edge},{duration},"
            + ",".join(g17(v) for v in z)
        )

    # The README says 113 of 117 tensors copy across a change of topology and
    # names the four that do not. Nothing had recorded either number.
    census = transfer_census()
    emit(
        "transfer-tensors.csv",
        "kind,count,names\n"
        f"copied,{len(census[0])},\n"
        f"skipped,{len(census[1])}," + " ".join(census[1]) + "\n",
        check,
    )

    emit(
        "reward-anatomy.csv",
        "term,cumulative\n" + "".join(f"{k},{g17(v)}\n" for k, v in terms.items()),
        check,
    )

    emit("episode-metrics.csv", "\n".join(metrics_rows) + "\n", check)
    emit("draws-capacity.csv", "\n".join(capacity_rows) + "\n", check)
    emit("draws-phase.csv", "\n".join(phase_rows) + "\n", check)
    emit("draws-step.csv", "\n".join(step_rows) + "\n", check)
    print(
        f"{'checked' if check else 'wrote'} {len(metrics_rows) - 1} episode rows, "
        f"{len(step_rows) - 1} step rows, "
        f"{len(census[0])} copied and {len(census[1])} skipped tensors"
    )


if __name__ == "__main__":
    main()

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

Run: python verify/export.py
"""

from __future__ import annotations

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
    """Write every fixture file verify/ reads."""
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
    (OUT / "transfer-tensors.csv").write_text(
        "kind,count,names\n"
        f"copied,{len(census[0])},\n"
        f"skipped,{len(census[1])}," + " ".join(census[1]) + "\n"
    )

    (OUT / "reward-anatomy.csv").write_text(
        "term,cumulative\n"
        + "".join(f"{k},{g17(v)}\n" for k, v in terms.items())
    )

    (OUT / "episode-metrics.csv").write_text("\n".join(metrics_rows) + "\n")
    (OUT / "draws-capacity.csv").write_text("\n".join(capacity_rows) + "\n")
    (OUT / "draws-phase.csv").write_text("\n".join(phase_rows) + "\n")
    (OUT / "draws-step.csv").write_text("\n".join(step_rows) + "\n")
    print(
        f"wrote {len(metrics_rows) - 1} episode rows, "
        f"{len(step_rows) - 1} step rows, "
        f"{len(census[0])} copied and {len(census[1])} skipped tensors"
    )


if __name__ == "__main__":
    main()

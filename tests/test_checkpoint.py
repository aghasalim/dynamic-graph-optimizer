"""Re-verify the committed checkpoint against the numbers in the README.

The point of committing a 1.1 MB policy is that the headline claim stops being a
number in a markdown table and becomes something CI re-derives.  Assertions are
written as inequalities against the baselines rather than as exact floats,
because the checkpoint was trained on macOS/arm64 and CI runs Linux/x86 -- the
ranking is stable across platforms, the sixth decimal place is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgno.baselines import BackpressurePolicy, ShortestPathPolicy, evaluate_policy
from dgno.env import DynamicRoutingEnv
from dgno.simulator import NetworkConfig
from dgno.train import TrainConfig
from dgno.transfer import rehost_policy

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "ppo-gnn-4m"
EPISODES = 10  # matches the episode count the README table reports


def _load() -> PPO:
    assert CHECKPOINT.with_suffix(".zip").exists(), f"missing checkpoint at {CHECKPOINT}"
    return PPO.load(str(CHECKPOINT), device="cpu")


def test_checkpoint_beats_both_baselines_on_the_training_grid() -> None:
    """The 4x5 result the README leads with: 0.929 served, 0.602 peak backlog."""
    agent = _load()
    env = DynamicRoutingEnv(seed=0)
    agent_metrics = evaluate_policy(agent, env, episodes=EPISODES)
    naive = evaluate_policy(ShortestPathPolicy(env), env, episodes=EPISODES)
    classical = evaluate_policy(BackpressurePolicy(env), env, episodes=EPISODES)

    assert agent_metrics["served_fraction"] > classical["served_fraction"] > naive[
        "served_fraction"
    ], "the agent no longer beats backpressure on throughput"
    assert agent_metrics["mean_peak_queue"] < classical["mean_peak_queue"] < naive[
        "mean_peak_queue"
    ], "the agent no longer beats backpressure on peak backlog"
    assert agent_metrics["served_fraction"] > 0.90, agent_metrics["served_fraction"]
    assert agent_metrics["mean_peak_queue"] < 0.65, agent_metrics["mean_peak_queue"]
    # Backpressure buys throughput by thrashing the controls; the agent should not.
    # The exact ratio is sample-dependent -- ~13x at 10 episodes, ~9x at 5 -- so this
    # bound is deliberately loose and only asserts the qualitative gap.
    assert agent_metrics["action_churn"] < 0.25 * classical["action_churn"], (
        f"agent churn {agent_metrics['action_churn']:.4f} vs "
        f"backpressure {classical['action_churn']:.4f}"
    )


def test_checkpoint_still_transfers_to_a_larger_grid() -> None:
    """Trained on 4x5, still ahead of backpressure on 6x8 with no retraining."""
    agent = _load()
    env = DynamicRoutingEnv(config=NetworkConfig(rows=6, cols=8), seed=0)
    policy = rehost_policy(agent, env, TrainConfig())

    transferred = evaluate_policy(policy, env, episodes=EPISODES)
    classical = evaluate_policy(BackpressurePolicy(env), env, episodes=EPISODES)
    assert transferred["served_fraction"] > classical["served_fraction"]
    assert transferred["mean_peak_queue"] < classical["mean_peak_queue"]


def test_checkpoint_policy_uses_its_action_range() -> None:
    """Guards against silently shipping an under-trained, near-static policy."""
    agent = _load()
    env = DynamicRoutingEnv(seed=0)
    observation, _ = env.reset(seed=0)
    actions = []
    for _ in range(60):
        action, _ = agent.predict(observation, deterministic=True)
        observation, *_ = env.step(np.asarray(action, dtype=np.float64))
        actions.append(np.asarray(action))
    stacked = np.stack(actions)
    assert stacked.max() - stacked.min() > 1.0, "policy barely uses its action range"
    assert stacked.std(axis=0).mean() > 0.05, "policy is effectively static over time"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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

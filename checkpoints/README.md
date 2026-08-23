# Checkpoints

`ppo-gnn-4m.zip` is the agent reported in the README: 4M steps on a 4x5 grid,
`bottleneck_mode=shaping`, default `TrainConfig`. It is the policy used for both
the headline table and the transfer results.

Load it with:

```python
from stable_baselines3 import PPO
agent = PPO.load("checkpoints/ppo-gnn-4m", device="cpu")
```

`tests/test_checkpoint.py` re-runs the evaluation against both baselines and
fails if the committed agent stops beating backpressure, so the numbers in the
README are checked rather than remembered.

The archive carries SB3's `system_info.txt`, which records the OS and library
versions it was trained under.

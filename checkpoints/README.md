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

`ablations/` holds the six other agents the README discusses, all 400k steps
except `entropy-4m.zip`:

| file | what changed |
|---|---|
| `shaping-400k.zip` | nothing; the 400k baseline |
| `gain-1.0-400k.zip` | action-head init gain 0.01 -> 1.0 |
| `no-churn-400k.zip` | smoothness weight 0.05 -> 0 |
| `gain-and-no-churn-400k.zip` | both of the above |
| `absolute-400k.zip` | `bottleneck_mode=absolute` |
| `entropy-4m.zip` | 4M steps with `ent_coef=0.01` |

Re-derive the whole table with `python -m dgno.ablations`. It scores every agent
under the same reward and seeds, so `return` is comparable across rows -- unlike
the per-run `evaluation.txt` files, each of which uses the reward its agent was
trained on.

The archives carry SB3's `system_info.txt`, which records the OS and library
versions they were trained under.

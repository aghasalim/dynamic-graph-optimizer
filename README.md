# Dynamic Graph Network Optimizer

A routing simulator where demand and link capacity keep moving, and a PPO agent
tries to keep traffic flowing. The network state goes through a GNN (PyTorch
Geometric) and the agent's action is a routing-weight offset on each edge.

I built this to have a project where the baselines were real, so everything is
scored against shortest-path routing and against backpressure on the same seeds.

![network state](docs/network-state.png)

The trained agent mid-episode. Node colour is backlog, green borders are sources
and blue are sinks. Edge width is flow, edge colour is the routing offset.

## Running it

```bash
pip install -e .
python tests/test_dgno.py
python -m dgno.train --timesteps 4000000
python -m dgno.visualize --policy backpressure --steps 150
```

## Results

10 eval episodes on shared seeds. `served` is delivered over offered demand,
`peak_q` is the episode mean of the worst node's backlog, `churn` is how much the
action moves per step.

| policy | served | dropped | peak_q | churn |
|---|---|---|---|---|
| shortest-path | 0.869 | 0.106 | 0.766 | 0.0000 |
| backpressure | 0.901 | 0.075 | 0.672 | 0.1994 |
| PPO + GNN, 400k steps | 0.891 | 0.083 | 0.746 | 0.0002 |
| PPO + GNN, 4M steps | **0.929** | **0.047** | **0.602** | 0.0155 |

At 4M steps it beats backpressure on everything, and does it with about a
thirteenth of the control effort. That last part is the bit I like: backpressure
buys its throughput by thrashing the routing weights every tick, and the agent
gets a better result while barely moving them.

Getting there took a lot more training than I expected. At 400k steps the agent
is basically static (churn 0.0002) and loses to backpressure. Two ablations at
400k, both of which I thought would fix it:

| | served | peak_q | churn |
|---|---|---|---|
| baseline | 0.891 | 0.746 | 0.0002 |
| action head gain 1.0 | 0.877 | 0.759 | 0.0069 |
| no churn penalty | 0.891 | 0.746 | 0.0002 |
| both | 0.875 | 0.767 | 0.0064 |

Raising the action head off its 0.01 init made things worse. Removing the churn
penalty changed nothing to three decimals. Neither was the problem — it just
needed roughly 10x the steps. An entropy bonus of 0.01 at 4M also hurt (0.902
served, 0.682 peak_q), so keeping exploration alive isn't what did it either.

One thing that nearly fooled me. Training `ep_rew_mean` is flat across the whole
4M run, sitting around 235 from 4k steps onward, so the curve in
`docs/curve-long4M.csv` looks completely converged by 100k. The deterministic
policy improved anyway: 0.891 to 0.929 served over the same span. Rollout reward
is collected under a Gaussian with std ~0.35, and the noise cost swamps the
improvement. I'd have stopped this run early if I'd trusted the curve.

The difference shows up directly in the actions. The 400k policy only spans
[-0.278, 0.122] of its available [-1, 1] and moves 0.057 when I slam every queue
to 95% full. The 4M policy uses the full range and moves 0.142.

Raw output in `docs/evaluation-*.txt`, curves in `docs/curve-*.csv`.

## How it works

A 4-connected grid, demand injected on the left column and absorbed on the right.
Each source has its own rush-hour phase so surges hit different corners at
different times, and roads randomly lose 85% of capacity for a while. When a node
fills up it pushes back on its upstream neighbours, one hop per tick, which is
what makes congestion actually spread.

Routing is a softmax over each node's out-edges. At zero action it's plain
shortest-path, so the agent is correcting a working default rather than learning
routing from nothing.

The reward is throughput, minus dropped demand, minus a congestion term, minus a
penalty on moving the action too much:

```
r = w_T * throughput - w_D * dropped
    - w_B * (gamma * Phi(u') - Phi(u))
    - w_S * ||a_t - a_{t-1}||^2
```

`Phi` is a log-sum-exp over normalised queues, which is a smooth stand-in for
"how bad is the worst node". Writing the congestion term as a difference makes it
potential-based shaping, so it can't change the optimal policy, only how fast you
find it. There's a catch I hit: because it telescopes it also contributes almost
nothing to episode return, so it gives no real pressure to flatten backlog.
`--bottleneck-mode absolute` drops the telescoping and does move the optimum, but
scored worse in practice. Despite that, `peak_q` still came down a long way once
the agent trained properly, so throughput and bottleneck relief were less at odds
here than I assumed.

The GNN is a residual GATv2 stack with edge features. Three layers, because
congestion spreads one hop per tick and that makes depth a physical quantity
rather than a taste one. One detail worth knowing if you read `models.py`: SB3
would normally put a dense `Linear(latent, num_edges)` on the policy output,
which throws away the permutation symmetry the GNN is there for, so I replace it
with a weight-shared per-edge head.

Longer write-up of the reward and the GNN choices: [docs/design-notes.md](docs/design-notes.md).

## Layout

```
dgno/simulator.py   queueing dynamics, demand, incidents, spillback
dgno/env.py         Gymnasium wrapper, observations, reward
dgno/models.py      GATv2 encoder, SB3 extractor, policy
dgno/baselines.py   shortest-path, backpressure, evaluation
dgno/train.py       PPO config and entry point
dgno/visualize.py   episode animation
tests/test_dgno.py  invariants, batching, shaping telescoping
```

## Known gaps

Spillback rationing is single-pass, so a throttled node can leave its upstream
slightly over-served within the same tick. It corrects on the next one.

`edge_index` is a buffer on the extractor, so transferring a trained agent to a
different grid size means re-instantiating it. The GNN weights are size-agnostic
and that transfer test is the best evidence for the equivariance argument, but I
haven't run it.

Flow is fluid rather than discrete vehicles, and the agent sees state with no
delay, which is where most of the real difficulty would be.

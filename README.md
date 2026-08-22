# Dynamic Graph Network Optimizer

A routing simulator where demand and link capacity keep moving, and a PPO agent
tries to keep traffic flowing. The network state goes through a GNN (PyTorch
Geometric) and the agent's action is a routing-weight offset on each edge.

I built this to have a project where the baselines were real, so everything is
scored against shortest-path routing and against backpressure on the same seeds.

![network state](docs/network-state.png)

Node colour is backlog, green borders are sources and blue are sinks. Edge width
is flow, edge colour is the routing offset.

## Running it

```bash
pip install -e .
python tests/test_dgno.py
python -m dgno.train --timesteps 400000
python -m dgno.visualize --policy backpressure --steps 150
```

## Results

400k steps, 8 envs, 10 eval episodes on shared seeds. `served` is delivered over
offered demand, `peak_q` is the episode mean of the worst node's backlog, `churn`
is how much the action moves per step.

| policy | served | dropped | peak_q | churn |
|---|---|---|---|---|
| shortest-path | 0.869 | 0.106 | 0.766 | 0.0000 |
| backpressure | 0.901 | 0.075 | 0.672 | 0.1994 |
| PPO + GNN | 0.891 | 0.083 | 0.746 | 0.0002 |

So it beats shortest-path and doesn't reach backpressure. I've left that as-is
rather than tuning until it wins.

The churn column is the useful part. Backpressure sits at 0.199 and the agent at
0.0002, which means the agent found a fixed rebalance of the routing weights and
never learned to react to anything. It isn't a wiring bug — I fed the trained
policy an observation with every queue at 95% full and its output moved by 0.024,
and its actions only span [-0.278, 0.122] out of [-1, 1]. It responds, barely.

I tried the two obvious fixes and both failed:

| | served | peak_q | churn |
|---|---|---|---|
| baseline | 0.891 | 0.746 | 0.0002 |
| action head gain 1.0 | 0.877 | 0.759 | 0.0069 |
| no churn penalty | 0.891 | 0.746 | 0.0002 |
| both | 0.875 | 0.767 | 0.0064 |

Raising the action head off its 0.01 init makes things worse. Removing the churn
penalty changes nothing at all, so that term was never what was holding the agent
still. Long-run training is the remaining thing to check.

Raw output is in `docs/evaluation-*.txt`.

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
scored worse in practice.

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

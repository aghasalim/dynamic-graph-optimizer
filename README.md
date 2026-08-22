# Dynamic Graph Network Optimizer

A routing simulator where demand and link capacity keep shifting, and a PPO agent
tries to keep traffic moving. State is encoded with a graph neural network
(PyTorch Geometric), and the agent's action is one routing-logit offset per
directed edge.

I built this mostly to have something where the baselines were real. Every result
below is scored against shortest-path routing and against backpressure
(max-weight) scheduling on the same seeds, same demand, same incidents.

![network state under backpressure routing](docs/network-state.png)

Backpressure routing mid-episode. Node colour is backlog, green borders are
sources, blue borders are sinks. Edge width is flow and edge colour is the
routing offset, red for boosted and blue for suppressed.

```bash
pip install -e .
python tests/test_dgno.py
python -m dgno.train --timesteps 400000
python -m dgno.visualize --policy backpressure --steps 150
```

## Results

400k timesteps, 8 parallel envs, evaluated over 10 episodes on shared seeds.

| policy | served | dropped | peak_q | mean_q | churn |
|---|---|---|---|---|---|
| shortest-path | 0.869 | 0.106 | 0.766 | 0.283 | 0.0000 |
| backpressure | **0.901** | **0.075** | **0.672** | **0.262** | 0.1994 |
| PPO + GNN (shaping) | 0.891 | 0.083 | 0.746 | 0.285 | 0.0002 |
| PPO + GNN (absolute) | 0.881 | 0.094 | 0.757 | 0.277 | 0.0001 |

`served` is delivered over offered demand. `peak_q` is the episode mean of the
worst node's backlog, which is the number I actually care about since a policy
that raises throughput by gridlocking one intersection hasn't solved anything.
`churn` is mean squared action change per step.

The agent beats shortest-path and does not reach backpressure. I'm leaving that
in rather than tuning until it wins, because the failure mode turned out to be
more interesting than the gap.

Look at the churn column. Backpressure sits at 0.199 and the agent at 0.0002, so
the agent has converged to something almost static. It found a fixed rebalance of
the routing weights that helps a bit, and never learned to react. I checked this
wasn't a wiring bug: feeding the trained policy an observation with every queue
slammed to 95% full moves its output by 0.024 on average, and its actions span
only [-0.278, 0.122] out of an available [-1, 1]. So it responds to state, just
barely. That's an under-trained policy, not a dead one.

The two reward modes are worth comparing directly. `shaping` scores better on
throughput than `absolute`, which is the opposite of what I expected when I added
the absolute mode specifically to attack bottlenecks. Note also that the two
modes have different reward functions, so their `return` values aren't comparable
across modes (raw numbers in `docs/evaluation-*.txt`).

What I'd try next, roughly in order: train 10x longer, since nothing here looks
converged; raise the gain on the action head, which currently initialises at 0.01
and stays small; and drop the churn penalty to zero as an ablation, because I may
be penalising exactly the reactivity that makes backpressure work.

## The environment

A 4-connected grid. The left column injects demand, the right column absorbs it.
Each node holds a backlog `q_i`, each directed edge has a per-step capacity.

One tick does five things:

1. Each source draws `lambda_s(t) = base * (1 + A sin(2 pi t / P + phi_s)) * (1 + noise)`.
   Every source gets its own phase, so rush hours hit different corners at
   different times. Anything that doesn't fit in the source queue is dropped and
   counted against the agent.
2. The split across a node's out-edges is a softmax over
   `logit_e = -beta * hops_to_sink(dst_e) + gain * action_e`. At `action = 0` this
   is plain shortest-path routing, so the agent is correcting a working default
   instead of learning routing from scratch.
3. `flow_e = min(q_src * split_e, c_e * incident_e)`.
4. If arrivals at a node exceed its headroom, every inbound edge scales down
   proportionally. This is what makes congestion spread: a full node pushes back
   on its upstream neighbours, one hop per tick.
5. With some probability a random road loses 85% of capacity for 8 to 25 steps,
   in both directions.

The observation packs 8 node features and 5 edge features into a flat vector. The
topology is static so it lives in the model, not the observation.

| node | edge |
|---|---|
| normalised backlog | live capacity |
| inflow, outflow | last flow |
| is-source, is-sink | utilisation |
| hops to sink | previous action |
| `sin`, `cos` of demand phase | differential backlog `q_i - q_j` |

The demand phase is observable on purpose. A real traffic controller knows the
time of day, and it's the one signal backpressure structurally cannot use.

## Reward

Write `u_i = q_i / q_max`.

### Why not mean queue

A bottleneck is concentration, not volume. Two networks with the same total
backlog aren't equally healthy if one has it spread evenly and the other has it
all on one link. For fixed total `sum_i q_i = Q`, Jensen says any convex `phi`
makes `sum_i phi(q_i)` minimal exactly at the uniform allocation. So convexity is
how you write down "spread the load". A linear penalty on mean backlog can't see
the difference at all.

### The bottleneck surrogate

The honest objective is `max_i u_i`, but it's non-smooth and its gradient is
one-hot, so on a 20-node grid 19 nodes get nothing. Log-sum-exp instead:

```
Phi_tau(u) = tau * log( (1/N) * sum_i exp(u_i / tau) )
d Phi / d u_i = softmax(u / tau)_i
```

Three properties I rely on. It's sandwiched, `mean(u) <= Phi_tau(u) <= max(u)` for
any `tau > 0`, by Jensen on the left and `mean(exp) <= exp(max)` on the right. It
interpolates, going to `mean(u)` as `tau` grows and to `max(u)` as `tau` shrinks,
closing that gap at rate `tau * log N`. And the gradient is a softmax over nodes,
so each node gets blamed in proportion to how nearly it *is* the bottleneck.
Default `tau = 0.25`.

### Drift, not level

`Phi_tau` is a Lyapunov function on backlog. The obvious move is to penalise its
level. I penalise its drift instead, in Ng-Harada-Russell form:

```
r_t = w_T * T_t / T_ref                                  throughput
    - w_D * D_t / T_ref                                  dropped demand
    - w_B * ( gamma * Phi_tau(u_{t+1}) - Phi_tau(u_t) )  congestion
    - w_S * || a_t - a_{t-1} ||^2 / |E|                  control churn
```

The congestion term is `F(s, a, s') = gamma * Psi(s') - Psi(s)` with `Psi = -Phi_tau`.
Potential-based shaping like this provably can't change the optimal policy of the
underlying throughput MDP, only how fast you find it. That matters because
otherwise congestion created at t=40 gets punished as a throughput loss around
t=90, and GAE has to carry the blame 50 steps through a noisy return.

The usual worry with shaping is an agent farming the bonus in a loop. `F`
telescopes, so around any closed path in state space it sums to zero at
`gamma = 1`. I didn't want to take that on faith, so
`test_shaping_reward_telescopes_over_a_rollout` zeroes every other weight and
asserts the summed return collapses to `Phi(start) - Phi(end)` for arbitrary
actions.

There's a catch I ran into. Because it telescopes, the congestion term contributes
roughly nothing to episode return, which is what policy-invariance *means*. So
`shaping` mode gives no incentive to flatten backlog beyond what throughput
already implies. That's why `bottleneck_mode="absolute"` exists: it swaps in
`-w_B * Phi_tau(u_{t+1})`, doesn't telescope, and does move the optimum. In the
table above it also scored worse, which I don't yet have a clean explanation for.

### Relation to backpressure

Minimising the drift of `V(q) = (1/2) sum_i q_i^2` gives, in closed form, routing
proportional to differential backlog `(q_i - q_j) * c_ij`. That's backpressure,
and it's throughput-optimal for this class of network. So a drift-penalising
reward makes the classical algorithm roughly a stationary point of the RL
objective, which is why I used it as the baseline rather than something easier to
beat.

The agent's theoretical edge is that backpressure is memoryless and clock-blind.
It only reacts to backlog that already exists, while the agent sees the demand
phase and could pre-position for a surge. As of this run it doesn't.

### Two things that are easy to get wrong

Everything is normalised to O(1) before weighting. Reward magnitude acts as an
implicit multiplier on the value-loss gradient, and an unnormalised throughput
around 40 sitting next to a shaping term around 0.01 means the critic never sees
the shaping.

The churn term `|| a_t - a_{t-1} ||^2` is the discrete version of an LQR
control-effort cost. Without it routing controllers flap, shifting load to an
empty branch, congesting it, shifting back. Given the results above I now suspect
`w_S = 0.05` is too high here.

## The GNN

### Equivariance

Relabel the intersections and the optimal control relabels the same way. An MLP on
flattened state has to learn that from data; a GNN gets it structurally. That only
holds if the symmetry survives to the output, which is why `GraphActorCriticPolicy`
deletes SB3's default `action_net`. SB3 would build `Linear(latent_pi, num_edges)`,
a dense mixing layer that throws away the structure the GNN was added for. The
action head is `Linear(edge_latent_dim, 1)` applied to every edge with shared
weights instead. `test_policy_action_head_is_permutation_equivariant` checks the
swap happened and that no parameters fell out of the optimizer along the way.

### Depth

Each message-passing layer widens a node's receptive field by one hop, and
spillback in this simulator propagates one hop per tick. So L layers is L
timesteps of spatial foresight, which means depth is set by the physics rather
than by taste. I use 3. Past three or four, over-smoothing costs more than the
extra range buys, which is also why every layer is residual with LayerNorm.

### Attention

`GCNConv` aggregates with fixed degree-normalised weights, but which neighbour
matters changes every step as incidents land. `GATv2Conv` with `edge_dim`
conditions each attention coefficient on that edge's live capacity, load and
utilisation, so a node can learn to ignore a neighbour whose link just lost most
of its capacity. GATv2 rather than GAT because GAT's attention is static in the
query, so its ranking of neighbours can't depend on which node is asking.

### Readouts and batching

Actions are per-edge, so I decode `z_e = MLP([h_i || h_j || e_ij])`, which is
equivariant. Value is a graph-level scalar, so I pool `[mean || max]` over nodes,
which is invariant. Mean carries average load and max carries the bottleneck, and
the critic needs both since the penalty term is itself a soft maximum.

Topology never changes, so rather than rebuilding a `torch_geometric.data.Batch`
every rollout step, the extractor replicates `edge_index` and offsets it by
`b * num_nodes`. Cheaper, and it keeps the observation a plain `Box` so SB3's
vectorised envs work untouched. An off-by-one there would silently wire batch
elements together and still train, just worse, so
`test_graph_extractor_batching_matches_single_samples` checks the batched forward
equals per-sample forwards.

## Layout

```
dgno/simulator.py   queueing dynamics, demand, incidents, spillback
dgno/env.py         Gymnasium wrapper, observation packing, reward
dgno/models.py      GATv2 encoder, SB3 feature extractor, equivariant policy
dgno/baselines.py   shortest-path, backpressure, evaluation harness
dgno/train.py       PPO config and entry point
dgno/visualize.py   episode animation
tests/test_dgno.py  invariants, batching, shaping telescoping
```

## Limitations

Spillback rationing is single-pass, so a node scaled down for lack of headroom can
leave its own upstream slightly over-served within the same tick. The residual is
bounded by one step of capacity and corrects on the next. A fixed-point iteration
would be exact but isn't worth it at this scale.

`edge_index` is a buffer on the extractor, so a trained agent only transfers to a
different grid size by re-instantiating the extractor. The GNN weights themselves
are size-agnostic, and that transfer experiment is the most convincing test of the
equivariance argument. I haven't run it.

Flow is fluid, not discrete vehicles, which is fine for capacity planning and
wrong for anything per-packet. And the agent sees state instantly, where real
controllers see it with lag, which is where most of the real difficulty lives.

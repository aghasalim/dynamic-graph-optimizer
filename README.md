# Dynamic Graph Network Optimizer

A simulation environment where a PPO agent learns to route traffic across a
spatial network whose demand and link capacity keep changing underneath it.
State is encoded by a graph neural network (PyTorch Geometric); control is one
routing-logit offset per directed edge.

![network state under backpressure routing](docs/network-state.png)

*Backpressure routing mid-episode. Node colour is backlog, node border marks
sources (green, left) and sinks (blue, right). Edge width is flow, edge colour is
the agent's routing offset — red boosts an edge, blue suppresses it.*

The point of the repo is the **evaluation**, not the demo: the agent is scored
against shortest-path routing and against backpressure (max-weight) scheduling on
identical demand realisations and incident sequences.

```bash
pip install -e .
python tests/test_dgno.py                                  # invariants + batching checks
python -m dgno.train --timesteps 400000                    # train, then print the comparison
python -m dgno.visualize --policy backpressure --steps 150 # animate an episode
```

---

## 1. The environment

A 4-connected grid of `rows x cols` intersections. The left column injects demand,
the right column absorbs it. Each node holds a backlog `q_i`; each directed edge
has a per-step capacity `c_e`.

**One tick.**

1. **Demand.** Each source draws `lambda_s(t) = base * (1 + A sin(2 pi t / P + phi_s)) * (1 + noise)`.
   Every source gets its own phase `phi_s`, so rush hours arrive at different
   corners at different times. Anything that will not fit in the source queue is
   **dropped** and counted against the agent.
2. **Routing.** For node `i`, the split across its out-edges is a softmax:

   ```
   logit_e = -beta * hops_to_sink(dst_e) + gain * action_e
   ```

   With `action = 0` this is pure shortest-path routing. The agent therefore
   learns a *correction* to a working default rather than a routing policy from
   nothing — the difference between a warm start and 100k wasted steps.
3. **Service.** `flow_e = min(q_src * split_e, c_e * incident_e)`.
4. **Spillback.** If arrivals at `j` exceed its remaining headroom, every inbound
   edge is scaled down proportionally. This is what makes congestion *spread*:
   a full node pushes back on its upstream neighbours, one hop per tick.
5. **Incidents.** With probability `incident_rate` a random road (both directions)
   loses 85% of its capacity for 8–25 steps. This is the non-stationarity the GNN
   has to react to; it is not in the observation as a flag the agent can look up
   ahead of time.

**Observation** — a flat vector packing two blocks, since the topology is static
and lives in the model rather than in the observation:

| node features (8) | edge features (5) |
|---|---|
| normalised backlog | live capacity |
| inflow, outflow | last flow |
| is-source, is-sink | utilisation |
| hops to sink | previous action |
| `sin`, `cos` of the demand phase | differential backlog `q_i - q_j` |

The demand phase is observable on purpose: a real traffic controller knows the
time of day, and it is precisely the signal a myopic policy like backpressure
cannot use.

**Action** — `Box(-1, 1, (num_edges,))`, one routing-logit offset per edge.

---

## 2. Reward design

Write `u_i = q_i / q_max` for normalised backlog.

### 2.1 Why not just the mean queue

A bottleneck is **concentration**, not volume. Two networks with the same total
backlog are not equally healthy: one with it spread evenly is fine, one with all
of it on a single link is gridlocked. For a fixed total `sum_i q_i = Q`, Jensen's
inequality says any convex `phi` makes `sum_i phi(q_i)` minimal exactly at the
uniform allocation. **Convexity is the mathematical encoding of "spread the
load."** A linear penalty on mean backlog is blind to the distinction.

### 2.2 The bottleneck surrogate

The honest bottleneck objective is `max_i u_i`. It is also useless as a training
signal: it is non-smooth, and its gradient is a one-hot vector, so on a 20-node
grid 19 nodes receive nothing. Use log-sum-exp instead:

```
Phi_tau(u) = tau * log( (1/N) * sum_i exp(u_i / tau) )
d Phi / d u_i = softmax(u / tau)_i
```

Properties that matter:

- **Sandwiched:** `mean(u) <= Phi_tau(u) <= max(u)` for every `tau > 0` (Jensen on
  the left, `mean(exp) <= exp(max)` on the right). So it is bounded and
  interpretable regardless of temperature.
- **Interpolates:** `Phi_tau -> mean(u)` as `tau -> inf`, and `Phi_tau -> max(u)`
  as `tau -> 0`, closing the gap at rate `tau * log N`.
- **Soft credit assignment:** the gradient is a softmax over nodes, so each node
  is blamed in proportion to how nearly it *is* the bottleneck. `tau` is the dial
  between "penalise the average" and "penalise only the worst offender."

Default `tau = 0.25`.

### 2.3 Lyapunov drift as potential-based shaping — the key move

`Phi_tau` is a Lyapunov function on backlog. The instinct is to penalise its
*level*. Penalise its *drift* instead, in exactly the Ng–Harada–Russell form:

```
r_t = w_T * T_t / T_ref                                  throughput
    - w_D * D_t / T_ref                                  dropped demand
    - w_B * ( gamma * Phi_tau(u_{t+1}) - Phi_tau(u_t) )  congestion  <- shaping
    - w_S * || a_t - a_{t-1} ||^2 / |E|                  control churn
```

The congestion term is `F(s, a, s') = gamma * Psi(s') - Psi(s)` with `Psi = -Phi_tau`.
Potential-based shaping is **policy-invariant**: it provably cannot change the
optimal policy of the underlying throughput MDP. It only changes how fast you find
it. That matters here because without it, congestion created at `t = 40` is
punished as a throughput loss around `t = 90`, and GAE has to carry the blame 50
steps through a noisy return. With it, the penalty lands on the step that caused
it.

The usual objection to shaping is reward hacking — an agent that farms the bonus
in a cycle. Potential-based shaping is immune by construction: `F` telescopes, so
around any closed loop in state space it sums to zero (at `gamma = 1`). That is
not a claim to take on faith, so it is a test:
`test_shaping_reward_telescopes_over_a_rollout` zeroes every other weight and
asserts the summed return collapses to `Phi(start) - Phi(end)` for arbitrary
actions.

`bottleneck_mode="absolute"` swaps in `-w_B * Phi_tau(u_{t+1})`, which does *not*
telescope and therefore **does** move the optimum — buying flatter backlog and
lower latency at the cost of raw throughput. Which you want is a product decision,
not a mathematical one, so it is a flag rather than a default.

### 2.4 The connection to backpressure

Minimising the drift of the quadratic Lyapunov function `V(q) = (1/2) sum_i q_i^2`
yields, in closed form, routing proportional to differential backlog
`(q_i - q_j) * c_ij` — that is **backpressure / max-weight**, which is
throughput-optimal for this class of network. So a drift-penalising reward makes
the classical algorithm approximately a stationary point of the RL objective.

This is why backpressure is the baseline and not a strawman: beating it is the
actual bar. The agent's edge is that backpressure is **myopic and clock-blind**.
It reacts to backlog that already exists. The agent sees `sin`/`cos` of the demand
phase and can pre-position capacity *before* a surge arrives — a policy no
memoryless max-weight rule can express.

### 2.5 Two details that are easy to get wrong

**Normalisation.** Every term is scaled to `O(1)` before weighting. Reward
magnitude is an implicit multiplier on the value-loss gradient; an unnormalised
throughput around 40 next to a shaping term around 0.01 means the critic never
sees the shaping at all. `VecNormalize(norm_reward=True)` handles the residual.

**Control churn.** `|| a_t - a_{t-1} ||^2` is the discrete analogue of an LQR
control-effort cost. Without it, routing controllers flap: shift load to an empty
branch, congest it, shift back, repeat — a limit cycle, and the mechanism behind
real route-oscillation and Braess-style pathologies. Quadratic damping kills it.
It is also why the eval table reports `churn`: backpressure wins on throughput
partly by being twitchy, and that is a cost a real deployment pays.

---

## 3. What the GNN layers capture

### 3.1 Permutation equivariance is the whole argument

Relabel the intersections and the optimal control relabels identically. An MLP on
a flattened state must *learn* that symmetry from data; a GNN has it structurally.
That is the sample-efficiency case, and it is only real if the symmetry survives
end to end — which is why `GraphActorCriticPolicy` **deletes SB3's default
`action_net`**. SB3 would otherwise build `Linear(latent_pi, num_edges)`, a dense
mixing layer that throws away exactly the structure the GNN was added to exploit.
The action head is instead `Linear(edge_latent_dim, 1)` applied to every edge with
shared weights. `test_policy_action_head_is_permutation_equivariant` asserts the
swap happened and that no parameters fell out of the optimizer in the process.

### 3.2 Depth is a physical quantity

Each message-passing layer widens a node's receptive field by one hop. In this
simulator spillback propagates exactly one hop per tick. So `L` layers is `L`
timesteps of spatial foresight — depth is set by the propagation speed of the
phenomenon being modelled, not by taste. `num_layers = 3` here. Past three or four
hops, over-smoothing (node embeddings converging toward a constant vector) costs
more than the extra range buys, which is also why every layer is residual +
`LayerNorm`.

### 3.3 Attention, because the bottleneck moves

`GCNConv` aggregates with fixed degree-normalised weights. But *which* neighbour
matters changes every step as incidents land. `GATv2Conv` with `edge_dim`
conditions each attention coefficient on the edge's live capacity, load and
utilisation, so a node can learn "ignore the neighbour whose link just lost 85% of
capacity." GATv2 rather than GAT because GAT's attention is static in the query —
its ranking of neighbours cannot depend on which node is asking.

### 3.4 Two readouts for two symmetries

Actions are per-edge, so decode `z_e = MLP([h_i || h_j || e_ij])` — equivariant.
Value is a graph-level scalar, so pool `[mean || max]` over nodes — invariant.
Mean carries average load, max carries the bottleneck; the critic needs both to
predict a return whose penalty term is itself a soft maximum.

### 3.5 Batching

Topology is static, so instead of rebuilding `torch_geometric.data.Batch` on every
rollout step, the extractor replicates `edge_index` and offsets it by
`b * num_nodes`. Cheaper, and it keeps the observation a plain `Box` so SB3's
vectorised envs and buffers work untouched. An off-by-one there would silently
wire batch elements together and still train, just worse — so
`test_graph_extractor_batching_matches_single_samples` checks the batched forward
equals per-sample forwards.

---

## 4. Results

See `runs/ppo/evaluation.txt` after training. All policies are scored on identical
seeds, so the comparison is paired.

| column | meaning |
|---|---|
| `return` | mean episode return (reward-shaped; comparable across policies, not to zero) |
| `served` | delivered / offered demand — the headline throughput number |
| `dropped` | demand refused at the source because the queue was full |
| `peak_q` | mean over the episode of the worst node's backlog — **the bottleneck metric** |
| `mean_q` | mean backlog across all nodes |
| `churn` | mean squared action change per step — control effort |

`served` and `peak_q` are the two that matter. A policy that raises throughput by
letting one intersection gridlock has not solved the problem.

---

## 5. Layout

```
dgno/simulator.py   queueing dynamics, demand process, incidents, spillback
dgno/env.py         Gymnasium wrapper, observation packing, reward
dgno/models.py      GATv2 encoder, SB3 feature extractor, equivariant policy
dgno/baselines.py   shortest-path + backpressure, shared evaluation harness
dgno/train.py       PPO configuration and training entry point
dgno/visualize.py   NetworkX/Matplotlib episode animation
tests/test_dgno.py  invariants, batching, shaping telescoping
```

## 6. Known limitations

- **Single-pass spillback rationing.** A node scaled down for lack of headroom may
  leave its own upstream marginally over-served within the same tick. The residual
  is bounded by one step of capacity and self-corrects on the next. A fixed-point
  iteration would be exact; it is not worth the cost at this scale.
- **Static topology.** `edge_index` is a buffer on the extractor, so a trained
  agent transfers to a new grid size only by re-instantiating the extractor. The
  GNN weights themselves are size-agnostic — that transfer experiment is the
  obvious next thing to run, and the most convincing evidence for the whole
  equivariance argument.
- **Unsplittable demand is not modelled.** Flow is a fluid, not discrete vehicles.
  Fine for capacity planning, wrong for anything latency-per-packet.
- **No delay in the observation.** The agent sees the current state instantly.
  Real controllers see it with lag, which is where most of the difficulty lives.

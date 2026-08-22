# Design notes

Longer reasoning behind the reward and the network, kept out of the README.

## Why the congestion term isn't just mean queue

A bottleneck is concentration, not volume. Two networks holding the same total
backlog aren't equally healthy if one spreads it evenly and the other piles it on
one link. For a fixed total, Jensen's inequality says any convex `phi` makes
`sum_i phi(q_i)` minimal exactly at the uniform allocation, so convexity is how
you write down "spread the load". A linear penalty on mean backlog can't tell the
two cases apart.

The honest objective is `max_i u_i` where `u_i = q_i / q_max`. It's also useless
to train on: non-smooth, and its gradient is one-hot, so on a 20-node grid 19
nodes get no signal. Log-sum-exp instead:

```
Phi_tau(u) = tau * log( (1/N) * sum_i exp(u_i / tau) )
d Phi / d u_i = softmax(u / tau)_i
```

Properties I use:

- Sandwiched: `mean(u) <= Phi_tau(u) <= max(u)` for any `tau > 0`. Jensen on the
  left, `mean(exp) <= exp(max)` on the right.
- Interpolates: goes to `mean(u)` as `tau` grows, to `max(u)` as `tau` shrinks,
  closing that gap at rate `tau * log N`.
- The gradient is a softmax over nodes, so blame lands in proportion to how nearly
  a node *is* the bottleneck.

Default `tau = 0.25`.

## Drift instead of level

`Phi_tau` is a Lyapunov function on backlog. The obvious thing is to penalise its
level. I penalise the drift, in Ng-Harada-Russell form, so the congestion term is
`F(s, a, s') = gamma * Psi(s') - Psi(s)` with `Psi = -Phi_tau`.

Potential-based shaping provably can't change the optimal policy of the
underlying MDP, only how fast you find it. Which matters here: without it,
congestion created around t=40 gets punished as a throughput loss around t=90 and
GAE has to carry the blame 50 steps through a noisy return.

The standard worry with shaping is an agent farming the bonus in a loop. `F`
telescopes, so around any closed path it sums to zero at `gamma = 1`. That's the
kind of claim worth checking rather than trusting, so
`test_shaping_reward_telescopes_over_a_rollout` zeroes every other weight and
asserts the summed return collapses to `Phi(start) - Phi(end)` for arbitrary
actions.

The catch is the flip side of the same property. Because it telescopes, the
congestion term contributes roughly nothing to episode return, which is exactly
what policy-invariance means. So it gives no pressure to flatten backlog beyond
what throughput already implies, and `peak_q` in the results barely moves off the
shortest-path number. `--bottleneck-mode absolute` swaps in `-w_B * Phi_tau(u')`,
which doesn't telescope and does move the optimum. It scored worse, and I don't
have a clean explanation for that yet.

## Why backpressure is the baseline

Minimising the drift of `V(q) = (1/2) sum_i q_i^2` gives, in closed form, routing
proportional to differential backlog `(q_i - q_j) * c_ij`. That's backpressure,
and it's throughput-optimal for this class of network. So a drift-penalising
reward makes the classical algorithm roughly a stationary point of the objective
I'm optimising, which is why it's the baseline rather than something easier.

The agent's theoretical edge is that backpressure is memoryless and clock-blind:
it only reacts to backlog that already exists. The agent sees `sin`/`cos` of the
demand phase and could pre-position before a surge. As of the current runs it
doesn't.

## Scaling

Every reward term is normalised to O(1) before weighting. Reward magnitude acts as
an implicit multiplier on the value-loss gradient, and an unnormalised throughput
around 40 next to a shaping term around 0.01 means the critic never sees the
shaping at all.

## The network

Relabel the intersections and the optimal control relabels the same way. An MLP on
flattened state has to learn that from data, a GNN gets it structurally. That only
holds if the symmetry survives to the output, which is why
`GraphActorCriticPolicy` deletes SB3's default `action_net` — SB3 would build
`Linear(latent_pi, num_edges)`, a dense mixing layer that discards the structure
the GNN was added for. The action head is `Linear(edge_latent_dim, 1)` applied to
every edge with shared weights instead.

Depth: each message-passing layer widens a node's receptive field by one hop, and
spillback propagates one hop per tick, so L layers is L timesteps of spatial
foresight. Three here. Past three or four, over-smoothing costs more than the
range buys, which is also why the layers are residual with LayerNorm.

`GATv2Conv` rather than `GCNConv` because which neighbour matters changes every
step as incidents land, and attention conditioned on `edge_dim` lets a node learn
to ignore a neighbour whose link just failed. GATv2 rather than GAT because GAT's
attention is static in the query, so its ranking of neighbours can't depend on
which node is asking.

Readouts: actions are per-edge so the decoder is `MLP([h_i || h_j || e_ij])`,
which is equivariant. Value is a graph-level scalar so it pools `[mean || max]`,
which is invariant. Mean carries average load and max carries the bottleneck, and
the critic needs both since the penalty term is itself a soft maximum.

Batching: topology never changes, so rather than rebuilding a
`torch_geometric.data.Batch` every rollout step, the extractor replicates
`edge_index` and offsets it by `b * num_nodes`. Cheaper, and it keeps the
observation a plain `Box` so SB3's vectorised envs work untouched. An off-by-one
there would silently wire batch elements together and still train, just worse, so
`test_graph_extractor_batching_matches_single_samples` checks that a batched
forward equals per-sample forwards.

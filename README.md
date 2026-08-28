# Dynamic Graph Network Optimizer

**Can a GNN policy learn routing control that beats backpressure, and does it transfer to graphs it never saw?**

[![ci](https://img.shields.io/badge/ci-passing-brightgreen.svg)](.github/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-13%20passing-brightgreen.svg)](tests/)
[![licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

![the trained agent mid-episode](docs/network-state.png)

<sub>The trained agent mid-episode. Node colour is backlog, green borders are
sources and blue are sinks; edge width is flow and edge colour is the routing
offset.</sub>

---

## Abstract

Routing on a spatial network under time-varying demand and random link failure is
a control problem with a strong classical baseline: backpressure scheduling is
throughput-optimal for this class of queueing network, and minimising the drift of
a quadratic Lyapunov function reproduces it in closed form. This work asks whether
a graph neural network policy trained with PPO can beat it, and whether what it
learns is a property of the graph it trained on.

At 4M steps the agent beats backpressure on four of the five metrics I score,
0.929 served against 0.901, 0.047 dropped against 0.075, 0.602 mean peak backlog
against 0.672, while moving the routing weights roughly an order of magnitude
less. It loses the fifth: mean backlog over all nodes is 0.268 against 0.262. It
holds the worst node down and carries a little more queue everywhere else.
Trained on a 4x5 grid, it stays ahead on grids up to 8x10, four times the nodes
and 4.6 times the edges, with no retraining and no decay in the margin. That
transfer is the payoff of keeping the policy permutation-equivariant: 113 of 117
tensors copy across a change of topology, and the four that do not
are `edge_index` buffers and `log_std`.

Getting there took ten times the training budget I expected, and two plausible
fixes that both made things worse. Those are reported alongside the result.

**Contributions.** (i) A queueing simulator with spillback, incidents and
per-source rush-hour phases where the harm of a routing decision is measurable.
(ii) A reward whose congestion term is potential-based shaping, with the
telescoping property verified by test rather than asserted. (iii) An equivariant
policy head that replaces SB3's dense action layer, with the transfer experiment
that justifies it. (iv) Ablations showing which of the obvious fixes did not work.

---

## 1. Running it


```bash
pip install -e .
python tests/test_dgno.py
python -m dgno.train --timesteps 4000000
python -m dgno.visualize --policy backpressure --steps 150
```

## 2. Results

### 2.1 Against the baselines
10 eval episodes on shared seeds. `served` is delivered over offered demand, `peak_q` is the episode mean of the worst node's backlog, `churn` is how much the action moves per step.

![policy comparison](docs/policy-comparison.png)
![learning curve](docs/learning-curve.png)

![the same run replayed in step order](docs/learning-curve.gif)

*The 4M run replayed in step order. The dashed line is its own first reading
at 4k steps, and the curve spends a long stretch below it before climbing back
past.*

![every checkpoint on the same seeds](docs/ablations.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#21-against-the-baselines).
### 2.2 Transfer to other grid sizes
Trained on 4x5, evaluated with no retraining. `sp` is shortest-path, `bp` is backpressure.

![transfer scaling](docs/transfer-scaling.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#22-transfer-to-other-grid-sizes).
## 3. Reproducibility

The agent behind both tables is committed at `checkpoints/ppo-gnn-4m.zip` (1.1 MB),
so you don't have to spend the four hours retraining it:

```bash
python tests/test_checkpoint.py
```

That reloads it, re-runs the evaluation against both baselines, and fails if it
stops beating backpressure on throughput or peak backlog, or if it drifts back
towards a static policy. It runs in CI, so the numbers above are re-derived on
every push rather than being a table I typed once.

The ablation agents are committed too, under `checkpoints/ablations/`. Re-derive
that whole table in about 15 seconds:

```bash
python -m dgno.ablations
```

It scores every agent under the same reward and seeds, so `return` is comparable
across rows there -- unlike the per-run files in `docs/`, each of which uses the
reward its own agent was trained on.

Retraining from scratch, if you want to: `python -m dgno.train --timesteps 4000000`,
about four hours on 8 CPU envs.

## 4. Method
The congestion term is written as potential-based shaping, `gamma*Phi(s') - Phi(s)`, which is what makes it policy-invariant.

![what each reward term contributes over an episode](docs/reward-anatomy.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#4-method).
## 5. Repository layout

```
dgno/simulator.py   queueing dynamics, demand, incidents, spillback
dgno/env.py         Gymnasium wrapper, observations, reward
dgno/models.py      GATv2 encoder, SB3 extractor, policy
dgno/baselines.py   shortest-path, backpressure, evaluation
dgno/train.py       PPO config and entry point
dgno/transfer.py    re-host a trained policy on a different grid
dgno/ablations.py   re-derive the ablation table from the checkpoints
checkpoints/        the 4M agent and the six ablation agents
dgno/visualize.py   episode animation
dgno/figures.py     regenerate the figures in this README
tests/test_dgno.py  invariants, batching, shaping telescoping
tests/test_checkpoint.py  re-derives the reported numbers from the checkpoint
```

## 6. Limitations

Spillback rationing is single-pass, so a throttled node can leave its upstream
slightly over-served within the same tick. It corrects on the next one.

Flow is fluid rather than discrete vehicles, and the agent sees state with no
delay, which is where most of the real difficulty would be.

## References

The papers and sources this implementation follows. Each one is here because
the code uses the method, the dataset or the metric it describes.

- **Veličković, Cucurull, Casanova, Romero, Liò, Bengio. Graph Attention Networks. ICLR 2018.** [arXiv:1710.10903](https://arxiv.org/abs/1710.10903) the GAT state encoder.
- **Schulman, Wolski, Dhariwal, Radford, Klimov. Proximal Policy Optimization Algorithms. 2017.** [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) the control policy.
- **Fey, Lenssen. Fast Graph Representation Learning with PyTorch Geometric. 2019.** [arXiv:1903.02428](https://arxiv.org/abs/1903.02428) the graph library.
- **Raffin, Hill, Gleave et al. Stable-Baselines3. JMLR 22, 2021.** the PPO implementation.

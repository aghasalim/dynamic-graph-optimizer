"""Graph neural network state encoder and the SB3 policy that sits on top of it.

The design constraint is symmetry.  Relabelling the intersections must relabel
the optimal control identically, so every stage is either permutation-equivariant
(the node/edge trunk, the per-edge action head) or permutation-invariant (the
pooled value head).  A dense ``Linear(hidden, num_edges)`` readout would discard
exactly the structure the GNN was added to exploit, so the action head is a
weight-shared decoder applied to each edge and SB3's own ``action_net`` is
replaced with an identity.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch_geometric.nn import GATv2Conv

from .env import GraphSpec

__all__ = [
    "GraphEncoder",
    "GraphFeaturesExtractor",
    "GraphMlpExtractor",
    "GraphActorCriticPolicy",
]


class GraphEncoder(nn.Module):
    """Residual GATv2 trunk over a static topology with live edge features.

    Depth is a physical quantity here: one layer widens a node's receptive field
    by one hop, and congestion spills back one hop per simulator tick, so
    ``num_layers`` is the agent's spatial lookahead in timesteps.  Past three or
    four hops over-smoothing (embeddings collapsing toward a constant) costs more
    than the extra range buys, which is why every layer is residual + normalised.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim

        self.node_embed = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.edge_embed = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.SiLU())
        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=hidden_dim,
                add_self_loops=False,
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.activation = nn.SiLU()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(node_embeddings, edge_embeddings)``."""
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = h + self.activation(norm(conv(h, edge_index, e)))
        return h, e


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    """Unpack the flat observation into a graph, encode it, emit latents.

    The output packs a per-edge latent block and a pooled graph-level block, which
    :class:`GraphMlpExtractor` splits again for the two heads.  Batching is done by
    hand rather than through ``torch_geometric.data.Batch``: the topology never
    changes, so replicating ``edge_index`` with a node offset per batch element is
    both cheaper and avoids rebuilding ``Data`` objects on every rollout step.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        graph_spec: GraphSpec,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        edge_latent_dim: int = 16,
    ) -> None:
        features_dim = graph_spec.num_edges * edge_latent_dim + 2 * hidden_dim
        super().__init__(observation_space, features_dim)

        self.spec = graph_spec
        self.hidden_dim = hidden_dim
        self.edge_latent_dim = edge_latent_dim
        self.encoder = GraphEncoder(
            graph_spec.node_dim, graph_spec.edge_dim, hidden_dim, num_layers, heads
        )
        # Weight-shared decoder: identical parameters for every edge, so the map
        # from graph state to edge latents is permutation-equivariant by construction.
        self.edge_decoder = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_latent_dim),
        )
        self.register_buffer(
            "edge_index", torch.as_tensor(graph_spec.edge_index, dtype=torch.long)
        )

    def _batched_edge_index(self, batch_size: int) -> torch.Tensor:
        offsets = (
            torch.arange(batch_size, device=self.edge_index.device) * self.spec.num_nodes
        ).view(batch_size, 1, 1)
        return (self.edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode a batch of flat observations into edge and pooled latents."""
        spec = self.spec
        batch = observations.shape[0]
        split = spec.num_nodes * spec.node_dim

        x = observations[:, :split].reshape(batch * spec.num_nodes, spec.node_dim)
        edge_attr = observations[:, split:].reshape(
            batch * spec.num_edges, spec.edge_dim
        )
        edge_index = self._batched_edge_index(batch)

        h, e = self.encoder(x, edge_index, edge_attr)
        z = self.edge_decoder(
            torch.cat([h[edge_index[0]], h[edge_index[1]], e], dim=-1)
        ).reshape(batch, spec.num_edges * self.edge_latent_dim)

        nodes = h.reshape(batch, spec.num_nodes, self.hidden_dim)
        # Mean carries average load, max carries the bottleneck; the critic needs both.
        pooled = torch.cat([nodes.mean(dim=1), nodes.amax(dim=1)], dim=-1)
        return torch.cat([z, pooled], dim=-1)


class GraphMlpExtractor(nn.Module):
    """Split the trunk output into an equivariant actor and an invariant critic."""

    def __init__(
        self,
        num_edges: int,
        edge_latent_dim: int,
        pooled_dim: int,
        vf_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.num_edges = num_edges
        self.edge_latent_dim = edge_latent_dim
        self.edge_split = num_edges * edge_latent_dim

        self.actor_head = nn.Linear(edge_latent_dim, 1)
        self.critic_head = nn.Sequential(
            nn.Linear(pooled_dim, vf_hidden),
            nn.SiLU(),
            nn.Linear(vf_hidden, vf_hidden),
            nn.SiLU(),
        )
        self.latent_dim_pi = num_edges
        self.latent_dim_vf = vf_hidden

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """One routing logit per edge, from the weight-shared decoder."""
        edges = features[:, : self.edge_split].reshape(
            features.shape[0], self.num_edges, self.edge_latent_dim
        )
        return self.actor_head(edges).squeeze(-1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Graph-level latent for the value head, from the pooled block."""
        return self.critic_head(features[:, self.edge_split :])

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(latent_pi, latent_vf)``."""
        return self.forward_actor(features), self.forward_critic(features)


class GraphActorCriticPolicy(ActorCriticPolicy):
    """PPO policy whose actor is a shared per-edge decoder rather than a dense head."""

    def __init__(self, *args, vf_hidden: int = 128, **kwargs) -> None:
        self.vf_hidden = vf_hidden
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        extractor = self.features_extractor
        self.mlp_extractor = GraphMlpExtractor(
            num_edges=extractor.spec.num_edges,
            edge_latent_dim=extractor.edge_latent_dim,
            pooled_dim=2 * extractor.hidden_dim,
            vf_hidden=self.vf_hidden,
        )

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # ``actor_head`` already produced one logit per edge, so SB3's default
        # Linear(latent_pi, action_dim) would be a dense mixing layer that breaks
        # equivariance.  Drop it and rebuild the optimizer over the new parameters.
        if self.mlp_extractor.latent_dim_pi == int(np.prod(self.action_space.shape)):
            nn.init.orthogonal_(self.mlp_extractor.actor_head.weight, gain=0.01)
            nn.init.zeros_(self.mlp_extractor.actor_head.bias)
            self.action_net = nn.Identity()
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )

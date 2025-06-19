import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool, GlobalAttention, Set2Set, TransformerConv
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
from .base_qnet import BaseQNetwork


class ComplexQNetwork(BaseQNetwork):
    def __init__(self, model):
        super().__init__(model)
        self.device = self.model.device
        embed_dim = self.model.embed_dim
        self.set2set = Set2Set(embed_dim, processing_steps=3)
        
        self.linear_no_tau = torch.nn.Linear(3 * embed_dim, embed_dim)
        self.projection_no_tau = torch.nn.Linear(embed_dim, 1)

        self.linear_with_tau = torch.nn.Linear(4 * embed_dim, embed_dim)
        self.projection_with_tau = torch.nn.Linear(embed_dim, 1)
        self.to(self.device)

    def compute_q_values(self, node_embeddings, raw_states, batch):
        graph_emb = self.set2set(node_embeddings, batch)
        graph_emb = graph_emb[batch] # broadcast to [N, embed_dim]

        taus = [state["tau"] if state["tau"] is not None else -1 for state in raw_states]  # Use -1 as placeholder for None
        mask_state_has_tau = torch.tensor([t != -1 for t in taus], dtype=torch.bool, device=self.device) # compute mask for states with tau
        mask_state_has_tau_per_node = mask_state_has_tau[batch]  # broadcast to [N]
        taus = torch.tensor([t if t != -1 else 0 for t in taus], dtype=torch.long, device=self.device) # replace -1 with 0 so every state has a tau

        node_counts = torch.bincount(batch) # get index of each graphs first node
        node_starts = torch.cat([node_embeddings.new_zeros(1, dtype=torch.long), node_counts.cumsum(0)[:-1]])  # shape: [batch_size]

        taus = node_starts + taus  # convert to global indices, shape: [B]
        taus = taus[batch]  # broadcast to [N]

        tau_embeddings_per_node = node_embeddings[taus] # get tau embeddings
        
        h_tau = torch.cat([node_embeddings, tau_embeddings_per_node, graph_emb], dim=-1)
        h_no_tau = torch.cat([node_embeddings, graph_emb], dim=-1)

        h_tau = F.relu(self.linear_with_tau(h_tau))
        h_no_tau = F.relu(self.linear_no_tau(h_no_tau))

        out_tau = self.projection_with_tau(h_tau)  # shape: [N, 1]
        out_no_tau = self.projection_no_tau(h_no_tau)  # shape: [N, 1]

        out = torch.where(mask_state_has_tau_per_node.unsqueeze(-1), out_tau, out_no_tau) # Select output based on whether tau is present
        return out.squeeze(-1)  # shape: [N]


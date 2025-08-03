import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from ..base_gnn import BaseGNN


class GlobalGNN(BaseGNN):
    """
    Custom "Global GNN" implementation for the Q-network.
    This architecture uses a global convolutional layer to aggregate information,
    i.e. in each layer, each node receives messages from all other nodes in the graph.
    To enable communication between all nodes, we compute the 'full_edge_index', i.e. the edge index
    of a complete graph. To keep track of which nodes are neighbors, we also compute an 'is_neighbor' mask.

    Number of parameters: 3 * num_layers * embed_dim^2 + (4+num_layers) * embed_dim
    """

    def __init__(self, graph_size_training, embed_dim=64, num_layers=4, **kwargs):
        super().__init__(embed_dim, **kwargs)
        self.input_layer = GlobalConv(in_channels=3, hidden_channels=embed_dim)
        self.layers = nn.ModuleList(
            [
                GlobalConv(in_channels=embed_dim, hidden_channels=embed_dim)
                for _ in range(num_layers)
            ]
        )
        self.to(self.device)

    def prepare_batch(self, raw_states):
        xs = []
        adjacency_matrices = []

        for state in raw_states:

            data = state["graph_data"]
            n_nodes = data.num_nodes

            # Build node features
            sigma_tensor = torch.tensor(
                state["sigma"], dtype=torch.float32, device=self.device
            ).view(-1, 1)
            tau_tensor = torch.zeros(
                (n_nodes, 1), dtype=torch.float32, device=self.device
            )
            if state["tau"] is not None:
                tau_tensor[state["tau"]] = 1
            l_tensor = torch.full(
                (n_nodes, 1),
                fill_value=state.get("edges_left", 0),
                dtype=torch.float32,
                device=self.device,
            )

            # Concatenate node features
            xs.append(torch.cat([sigma_tensor, tau_tensor, l_tensor], dim=1))

            # Move edge_index to device
            data.edge_index = data.edge_index.to(self.device)

            # Convert edge_index to adjacency matrix
            adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=n_nodes).squeeze(0)
            adj_matrix.fill_diagonal_(0)  # Remove self-loops
            adjacency_matrices.append(adj_matrix)

        # Stack node features and adjacency matrices
        x_tensor = torch.stack(xs, dim=0)
        adjacency_tensor = torch.stack(adjacency_matrices, dim=0)

        return (x_tensor, adjacency_tensor)

    def forward_batch(self, batch):
        x_tensor, adjacency_tensor = batch
        B, n_nodes, _ = x_tensor.shape

        x_tensor = F.relu(self.input_layer(x_tensor, adjacency_tensor))
        for layer in self.layers:
            x_tensor = F.relu(layer(x_tensor, adjacency_tensor))

        batch_vector = torch.arange(B, device=self.device).repeat_interleave(n_nodes)
        return x_tensor.view(-1, x_tensor.size(-1)), batch_vector


class GlobalConv(nn.Module):
    """
    Custom global convolutional layer that aggregates information from all nodes in the graph.
    We use different linear layers for neighbor and non-neighbor messages,
    and a separate linear layer for self features.
    """

    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        # For neighbor messages
        self.lin_nbr = nn.Linear(in_channels, hidden_channels, bias=False)
        # For non-neighbor messages
        self.lin_non = nn.Linear(in_channels, hidden_channels, bias=False)
        # For self features
        self.lin_self = nn.Linear(in_channels, hidden_channels)

    def forward(self, x, adj):
        # x: [B, N, D], adj: [B, N, N]
        B, N, _ = x.shape
        I = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        num_non = (N - 1 - deg).clamp(min=1)

        nbr_mask = adj / deg
        non_mask = (1 - adj - I) / num_non

        x_self = self.lin_self(x)
        x_nbr = self.lin_nbr(x)
        x_non = self.lin_non(x)

        agg_nbr = torch.bmm(nbr_mask, x_nbr)
        agg_non = torch.bmm(non_mask, x_non)

        return x_self + agg_nbr + agg_non

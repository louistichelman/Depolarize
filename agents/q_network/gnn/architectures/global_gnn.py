import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
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
        self.full_edge_index = self._precompute_full_edge_index(graph_size_training)
        self.to(self.device)

    def _precompute_full_edge_index(self, graph_size_training):
        """
        Precompute the edge index for a complete graph of size `graph_size_training`.
        This speeds up training, since all graphs in training have the same size.
        """
        nodes = torch.arange(graph_size_training, device=self.device)
        row, col = torch.combinations(nodes, r=2, with_replacement=False).T
        row, col = torch.cat([row, col]), torch.cat([col, row])
        return row, col

    def prepare_batch(self, raw_states):
        training = len(raw_states) > 1
        data_list = []
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
            data.x = torch.cat([sigma_tensor, tau_tensor, l_tensor], dim=1)

            # Move edge_index to device
            data.edge_index = data.edge_index.to(self.device)

            # Create full edge index (i.e. edge index for a complete graph)
            # We need the full edge index to compute messages from all nodes.
            # If training, use precomputed full_edge_index.
            if training:
                row, col = self.full_edge_index
            else:
                row, col = torch.combinations(
                    torch.arange(n_nodes, device=self.device),
                    r=2,
                    with_replacement=False,
                ).T
                row, col = torch.cat([row, col]), torch.cat([col, row])
            full_edge_index = torch.stack([row, col], dim=0)

            # Build is_neighbor mask
            adj = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=self.device)
            adj[data.edge_index[0], data.edge_index[1]] = True
            is_neighbor = adj[row, col].unsqueeze(1)  # shape: [E, 1]

            # Store in data
            data.full_edge_index = full_edge_index  # [2, num_edges]
            data.is_neighbor = is_neighbor  # [num_edges, 1]
            data_list.append(data)

        return Batch.from_data_list(data_list).to(self.device)

    def forward_batch(self, batch):
        x = batch.x
        full_edge_index = batch.full_edge_index
        is_neighbor = batch.is_neighbor
        x = F.relu(self.input_layer(x, full_edge_index, is_neighbor))
        for layer in self.layers:
            x = F.relu(layer(x, full_edge_index, is_neighbor))
        return x, batch.batch


class GlobalConv(MessagePassing):
    """
    Custom global convolutional layer that aggregates information from all nodes in the graph.
    We use different linear layers for neighbor and non-neighbor messages,
    and a separate linear layer for self features.
    """

    def __init__(self, in_channels, hidden_channels):
        super().__init__(aggr="mean")

        # For neighbor messages
        self.lin_nbr = nn.Linear(in_channels, hidden_channels, bias=False)
        # For non-neighbor messages
        self.lin_non = nn.Linear(in_channels, hidden_channels, bias=False)
        # For self features
        self.lin_self = nn.Linear(in_channels, hidden_channels)

    def forward(self, x, full_edge_index, is_neighbor):
        self._is_neighbor = is_neighbor  # cache for use in message()

        aggr = self.propagate(full_edge_index, x=x)

        return aggr + self.lin_self(x)

    def message(self, x_j):
        neighbor_msg = self.lin_nbr(x_j)
        non_neighbor_msg = self.lin_non(x_j)
        return neighbor_msg * self._is_neighbor + non_neighbor_msg * (
            ~self._is_neighbor
        )

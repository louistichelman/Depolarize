import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
from .base_architecture import BaseArchitecture

    
class GlobalGNN(BaseArchitecture):
    def __init__(self, embed_dim=60, number_of_layers = 3, **kwargs):
        super().__init__(embed_dim, **kwargs)
        self.input_layer = GlobalConv(in_channels=3, hidden_channels=embed_dim)
        self.layers = nn.ModuleList([GlobalConv(in_channels=embed_dim, hidden_channels=embed_dim) for _ in range(number_of_layers)])
        self.to(self.device)

    def prepare_batch(self, raw_states):
        data_list = []
        for state in raw_states: 
            G, sigma, tau, l = state

            data = from_networkx(G)
            n_nodes = data.num_nodes

            # --- Build node features ---
            sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=self.device).view(-1, 1)
            tau_tensor = torch.zeros((n_nodes, 1), dtype=torch.float32, device=self.device)
            if tau is not None:
                tau_tensor[tau] = 1.0
            l_tensor = torch.full((n_nodes, 1), fill_value=l, dtype=torch.float32, device=self.device)
            
            # --- Concatenate node features ---
            data.x = torch.cat([sigma_tensor, tau_tensor, l_tensor], dim=1)

            # --- Move edge_index to device ---
            data.edge_index = data.edge_index.to(self.device)

            # --- Create full edge index ---
            row, col = torch.combinations(torch.arange(n_nodes), r=2, with_replacement=False).T
            row, col = torch.cat([row, col]), torch.cat([col, row])  # undirected
            full_edge_index = torch.stack([row, col], dim=0).to(self.device)

            # --- Build is_neighbor mask ---
            adj = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=self.device)
            adj[data.edge_index[0], data.edge_index[1]] = True
            is_neighbor = adj[row, col].unsqueeze(1)  # shape: [E, 1]

            # --- Store in data ---
            data.full_edge_index = full_edge_index  # [2, num_edges]
            data.is_neighbor = is_neighbor          # [num_edges, 1]
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
    def __init__(self, in_channels, hidden_channels):
        super().__init__(aggr='mean') 
        self.lin_nbr = nn.Linear(in_channels, hidden_channels)      # For neighbor messages
        self.lin_non = nn.Linear(in_channels, hidden_channels)      # For non-neighbor messages
        self.lin_self = nn.Linear(in_channels, hidden_channels)     # For self features

    def forward(self, x, full_edge_index, is_neighbor):
        self._is_neighbor = is_neighbor  # cache for use in message()

        aggr = self.propagate(full_edge_index, x=x)

        return aggr + self.lin_self(x)


    def message(self, x_j):
        neighbor_msg = self.lin_nbr(x_j)
        non_neighbor_msg = self.lin_non(x_j)
        return neighbor_msg * self._is_neighbor + non_neighbor_msg * (~self._is_neighbor)




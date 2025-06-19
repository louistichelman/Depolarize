import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from .base_architecture import BaseArchitecture
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx

class GraphSAGE(BaseArchitecture):
    def __init__(self, embed_dim=64, num_layers = 4, **kwargs):
        super().__init__(embed_dim, **kwargs)
        self.input_layer = SAGEConv(3, embed_dim)
        self.layers = torch.nn.ModuleList([SAGEConv(embed_dim, embed_dim) for _ in range(num_layers)])
        self.to(self.device)

    def prepare_batch(self, raw_states):
        data_list = []
        for state in raw_states: 
            
            data = state["graph_data"]
            n_nodes = data.num_nodes
            
            # --- Build node features ---
            sigma_tensor = torch.tensor(state["sigma"], dtype=torch.float32, device=self.device).view(-1, 1)
            tau_tensor = torch.zeros((n_nodes, 1), dtype=torch.float32, device=self.device)
            if state["tau"] is not None:
                tau_tensor[state["tau"]] = 1.0
            l_tensor = torch.full((n_nodes, 1), fill_value=state["edges_left"], dtype=torch.float32, device=self.device)
            
            # --- Concatenate node features ---
            data.x = torch.cat([sigma_tensor, tau_tensor, l_tensor], dim=1)

            # --- Move edge_index to device ---
            data.edge_index = data.edge_index.to(self.device)

            data_list.append(data)

        return Batch.from_data_list(data_list).to(self.device)

    def forward_batch(self, batch):
        x, edge_index = batch.x, batch.edge_index

        x = F.relu(self.input_layer(x, edge_index))
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        return x, batch.batch
import torch
from torch_geometric.nn import GATConv, Linear
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x



class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # ('state', 's,a', 'action')
        # ('state', 's,bg', 'background')
        # ('state', 's,t', 't')
        # ('action', 'a,bg', 'background')
        # ('action', 'a,t', 't')
        # ('background', 'bg,t', 't')

        num_layers = len(hidden_channels)
        self.convs = torch.nn.ModuleList()
        
        for layer in range(num_layers):
            conv_dict = {}
            key_l = ['state', 'action', 'background', 't']
            for k1 in key_l:
                for k2 in key_l:
                    if k1 == k2:
                        continue
                    conv_dict[(k1, f'{k1}, {k2}', k2)] = GATConv((-1, -1), hidden_channels[layer], add_self_loops=False)
                    # ('state', 's,a', 'action'): GCNConv(-1, hidden_channels),
                    # ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
                    # ('paper', 'rev_writes', 'author'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels[-1], out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['state'])

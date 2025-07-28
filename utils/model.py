import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, HANConv, GCNConv, GATConv, SAGEConv

import warnings
warnings.filterwarnings("ignore")

class MLPRegression(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.3, activation='relu'):
        super().__init__()

        self.dropout = dropout
        self.activation = self._get_activation_fn(activation)

        layers = []
        input_dim = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def _get_activation_fn(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class GCNRegression(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.3, mlp_layers=1):
        super().__init__()
        assert num_layers >= 1

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.mlps = nn.Sequential()
        for i in range(mlp_layers - 1):
            self.mlps.add_module(f'linear_{i}', nn.Linear(hidden_dim, hidden_dim))
            self.mlps.add_module(f'relu_{i}', nn.ReLU())
            self.mlps.add_module(f'dropout_{i}', nn.Dropout(dropout))

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.mlps(x)
        return self.output(x).squeeze(-1)  # [N]

class GATRegression(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, heads=2, dropout=0.3):
        super(GATRegression, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        # 첫 번째 GAT layer: heads > 1, concat = True
        self.layers.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        )

        # 중간 GAT layers (concat=False, heads=1)
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=True, dropout=dropout)
            )

        # 마지막 GAT layer: heads=1, concat=False to keep [N, hidden_channels]
        self.layers.append(
            GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        )

        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return self.linear(x).squeeze(-1)  # [N]

class GraphSAGERegression(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.3):
        super().__init__()
        assert num_layers >= 1

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return self.out(x).squeeze(-1)  # [N]

class RGCNRegression(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_relations, dropout=0.3):
        super().__init__()
        assert num_layers >= 1

        self.dropout = nn.Dropout(dropout)
        self.convs = ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))

        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_type):
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)
        return self.out(x).squeeze(-1)  # [N]

class HANRegression(nn.Module):
    def __init__(
        self,
        metadata,
        in_channels,
        hidden_channels,
        heads=1,
        dropout=0.3,
        num_layers=1,      # HANConv layer 수
        mlp_layers=1       # MLP 계층 수
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.hidden_channels = hidden_channels

        self.han_layers = nn.ModuleList()

        for i in range(num_layers):
            layer_in = in_channels if i == 0 else hidden_channels * heads
            self.han_layers.append(
                HANConv(
                    in_channels=layer_in,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=heads
                )
            )

        # MLP head after HAN
        mlp = []
        dim = hidden_channels * heads
        for _ in range(mlp_layers - 1):
            mlp.append(nn.Linear(dim, dim))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout))
        mlp.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x_dict, edge_index_dict):
        for han in self.han_layers:
            x_dict = han(x_dict, edge_index_dict)
        x_job = self.dropout(x_dict['job'])
        return self.mlp(x_job).squeeze(-1)  # [N]

class MuxGNNLayer(nn.Module):
    def __init__(
            self,
            gnn_type,
            relations,
            in_dim,
            out_dim,
            dim_a,
            dropout=0.,
            activation=None,
            use_norm=False,
    ):
        super(MuxGNNLayer, self).__init__()
        self.gnn_type = gnn_type
        self.relations = relations
        self.num_relations = len(self.relations)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.act_str = activation

        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(self.act_str)

        if self.gnn_type == 'gcn':
            self.gnn = GraphConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                norm='both',
                weight=True,
                bias=True,
                activation=self.activation,
                allow_zero_in_degree=True
            )
        elif self.gnn_type == 'gat':
            self.gnn = GATConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                num_heads=2,
                feat_drop=dropout,
                residual=False,
                activation=self.activation,
                allow_zero_in_degree=True
            )
        elif self.gnn_type == 'gin':
            self.gnn = GINConv(
                apply_func=nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    self.dropout,
                    self.activation,
                    nn.Linear(out_dim, out_dim),
                    self.dropout,
                    self.activation,
                ),
                aggregator_type='sum',
            )
        else:
            raise ValueError('Invalid GNN type.')

        self.attention = SemanticAttention(self.num_relations, self.out_dim, self.dim_a)
        self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=True) if use_norm else None

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, block, node_feat, return_attn=False):
        num_dst_nodes = block.number_of_dst_nodes()
        h = torch.empty(self.num_relations, num_dst_nodes, self.out_dim, device=block.device)
        with block.local_scope():
            for i, graph_layer in enumerate(self.relations):
                rel_graph = block['node', graph_layer, 'node']

                h_out = self.gnn(rel_graph, node_feat[:, i]).squeeze()
                if self.gnn_type == 'gat':
                    h_out = h_out.sum(dim=1)

                h[i] = h_out

        if self.norm:
            h = self.norm(h)

        return self.attention(h, return_attn=return_attn)

class SemanticAttention(nn.Module):
    def __init__(self, num_relations, in_dim, dim_a, dropout=0.):
        super(SemanticAttention, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.dim_a, self.num_relations)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, h, return_attn=False):
        # Shape of h: (num_relations, batch_size, dim)
        attention = F.softmax(
            torch.matmul(
                torch.tanh(
                    torch.matmul(h, self.weights_s1)
                ),
                self.weights_s2
            ),
            dim=0
        ).permute(1, 0, 2)

        attention = self.dropout(attention)

        # Output shape: (batch_size, num_relations, dim)
        h = torch.matmul(attention, h.permute(1, 0, 2))

        return h, attention if return_attn else None

class SemanticAttentionBatched(nn.Module):
    def __init__(self, num_relations, in_dim, dim_a, out_dim=1, dropout=0.):
        super(SemanticAttentionBatched, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.dim_a, self.out_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, graph, h, batch_size=512):
        # Shape of input h: (num_relations, num_nodes, dim)
        # Output shape: (num_nodes, dim)
        graph.ndata['h'] = torch.zeros(graph.num_nodes(), h.size(-1), device=graph.device)

        node_loader = DataLoader(
            graph.nodes(),
            batch_size=batch_size,
            shuffle=False,
        )

        for node_batch in node_loader:
            h_batch = h[:, node_batch, :]

            attention = F.softmax(
                torch.matmul(
                    torch.tanh(
                        torch.matmul(h_batch, self.weights_s1)
                    ),
                    self.weights_s2
                ),
                dim=0
            ).squeeze()

            attention = self.dropout(attention)

            try:
                graph.ndata['h'][node_batch] = torch.einsum('rb,rbd->bd', attention, h_batch)
            except RuntimeError:
                graph.ndata['h'][node_batch] = torch.einsum('rb,rbd->bd', attention.unsqueeze(1), h_batch)

        return graph.ndata.pop('h')
    
class MuxGNNRegression(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, gnn_type='gcn', dim_a=64, dropout=0.3, num_layers=2):
        super().__init__()
        self.num_relations = num_relations
        self.out_channels = out_channels
        self.gnn_type = gnn_type
        self.dim_a = dim_a
        self.num_layers = num_layers

        # 다중 레이어 GNN 구성 (layer_list[layer_idx][relation_idx])
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer = nn.ModuleList()
            input_dim = in_channels if layer_idx == 0 else out_channels
            for _ in range(num_relations):
                if gnn_type == 'gcn':
                    layer.append(GCNConv(input_dim, out_channels))
                elif gnn_type == 'gat':
                    layer.append(GATConv(input_dim, out_channels, heads=1, concat=False))
                elif gnn_type == 'sage':
                    layer.append(SAGEConv(input_dim, out_channels))
                else:
                    raise ValueError(f"Unsupported gnn_type: {gnn_type}")
            self.layers.append(layer)

        # attention weights per layer
        self.W1 = nn.ParameterList([
            nn.Parameter(torch.randn(num_relations, out_channels, dim_a)) for _ in range(num_layers)
        ])
        self.W2 = nn.ParameterList([
            nn.Parameter(torch.randn(num_relations, dim_a, 1)) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.predictor = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index_list):
        # x: [N, F]
        for layer_idx in range(self.num_layers):
            h_list = []
            for i in range(self.num_relations):
                h_i = self.layers[layer_idx][i](x, edge_index_list[i])  # [N, D]
                h_i = F.relu(h_i)
                h_list.append(h_i)
            H = torch.stack(h_list, dim=1)  # [N, R, D]

            # Semantic attention per relation
            attn = []
            for r in range(self.num_relations):
                h_r = H[:, r, :]             # [N, D]
                w1 = self.W1[layer_idx][r]  # [D, dim_a]
                w2 = self.W2[layer_idx][r]  # [dim_a, 1]

                a_r = torch.tanh(h_r @ w1)  # [N, dim_a]
                a_r = a_r @ w2              # [N, 1]
                attn.append(a_r)

            attn = torch.stack(attn, dim=1)         # [N, R, 1]
            attn = F.softmax(attn, dim=1)           # [N, R, 1]
            x = (H * attn).sum(dim=1)               # [N, D]
            x = self.dropout(x)

        return self.predictor(x).squeeze(-1)  # [N]
    
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        for conv in self.gcn_layers:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.linear(x)
    
class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, heads=2, num_layers=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))

        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.linear(x)

class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.out = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)
    
class RGCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_relations, num_layers=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(RGCNConv(in_channels, hidden_channels, num_relations))
        for _ in range(num_layers - 1):
            self.layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        for conv in self.layers:
            x = F.relu(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
    
class HANClassifier(nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels, num_classes, heads=1, dropout=0.3, num_layers=1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.han_layers = nn.ModuleList([
            HANConv(
                in_channels=in_channels if i == 0 else hidden_channels * heads,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=heads
            ) for i in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_channels * heads, num_classes)

    def forward(self, x_dict, edge_index_dict):
        for layer in self.han_layers:
            x_dict = layer(x_dict, edge_index_dict)
        x_job = self.dropout(x_dict['job'])
        return self.linear(x_job)

class MuxGNNClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, num_classes, gnn_type='gcn', dim_a=64, dropout=0.3, num_layers=2):
        super().__init__()
        self.num_relations = num_relations
        self.out_channels = out_channels
        self.gnn_type = gnn_type
        self.dim_a = dim_a
        self.num_layers = num_layers

        # GNN layers per relation per layer
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer = nn.ModuleList()
            input_dim = in_channels if layer_idx == 0 else out_channels
            for _ in range(num_relations):
                if gnn_type == 'gcn':
                    layer.append(GCNConv(input_dim, out_channels))
                elif gnn_type == 'gat':
                    layer.append(GATConv(input_dim, out_channels, heads=1, concat=False))
                elif gnn_type == 'sage':
                    layer.append(SAGEConv(input_dim, out_channels))
                else:
                    raise ValueError(f"Unsupported gnn_type: {gnn_type}")
            self.layers.append(layer)

        # attention parameters
        self.W1 = nn.ParameterList([
            nn.Parameter(torch.randn(num_relations, out_channels, dim_a)) for _ in range(num_layers)
        ])
        self.W2 = nn.ParameterList([
            nn.Parameter(torch.randn(num_relations, dim_a, 1)) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.predictor = nn.Linear(out_channels, num_classes)  # For classification

    def forward(self, x, edge_index_list):
        for layer_idx in range(self.num_layers):
            h_list = []
            for i in range(self.num_relations):
                h_i = self.layers[layer_idx][i](x, edge_index_list[i])
                h_i = F.relu(h_i)
                h_list.append(h_i)

            H = torch.stack(h_list, dim=1)  # [N, R, D]

            attn = []
            for r in range(self.num_relations):
                h_r = H[:, r, :]                 # [N, D]
                w1 = self.W1[layer_idx][r]       # [D, dim_a]
                w2 = self.W2[layer_idx][r]       # [dim_a, 1]

                a_r = torch.tanh(h_r @ w1)       # [N, dim_a]
                a_r = a_r @ w2                   # [N, 1]
                attn.append(a_r)

            attn = torch.stack(attn, dim=1)      # [N, R, 1]
            attn = F.softmax(attn, dim=1)        # [N, R, 1]
            x = (H * attn).sum(dim=1)            # [N, D]
            x = self.dropout(x)

        return self.predictor(x)                 # [N, num_classes]
    
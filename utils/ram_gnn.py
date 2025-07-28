import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

import warnings
warnings.filterwarnings("ignore")

class MultiplexLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, gnn_type='gcn', 
                 num_layers=2, dropout=0.3, activation='relu'):
        super().__init__()
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout
        self.act_fn = self._get_activation_fn(activation)

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            convs = nn.ModuleList()
            for _ in range(num_relations):
                input_dim = in_channels if layer_idx == 0 else out_channels
                conv = self._get_conv(gnn_type, input_dim, out_channels)
                convs.append(conv)
            self.layers.append(convs)

        self.dropout_layer = nn.Dropout(dropout)

    def _get_conv(self, gnn_type, in_dim, out_dim):
        if gnn_type == 'gcn':
            return GCNConv(in_dim, out_dim)
        elif gnn_type == 'gat':
            return GATConv(in_dim, out_dim, heads=1, concat=False)
        elif gnn_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        else:
            raise NotImplementedError(f"gnn_type '{gnn_type}' is not supported.")

    def _get_activation_fn(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'gelu':
            return nn.GELU()
        elif name is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x, edge_index_list):
        for convs in self.layers:
            h_list = []
            for i in range(self.num_relations):
                h = convs[i](x, edge_index_list[i])
                h = self.act_fn(h)
                h_list.append(h)
            H = torch.stack(h_list, dim=0)
            x = H.max(dim=0).values
            x = self.dropout_layer(x)
        return x  # [N, out_channels]

class MultiplexAttentionLayer(MultiplexLayer):
    def __init__(self, in_channels, out_channels, num_relations, gnn_type='gcn', 
                 num_layers=2, dropout=0.3, activation='relu'):
        super().__init__(in_channels, out_channels, num_relations, gnn_type, 
                         num_layers, dropout, activation)

        # attention MLP
        self.att_mlp = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index_list):
        for convs in self.layers:
            h_list = []
            attn_scores = []
            for i in range(self.num_relations):
                h = convs[i](x, edge_index_list[i])
                h = self.act_fn(h)
                h_list.append(h)
                attn_scores.append(self.att_mlp(h))
            H = torch.stack(h_list, dim=1)           # [N, R, D]
            A = torch.stack(attn_scores, dim=1)      # [N, R, 1]
            attn_weights = F.softmax(A, dim=1)
            x = (H * attn_weights).sum(dim=1)        # [N, D]
            x = self.dropout_layer(x)
        return x, attn_weights.squeeze(-1)

class RelationalMultiplexAttentionLayer(MultiplexLayer):
    def __init__(self, in_channels, out_channels, num_relations, gnn_type='gcn', 
                 num_layers=2, dropout=0.3, activation='relu'):
        super().__init__(in_channels, out_channels, num_relations, gnn_type, 
                         num_layers, dropout, activation)

        self.att_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_channels, 64),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            ) for _ in range(num_relations)
        ])

    def forward(self, x, edge_index_list):
        for convs in self.layers:
            h_list = []
            attn_logits = []

            for i in range(self.num_relations):
                h = convs[i](x, edge_index_list[i])
                h = self.act_fn(h)
                h = self.dropout_layer(h)
                h_list.append(h)
                attn_logits.append(self.att_mlp[i](h))

            H = torch.stack(h_list, dim=1)       
            A = torch.stack(attn_logits, dim=1)   
            attn_weights = F.softmax(A, dim=1)
            x = (H * attn_weights).sum(dim=1)   

        return x, attn_weights.squeeze(-1)

class MultiplexRegression(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, gnn_type='gcn', 
                 num_layers=2, dropout=0.3, activation='relu'):
        super().__init__()
        self.encoder = MultiplexLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            gnn_type=gnn_type,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )
        self.predictor = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index_list):
        h = self.encoder(x, edge_index_list)        # [N, out_channels]
        return self.predictor(h).squeeze(-1)

class A_MultiplexRegression(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, gnn_type='gcn', 
                 num_layers=2, dropout=0.3, activation='relu'):
        super().__init__()
        self.encoder = MultiplexAttentionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            gnn_type=gnn_type,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )
        self.predictor = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index_list):
        h, _ = self.encoder(x, edge_index_list)  # [N, out_channels]
        return self.predictor(h).squeeze(-1)

class RA_MultiplexRegression(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, gnn_type='gcn', num_layers=2, dropout=0.3, activation='relu'):
        super().__init__()
        self.encoder = RelationalMultiplexAttentionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            gnn_type=gnn_type,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )
        self.predictor = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index_list):
        h, attn = self.encoder(x, edge_index_list)

        return self.predictor(h).squeeze(-1)

class MultiplexClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, num_classes, num_layers=2, gnn_type='gcn', dropout=0.3):
        super().__init__()
        self.encoder = MultiplexLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            gnn_type=gnn_type,
            num_layers=num_layers,
            dropout=dropout
        )
        self.predictor = nn.Linear(out_channels, num_classes)  # 분류용 projection

    def forward(self, x, edge_index_list):
        h = self.encoder(x, edge_index_list)        # [N, out_channels]
        logits = self.predictor(h)                  # [N, num_classes]
        return logits  # softmax는 CrossEntropyLoss 안에서 적용됨

class A_MultiplexClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, num_classes, num_layers=2, gnn_type='gcn', dropout=0.3):
        super().__init__()
        self.encoder = MultiplexAttentionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            gnn_type=gnn_type,
            num_layers=num_layers,
            dropout=dropout
        )
        self.predictor = nn.Linear(out_channels, num_classes)

    def forward(self, x, edge_index_list):
        h, attn = self.encoder(x, edge_index_list)  # h: [N, out_channels]
        logits = self.predictor(h)                  # [N, num_classes]
        return logits  # CrossEntropyLoss를 쓸 경우 softmax는 생략

class RA_MultiplexClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, num_classes, num_layers=2, gnn_type='gcn', dropout=0.3):
        super().__init__()
        self.encoder = RelationalMultiplexAttentionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            gnn_type=gnn_type,
            num_layers=num_layers,
            dropout=dropout
        )
        self.predictor = nn.Linear(out_channels, num_classes)

    def forward(self, x, edge_index_list):
        h, attn = self.encoder(x, edge_index_list)  # h: [N, out_channels]
        logits = self.predictor(h)                  # [N, num_classes]
        return logits

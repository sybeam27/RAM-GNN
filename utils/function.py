import torch
import pandas as pd
import numpy as np
import random

import torch.nn.functional as F
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ram_gnn import MultiplexRegression, A_MultiplexRegression, RA_MultiplexRegression
from ram_gnn import MultiplexClassifier, A_MultiplexClassifier, RA_MultiplexClassifier
from model import MLPRegression, GCNRegression, GATRegression, GraphSAGERegression, RGCNRegression, HANRegression, MuxGNNRegression
from model import MLPClassifier, GCNClassifier, GATClassifier, GraphSAGEClassifier, RGCNClassifier, HANClassifier, MuxGNNClassifier

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_soc_code(code):
    return code.split('.')[0] if isinstance(code, str) else str(code).split('.')[0]

def create_hetero_graph(node_df, edge_df, feature_cols):
    soc2idx = {soc: i for i, soc in enumerate(node_df['SOC_CODE'])}
    num_nodes = len(node_df)

    features = node_df[feature_cols].values
    features = StandardScaler().fit_transform(features)

    data = HeteroData()
    data['job'].x = torch.tensor(features, dtype=torch.float)
    data['job'].y = torch.tensor(node_df['AIOE'].values, dtype=torch.float).unsqueeze(-1)
    data['job'].node_id = torch.arange(num_nodes)

    for rel in edge_df['relation'].unique():
        rel_df = edge_df[edge_df['relation'] == rel]
        valid_rel_df = rel_df[
            rel_df['source'].isin(soc2idx) & rel_df['target'].isin(soc2idx)
        ].copy()

        valid_rel_df['src_idx'] = valid_rel_df['source'].map(soc2idx)
        valid_rel_df['tgt_idx'] = valid_rel_df['target'].map(soc2idx)

        valid_rel_df['edge'] = list(zip(
            valid_rel_df[['src_idx', 'tgt_idx']].min(axis=1),
            valid_rel_df[['src_idx', 'tgt_idx']].max(axis=1)
        ))

        unique_edges = valid_rel_df['edge'].drop_duplicates()
        unique_edges = [e for e in unique_edges if e[0] != e[1]]

        if len(unique_edges) > 0:
            src, tgt = zip(*unique_edges)
            edge_index = torch.tensor([src, tgt], dtype=torch.long)
            data['job', rel, 'job'].edge_index = edge_index

    return data

def create_classification_hetero_graph(node_df, edge_df, feature_cols, mode='binary'):
    soc2idx = {soc: i for i, soc in enumerate(node_df['SOC_CODE'])}
    num_nodes = len(node_df)

    features = node_df[feature_cols].values
    features = StandardScaler().fit_transform(features)

    if mode == 'binary':
        threshold = 0 
        node_df['AIOE_Class'] = (node_df['AIOE'] > threshold).astype(int)
    elif mode == 'three_class':
        mean = node_df['AIOE'].mean()
        std = node_df['AIOE'].std()
        def label_aioe(val):
            if val <= mean - 0.5 * std:
                return 0 
            elif val >= mean + 0.5 * std:
                return 2  
            else:
                return 1 
        node_df['AIOE_Class'] = node_df['AIOE'].apply(label_aioe)
    elif mode == 'anomaly':
        node_df['AIOE_Class'] = node_df['Anomaly_label']
    else:
        raise ValueError("mode must be 'binary' or 'three_class'")

    data = HeteroData()
    data['job'].x = torch.tensor(features, dtype=torch.float)
    data['job'].y = torch.tensor(node_df['AIOE_Class'].values, dtype=torch.long)
    data['job'].node_id = torch.arange(num_nodes)

    for rel in edge_df['relation'].unique():
        rel_df = edge_df[edge_df['relation'] == rel]
        valid_rel_df = rel_df[
            rel_df['source'].isin(soc2idx) & rel_df['target'].isin(soc2idx)
        ].copy()

        valid_rel_df['src_idx'] = valid_rel_df['source'].map(soc2idx)
        valid_rel_df['tgt_idx'] = valid_rel_df['target'].map(soc2idx)

        valid_rel_df['edge'] = list(zip(
            valid_rel_df[['src_idx', 'tgt_idx']].min(axis=1),
            valid_rel_df[['src_idx', 'tgt_idx']].max(axis=1)
        ))

        unique_edges = valid_rel_df['edge'].drop_duplicates()
        unique_edges = [e for e in unique_edges if e[0] != e[1]]

        if len(unique_edges) > 0:
            src, tgt = zip(*unique_edges)
            edge_index = torch.tensor([src, tgt], dtype=torch.long)
            data['job', rel, 'job'].edge_index = edge_index

    return data

def split_masks(data, split_ratio=(0.6, 0.2, 0.2), seed=None):
    num_nodes = data['job'].num_nodes
    idx = list(range(num_nodes))

    train_idx, test_idx = train_test_split(idx, test_size=split_ratio[2], random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=split_ratio[1]/(split_ratio[0] + split_ratio[1]), random_state=seed)

    data['job'].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data['job'].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data['job'].test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data['job'].train_mask[train_idx] = True
    data['job'].val_mask[val_idx] = True
    data['job'].test_mask[test_idx] = True

    return data

def split_masks_stratified(data, label_name='Anomaly_label', split_ratio=(0.6, 0.2, 0.2), seed=None):
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1"

    num_nodes = data['job'].num_nodes
    labels = data['job'].y.cpu().numpy()  
    indices = list(range(num_nodes))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=split_ratio[2], stratify=labels, random_state=seed
    )

    train_val_labels = [labels[i] for i in train_val_idx]

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]),
        stratify=train_val_labels,
        random_state=seed
    )

    data['job'].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data['job'].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data['job'].test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data['job'].train_mask[train_idx] = True
    data['job'].val_mask[val_idx] = True
    data['job'].test_mask[test_idx] = True

    return data

def deduplicate_edges(edge_index):
    # edge_index: [2, num_edges]
    edge_tuples = edge_index.t().tolist()
    unique_edges = list(set(map(tuple, edge_tuples)))
    return torch.tensor(unique_edges, dtype=torch.long).t()

def load_data(graph_path):
    graph_data = torch.load(graph_path, weights_only=False)
    graph_data = split_masks(graph_data)
    return graph_data

def get_model_dict(graph_data, hidden_dim, num_layers, dropout):
    in_dim = graph_data['job'].x.shape[1]
    num_rel = len(graph_data.edge_types)

    return {
        "Linear": lambda: LinearRegression(),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=42), 
        "XGBoost": lambda: XGBRegressor(
            n_estimators=100, max_depth=6, n_jobs=1,
            tree_method='hist', verbosity=0,
            objective='reg:squarederror', random_state=42
        ),

        "MLP": lambda: MLPRegression(in_dim, hidden_dim, num_layers, dropout),
        "GCN": lambda: GCNRegression(in_dim, hidden_dim, num_layers, dropout),
        "GAT": lambda: GATRegression(in_dim, hidden_dim, num_layers, heads=2, dropout=dropout),
        "GraphSAGE": lambda: GraphSAGERegression(in_dim, hidden_dim, num_layers, dropout),
        "RGCN": lambda: RGCNRegression(in_dim, hidden_dim, num_layers, num_rel),
        "HAN": lambda: HANRegression(graph_data.metadata(), in_dim, hidden_dim, heads=1, num_layers=num_layers, dropout=dropout),
        "MuxGNN": lambda: MuxGNNRegression(in_dim, hidden_dim, num_relations=num_rel, gnn_type='gcn', num_layers=num_layers, dropout=dropout),  
        "M-GNN": lambda: MultiplexRegression(in_dim, hidden_dim, num_rel, 'gcn', num_layers, dropout),
        "MA-GNN": lambda: A_MultiplexRegression(in_dim, hidden_dim, num_rel, 'gcn', num_layers, dropout),
        "RAM-GNN": lambda: RA_MultiplexRegression(in_dim, hidden_dim, num_rel, 'gcn', num_layers, dropout),
    }

def prediction_train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    # HAN
    if isinstance(model, HANRegression):
        out = model(data.x_dict, data.edge_index_dict).squeeze(-1)

    # Our
    elif isinstance(model, (MultiplexRegression, A_MultiplexRegression, RA_MultiplexRegression)):
        edge_index_list = [ei for ei in data.edge_index_dict.values()]
        out = model(data['job'].x, edge_index_list).squeeze(-1)

    # RGCN
    elif isinstance(model, RGCNRegression):
        edge_index = []
        edge_type = []
        rel_map = {rel: i for i, rel in enumerate(data.edge_types)}
        for rel, ei in data.edge_index_dict.items():
            edge_index.append(ei)
            edge_type.append(torch.full((ei.size(1),), rel_map[rel], dtype=torch.long))
        edge_index = torch.cat(edge_index, dim=1)
        edge_type = torch.cat(edge_type, dim=0)
        out = model(data['job'].x, edge_index, edge_type).squeeze(-1)

    # GCN / GAT
    elif hasattr(model, 'forward') and 'edge_index' in model.forward.__code__.co_varnames:
        # homogeneous edge 구성
        def deduplicate_edges(edge_index):
            # edge_index: [2, num_edges]
            edge_tuples = edge_index.t().tolist()
            unique_edges = list(set(map(tuple, edge_tuples)))
            return torch.tensor(unique_edges, dtype=torch.long).t()
        raw_edge_index = torch.cat([ei for _, ei in data.edge_index_dict.items()], dim=1)
        edge_index = deduplicate_edges(raw_edge_index)
        out = model(data['job'].x, edge_index).squeeze(-1)

    elif isinstance(model, (MultiplexRegression, A_MultiplexRegression, RA_MultiplexRegression, MuxGNNRegression)):
        edge_index_list = [ei for ei in data.edge_index_dict.values()]
        out = model(data['job'].x, edge_index_list).squeeze(-1)

    # MLP
    else:
        out = model(data['job'].x).squeeze(-1)

    mask = data['job'].train_mask
    loss = F.mse_loss(out[mask], data['job'].y[mask].squeeze(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def prediction_evaluate(model, data, mask):
    model.eval()

    if isinstance(model, HANRegression):
        out = model(data.x_dict, data.edge_index_dict).squeeze(-1)

    elif isinstance(model, RGCNRegression):
        edge_index = []
        edge_type = []
        rel_map = {rel: i for i, rel in enumerate(data.edge_types)}
        for rel, ei in data.edge_index_dict.items():
            edge_index.append(ei)
            edge_type.append(torch.full((ei.size(1),), rel_map[rel], dtype=torch.long))
        edge_index = torch.cat(edge_index, dim=1)
        edge_type = torch.cat(edge_type, dim=0)
        out = model(data['job'].x, edge_index, edge_type).squeeze(-1)

    elif isinstance(model, (MultiplexRegression, A_MultiplexRegression, RA_MultiplexRegression, MuxGNNRegression)):
        edge_index_list = [ei for ei in data.edge_index_dict.values()]
        out = model(data['job'].x, edge_index_list).squeeze(-1)

    elif hasattr(model, 'forward') and 'edge_index' in model.forward.__code__.co_varnames:        
        raw_edge_index = torch.cat([ei for _, ei in data.edge_index_dict.items()], dim=1)
        edge_index = deduplicate_edges(raw_edge_index)
        out = model(data['job'].x, edge_index).squeeze(-1)

    else:
        out = model(data['job'].x).squeeze(-1)

    y_true = data['job'].y[mask].squeeze(-1)
    y_pred = out[mask]

    mse = F.mse_loss(y_pred, y_true).item()
    rmse = np.sqrt(mse).item()
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # var = torch.var(y_true).item()
    # r2 = 1 - mse / (var + 1e-6)
    return mse, rmse, mae, r2

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def get_classification_models(graph_data, hidden_dim, num_classes, num_layers, drop_out):
    in_dim = graph_data['job'].x.shape[1]

    return {
        # 비그래프 기반
        "Linear": lambda: LogisticRegression(max_iter=1000, n_jobs=1),

        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=100,
            max_depth=6,
            n_jobs=1,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            random_state=42
        ),
        
        # 그래프 기반
        "MLP": lambda: MLPClassifier(in_dim, hidden_dim, num_classes, num_layers),
        "GCN": lambda: GCNClassifier(in_dim, hidden_dim, num_classes, num_layers),
        "GAT": lambda: GATClassifier(in_dim, hidden_dim, num_classes, num_layers),
        "GraphSAGE": lambda: GraphSAGEClassifier(in_dim, hidden_dim, num_classes, num_layers),
        
        "RGCN": lambda: RGCNClassifier(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_classes=num_classes,
            num_relations=len(graph_data.edge_types),
            num_layers=num_layers
        ),
        "HAN": lambda: HANClassifier(
            metadata=graph_data.metadata(),
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=drop_out
        ),
        
        "MuxGNN": lambda: MuxGNNClassifier(
            in_channels=graph_data['job'].x.shape[1],
            out_channels=hidden_dim,
            gnn_type='gcn',
            num_layers=num_layers,
            num_classes=num_classes,
            num_relations=len(graph_data.edge_types),
            dropout=drop_out
        ),

        "M-GNN": lambda: MultiplexClassifier(
            in_channels=graph_data['job'].x.shape[1],
            out_channels=hidden_dim,
            num_relations=len(graph_data.edge_types),
            num_classes = num_classes,
            num_layers=num_layers,
            gnn_type='gcn',
            dropout=drop_out
        ),

        "MA-GNN": lambda: A_MultiplexClassifier(
            in_channels=graph_data['job'].x.shape[1],
            out_channels=hidden_dim,
            num_relations=len(graph_data.edge_types),
            num_classes = num_classes,
            num_layers=num_layers,
            gnn_type='gcn',
            dropout=drop_out
        ),

        "RAM-GNN": lambda: RA_MultiplexClassifier(
            in_channels=graph_data['job'].x.shape[1],
            out_channels=hidden_dim,
            num_relations=len(graph_data.edge_types),
            num_classes = num_classes,
            num_layers=num_layers,
            gnn_type='gcn',
            dropout=drop_out
        ),
                
    }

def detection_train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    if isinstance(model, HANClassifier):
        out = model(data.x_dict, data.edge_index_dict)
        y = data['job'].y
        mask = data['job'].train_mask

    elif isinstance(model, (MultiplexClassifier, A_MultiplexClassifier, RA_MultiplexClassifier, MuxGNNClassifier)):
        edge_index_list = [ei for ei in data.edge_index_dict.values()]
        out = model(data['job'].x, edge_index_list)
        y = data['job'].y
        mask = data['job'].train_mask
        
    elif isinstance(model, RGCNClassifier):
        edge_index = []
        edge_type = []
        rel_map = {rel: i for i, rel in enumerate(data.edge_types)}
        for rel, ei in data.edge_index_dict.items():
            edge_index.append(ei)
            edge_type.append(torch.full((ei.size(1),), rel_map[rel], dtype=torch.long))
        edge_index = torch.cat(edge_index, dim=1)
        edge_type = torch.cat(edge_type, dim=0)

        out = model(data['job'].x, edge_index, edge_type)
        y = data['job'].y
        mask = data['job'].train_mask
        
    elif hasattr(model, 'forward') and 'edge_index' in model.forward.__code__.co_varnames:
        raw_edge_index = torch.cat([ei for _, ei in data.edge_index_dict.items()], dim=1)
        edge_index = deduplicate_edges(raw_edge_index)
        out = model(data['job'].x, edge_index)
        y = data['job'].y
        mask = data['job'].train_mask

    else:
        out = model(data['job'].x)
        y = data['job'].y
        mask = data['job'].train_mask

    loss = F.cross_entropy(out[mask], y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def detection_evaluate(model, data, mask):
    model.eval()

    if isinstance(model, HANClassifier):
        out = model(data.x_dict, data.edge_index_dict)
        y = data['job'].y

    elif isinstance(model, (MultiplexClassifier, A_MultiplexClassifier, RA_MultiplexClassifier, MuxGNNClassifier)):
        edge_index_list = [ei for ei in data.edge_index_dict.values()]
        out = model(data['job'].x, edge_index_list)
        y = data['job'].y

    elif isinstance(model, RGCNClassifier):
        edge_index = []
        edge_type = []
        rel_map = {rel: i for i, rel in enumerate(data.edge_types)}
        for rel, ei in data.edge_index_dict.items():
            edge_index.append(ei)
            edge_type.append(torch.full((ei.size(1),), rel_map[rel], dtype=torch.long))
        edge_index = torch.cat(edge_index, dim=1)
        edge_type = torch.cat(edge_type, dim=0)

        out = model(data['job'].x, edge_index, edge_type)
        y = data['job'].y

    elif hasattr(model, 'forward') and 'edge_index' in model.forward.__code__.co_varnames:
        raw_edge_index = torch.cat([ei for _, ei in data.edge_index_dict.items()], dim=1)
        edge_index = deduplicate_edges(raw_edge_index)
        out = model(data['job'].x, edge_index)
        y = data['job'].y
    else:
        out = model(data['job'].x)
        y = data['job'].y

    pred = out[mask].argmax(dim=1).cpu().numpy()
    true = y[mask].cpu().numpy()

    acc = accuracy_score(true, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='macro', zero_division=0)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


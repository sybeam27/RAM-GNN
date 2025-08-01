#!/bin/bash
python ./utils/prediction.py --model RAM-GNN --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
# python ./utils/prediction.py --model M-GNN --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
python ./utils/prediction.py --model RGCN --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
python ./utils/prediction.py --model HAN --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
python ./utils/prediction.py --model MuxGNN --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
python ./utils/prediction.py --model Linear --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
python ./utils/prediction.py --model RandomForest --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
python ./utils/prediction.py --model XGBoost --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
python ./utils/prediction.py --model MLP --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
python ./utils/prediction.py --model GCN --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
python ./utils/prediction.py --model GAT --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 
python ./utils/prediction.py --model GraphSAGE --graph_path './data/prediction/graph_data.pt' --res_path './res/prediction' 



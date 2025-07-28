# graph data generation
# python graph_data.py --mode prediction --save_path ../data/prediction/graph_data.pt

import os
import pandas as pd
import warnings
import argparse
import torch

from function import (
    normalize_soc_code,
    create_hetero_graph,
    create_classification_hetero_graph,
    set_seed,
)

warnings.filterwarnings("ignore")
set_seed(42)

def main(mode='prediction', save_path='./data/graph_data.pt'):
    try:
        node_df = pd.read_pickle(os.path.join("..", "data", "node_clean_df.pkl"))
        edge_df = pd.read_pickle(os.path.join("..", "data", "edge_df.pkl"))
        # occupation_df = pd.read_csv(
        #     os.path.join(".", "aaai_dataset", "Kaggle", "automation_data_by_state.csv"),
        #     encoding='latin1'
        # ).iloc[:, :2]
    except Exception as e:
        raise FileNotFoundError(f"Error loading data files: {e}")

    # Normalize SOC codes in edges
    edge_df['source'] = edge_df['source'].apply(normalize_soc_code)
    edge_df['target'] = edge_df['target'].apply(normalize_soc_code)

    # Filter edges with valid SOC codes
    soc2idx = {soc: i for i, soc in enumerate(node_df['SOC_CODE'])}
    edge_df = edge_df[edge_df['source'].isin(soc2idx) & edge_df['target'].isin(soc2idx)]

    # Calculate time series mean values and growth
    node_df['Employ_Mean_Recent'] = node_df[[f'Employ_{y}' for y in range(2022, 2025)]].mean(axis=1)
    node_df['Wage_Mean_Recent'] = node_df[[f'Wage_{y}' for y in range(2022, 2025)]].mean(axis=1)
    node_df['Employ_Growth'] = (node_df['Employ_2024'] - node_df['Employ_2019']) / node_df['Employ_2019']
    node_df['Wage_Growth'] = (node_df['Wage_2024'] - node_df['Wage_2019']) / node_df['Wage_2019']

    # Define node features
    feature_cols = [
        'Probability', 'is_STEM', 'Education_Level', 
        'Tech_Count', 'Hot_Tech_Count', 'In_Demand_Count', 
        'Employ_Mean_Recent', 'Wage_Mean_Recent', 
        'Employ_Growth', 'Wage_Growth'
    ]

    # Define anomaly risk conditions
    high_AIOE = node_df['AIOE'] >= node_df['AIOE'].quantile(0.75) 
    low_wage = node_df['Wage_Mean_Recent'] < node_df['Wage_Mean_Recent'].median()  
    low_edu = node_df['Education_Level'] < 2 
    non_STEM = node_df['is_STEM'] == 0 
    shrinking_employ = node_df['Employ_Growth'] < 0

    risk_factors = (
        low_wage.astype(int) +
        low_edu.astype(int) +
        non_STEM.astype(int) +
        shrinking_employ.astype(int)
    )

    # Assign binary anomaly label
    node_df['Anomaly_label'] = ((high_AIOE) & (risk_factors >= 2)).astype(int)

    # Construct graph data
    if mode == 'prediction':
        graph_data = create_hetero_graph(node_df, edge_df, feature_cols)
    elif mode == 'detection':
        graph_data = create_classification_hetero_graph(node_df, edge_df, feature_cols, mode='anomaly')
    else:
        raise ValueError("Invalid mode. Use 'prediction' or 'detection'.")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save graph data to file
    torch.save(graph_data, save_path)
    print(f"Graph data has been successfully generated and saved to {save_path}")

    return graph_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct and save multiplex-graph from occupational data.")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="prediction",
        choices=["prediction", "detection"],
        help="Graph construction mode: 'prediction' for feature-based graph, 'detection' for anomaly labeling."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/graph_data.pt",
        help="Path to save the generated graph data (e.g., ./data/graph_data.pt)."
    )
    args = parser.parse_args()

    main(mode=args.mode, save_path=args.save_path)
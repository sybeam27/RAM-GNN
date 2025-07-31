# prediction 
# python prediction_all.py

import os
import csv
import torch
import joblib
import argparse
import copy

from function import (
    set_seed, split_masks, load_data, get_model_dict,
    prediction_train, prediction_evaluate, compute_metrics, 
)

def run_model(model_name, args, graph_data, device):
    x_np = graph_data['job'].x.cpu().numpy()
    y_np = graph_data['job'].y.squeeze().cpu().numpy()
    train_mask = graph_data['job'].train_mask.cpu().numpy()
    val_mask = graph_data['job'].val_mask.cpu().numpy()
    test_mask = graph_data['job'].test_mask.cpu().numpy()

    model_dict = get_model_dict(graph_data, args.hidden_dim, args.num_layers, args.dropout)
    if model_name not in model_dict:
        raise ValueError(f"Model '{model_name}' is not supported.")

    model = model_dict[model_name]().to(device)
    result_str = f"\n{'='*30}\n▶ {model_name} Evaluation\n"

    model_save_dir = os.path.join(args.save_path, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    if hasattr(model, 'fit'):
        save_model_path = os.path.join(model_save_dir, f"{model_name}_best.pkl")

        model.fit(x_np[train_mask], y_np[train_mask])
        pred_val = model.predict(x_np[val_mask])
        val_mse, _, _, _ = compute_metrics(y_np[val_mask], pred_val)

        joblib.dump(model, save_model_path)
        best_model = joblib.load(save_model_path)
        pred_test = best_model.predict(x_np[test_mask])
        test_mse, test_rmse, test_mae, test_r2 = compute_metrics(y_np[test_mask], pred_test)

    else: 
        save_model_path = os.path.join(model_save_dir, f"{model_name}_best.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_mse = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            loss = prediction_train(model, graph_data, optimizer, device)
            val_mse, _, _, val_r2 = prediction_evaluate(model, graph_data, graph_data['job'].val_mask, device)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, save_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break

        model.load_state_dict(torch.load(save_model_path))
        model.eval()
        test_mse, test_rmse, test_mae, test_r2 = prediction_evaluate(model, graph_data, graph_data['job'].test_mask, device)

    result_str += f"[{model_name}] Test MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}"
    print(result_str)

    result_dir = os.path.join(args.res_path, model_name)
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, "prediction_exp_results.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        fieldnames = ['model', 'mse', 'rmse', 'mae', 'r2', 'epochs', 'lr', 'dropout', 'hidden_dim']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'model': model_name,
            'mse': round(test_mse, 4),
            'rmse': round(test_rmse, 4),
            'mae': round(test_mae, 4),
            'r2': round(test_r2, 4),
            'epochs': args.epochs,
            'lr': args.lr,
            'dropout': args.dropout,
            'hidden_dim': args.hidden_dim
        })

def main(args):
    # set_seed(42)
    # graph_data = torch.load(args.graph_path, weights_only=False)
    # graph_data = split_masks(graph_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    graph_data = load_data(args.graph_path)
    model_list = [
        "Linear", "RandomForest", "XGBoost", "MLP", 
        "GCN", "GAT", "GraphSAGE", 
        "RGCN", "HAN", 
        "MuxGNN", "RAM-GNN"
    ]

    for model_name in model_list:
        try:
            run_model(model_name, args, graph_data, device)
        except Exception as e:
            print(f"[ERROR] Failed to run model {model_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all prediction models on graph data")
    parser.add_argument('--graph_path', type=str, default='../data/prediction/graph_data.pt')
    parser.add_argument('--save_path', type=str, default='../model/prediction/')
    parser.add_argument('--res_path', type=str, default='../res/prediction/')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=100)
    
    args = parser.parse_args()
    main(args)
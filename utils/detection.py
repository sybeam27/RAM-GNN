# prediction 
# python detection.py --model RAM-GNN

import os
import csv
import copy
import torch
import joblib
import argparse
import numpy as np

from function import (
    load_data, get_classification_models,
    detection_train, detection_evaluate
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(args):
    graph_data = load_data(args.graph_path)

    x_np = graph_data['job'].x.cpu().numpy()
    y_np = graph_data['job'].y.cpu().numpy()
    if y_np.ndim > 1:
        y_np = y_np.squeeze()
    y_np = y_np.astype(np.int64)

    train_mask = graph_data['job'].train_mask.cpu().numpy()
    val_mask = graph_data['job'].val_mask.cpu().numpy()
    test_mask = graph_data['job'].test_mask.cpu().numpy()

    model_dict = get_classification_models(graph_data, args.hidden_dim, args.num_classes, args.num_layers, args.dropout)

    if args.model not in model_dict:
        raise ValueError(f"Model '{args.model}' is not supported.")

    model = model_dict[args.model]()
    result_str = f"\n{'='*30}\n▶ {args.model} Evaluation\n"

    model_save_dir = os.path.join(args.save_path, args.model)
    os.makedirs(model_save_dir, exist_ok=True)

    if hasattr(model, 'fit'):  # sklearn-style model
        save_model_path = os.path.join(model_save_dir, f"{args.model}_best.pkl").replace("\\", "/")

        model.fit(x_np[train_mask], y_np[train_mask])
        val_pred = model.predict(x_np[val_mask])
        val_acc = accuracy_score(y_np[val_mask], val_pred)
        val_f1 = f1_score(y_np[val_mask], val_pred)

        joblib.dump(model, save_model_path)
        best_model = joblib.load(save_model_path)        
        pred_test = best_model.predict(x_np[test_mask])
        test_acc = accuracy_score(y_np[test_mask], pred_test)
        test_f1 = f1_score(y_np[test_mask], pred_test)
        test_precision = precision_score(y_np[test_mask], pred_test, average='macro', zero_division=0)
        test_recall = recall_score(y_np[test_mask], pred_test, average='macro', zero_division=0)

    else:  # torch-based model
        save_model_path = os.path.join(model_save_dir, f"{args.model}_best.pt").replace("\\", "/")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_acc = 0.0
        best_model_state = None
        patience = 500
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            loss = detection_train(model, graph_data, optimizer)
            val_result = detection_evaluate(model, graph_data, graph_data['job'].val_mask)
            val_acc = val_result['accuracy']

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, save_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            if epoch % 100 == 0:
                print(f"[{args.model}] Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_result['f1']:.4f}")

        model.load_state_dict(torch.load(save_model_path))
        model.eval()
        test_result = detection_evaluate(model, graph_data, graph_data['job'].test_mask)
        test_acc = test_result['accuracy']
        test_precision = test_result['precision']
        test_recall = test_result['recall'] 
        test_f1 = test_result['f1']

    result_str += f"[{args.model}] Test Acc: {test_acc:.4f} | Prec: {test_precision:.4f} | Rec: {test_recall:.4f} | F1: {test_f1:.4f}"
    print(result_str)

    result_dir = os.path.join(args.res_path, args.model)
    os.makedirs(result_dir, exist_ok=True)

    # Save to CSV (append or create)
    csv_path = os.path.join(result_dir, "detection_exp_results.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        fieldnames = ['model', 'accuracy', 'precision', 'recall', 'f1', 'graph_path', 'epochs', 'lr', 'dropout', 'hidden_dim']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'model': args.model,
            'accuracy': round(test_acc, 4),
            'precision': round(test_precision, 4),
            'recall': round(test_recall, 4),
            'f1': round(test_f1, 4),
            # 'graph_path': args.graph_path,
            'epochs': args.epochs,
            'lr': args.lr,
            'dropout': args.dropout,
            'hidden_dim': args.hidden_dim
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detection model on graph data")
    parser.add_argument('--model', type=str, required=True,
                        help="Model name: Linear, RandomForest, XGBoost, MLP, GCN, GAT, GraphSAGE, RGCN, HAN, MuxGNN, M-GNN, MA-GNN, RAM-GNN")
    parser.add_argument('--graph_path', type=str, default='../data/detection/graph_data.pt',
                        help="Path to saved graph_data.pt")
    parser.add_argument('--save_path', type=str, default='../model/detection/',
                        help="Base directory to save model (actual path will include model name)")
    parser.add_argument('--res_path', type=str, default='../res/detection/')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=500)

    args = parser.parse_args()
    print(f"\n{'='*30}\n▶ {args.model} 시작")
    main(args)
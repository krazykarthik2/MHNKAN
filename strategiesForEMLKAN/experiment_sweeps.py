import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Adjust path to import from KAN_EML
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'KAN_EML')))
from eml_network import EMLKAN

def generate_regression_data(n_samples=200):
    torch.manual_seed(42)
    X = (torch.rand(n_samples, 2) * 2.0 - 1.0).double() # [n_samples, 2] in [-1, 1]^2
    # Non-linear function: sin(pi * x1) * exp(x2)
    y = (torch.sin(np.pi * X[:, 0]) * torch.exp(X[:, 1])).unsqueeze(-1).double()
    return X, y

def generate_classification_data(n_samples=200, n_features=5, n_classes=3):
    torch.manual_seed(42)
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X_np, y_np = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=3, 
        n_classes=n_classes, 
        random_state=42
    )
    X = torch.tensor(X_np, dtype=torch.double)
    y = torch.tensor(y_np, dtype=torch.long)
    return X, y

def train_regression_model(layers_hidden, num_eml_components, X, y, init_scale=0.5, epochs=300):
    model = EMLKAN(layers_hidden=layers_hidden, num_eml_components=num_eml_components).double()
    
    # Custom initialization scale control
    with torch.no_grad():
        for layer in model.layers:
            layer.weight_base.normal_(0, 0.1)
            layer.weight_eml.normal_(0, 0.1)
            layer.a.normal_(0, init_scale)
            layer.b.normal_(0, 0.1)
            layer.c.normal_(0, init_scale)
            layer.d.normal_(0, 0.1)
            
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-4)
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
    # Apply L-BFGS for fine-tuning in regression
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=0.5, 
        max_iter=50, 
        line_search_fn="strong_wolfe"
    )
    
    def closure():
        optimizer_lbfgs.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        return loss
        
    try:
        optimizer_lbfgs.step(closure)
    except Exception:
        pass # Handle any numerical L-BFGS instability gracefully
        
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(X), y).item()
    duration = time.time() - start_time
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return final_loss, params, duration

def train_classification_model(layers_hidden, num_eml_components, X, y, init_scale=0.5, epochs=200):
    model = EMLKAN(layers_hidden=layers_hidden, num_eml_components=num_eml_components).double()
    
    with torch.no_grad():
        for layer in model.layers:
            layer.weight_base.normal_(0, 0.1)
            layer.weight_eml.normal_(0, 0.1)
            layer.a.normal_(0, init_scale)
            layer.b.normal_(0, 0.1)
            layer.c.normal_(0, init_scale)
            layer.d.normal_(0, 0.1)
            
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-4)
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = criterion(logits, y).item()
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).double().mean().item()
        
    duration = time.time() - start_time
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return loss, accuracy, params, duration

def main():
    print("Running Regression Sweeps...")
    X_reg, y_reg = generate_regression_data(200)
    
    # Configurations: [layers_hidden, K, init_scale]
    reg_configs = [
        # Width impact
        ([2, 2, 1], 1, 0.5),
        ([2, 4, 1], 1, 0.5),
        ([2, 8, 1], 1, 0.5),
        
        # Depth impact
        ([2, 4, 4, 1], 1, 0.5),
        
        # EML Components (K) impact
        ([2, 4, 1], 2, 0.5),
        ([2, 4, 1], 4, 0.5),
        
        # Init Scale impact
        ([2, 4, 1], 2, 0.1),
        ([2, 4, 1], 2, 1.0),
    ]
    
    reg_results = []
    for layers, K, init_scale in reg_configs:
        loss, params, duration = train_regression_model(layers, K, X_reg, y_reg, init_scale)
        reg_results.append({
            "layers": str(layers),
            "K": K,
            "init_scale": init_scale,
            "params": params,
            "loss": loss,
            "duration": duration
        })
        print(f"Layers: {layers} | K: {K} | Init Scale: {init_scale} | Params: {params} | Loss: {loss:.6f} | Time: {duration:.2f}s")
        
    print("\nRunning Classification Sweeps...")
    X_clf, y_clf = generate_classification_data(200, n_features=5, n_classes=3)
    
    clf_configs = [
        ([5, 3], 1, 0.5),
        ([5, 4, 3], 1, 0.5),
        ([5, 8, 3], 1, 0.5),
        ([5, 4, 3], 2, 0.5),
        ([5, 4, 3], 2, 0.1),
    ]
    
    clf_results = []
    for layers, K, init_scale in clf_configs:
        loss, acc, params, duration = train_classification_model(layers, K, X_clf, y_clf, init_scale)
        clf_results.append({
            "layers": str(layers),
            "K": K,
            "init_scale": init_scale,
            "params": params,
            "loss": loss,
            "accuracy": acc,
            "duration": duration
        })
        print(f"Layers: {layers} | K: {K} | Init Scale: {init_scale} | Params: {params} | Loss: {loss:.4f} | Acc: {acc*100:.2f}% | Time: {duration:.2f}s")
        
    # Write results to md file
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    report_path = os.path.join(os.path.dirname(__file__), "experiment_results.md")
    with open(report_path, "w") as f:
        f.write("# EML-KAN Sweep Experiment Observations\n\n")
        
        f.write("## 1. Regression Sweeps\n\n")
        f.write("| Layers Structure | EML Components (K) | Init Scale | Params | Final MSE Loss | Duration (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for res in reg_results:
            f.write(f"| `{res['layers']}` | {res['K']} | {res['init_scale']} | {res['params']} | {res['loss']:.8f} | {res['duration']:.2f} |\n")
            
        f.write("\n## 2. Classification Sweeps\n\n")
        f.write("| Layers Structure | EML Components (K) | Init Scale | Params | Cross-Entropy Loss | Accuracy (%) | Duration (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for res in clf_results:
            f.write(f"| `{res['layers']}` | {res['K']} | {res['init_scale']} | {res['params']} | {res['loss']:.4f} | {res['accuracy']*100:.2f}% | {res['duration']:.2f} |\n")
            
    print(f"\nResults successfully written to {report_path}")

if __name__ == "__main__":
    main()

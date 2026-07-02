import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_wine, load_digits

# Adjust path to import from KAN_EML
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'KAN_EML')))
from eml_network import EMLKAN

def get_tabular_data():
    data = load_wine()
    X = torch.tensor(data.data, dtype=torch.double)
    y = torch.tensor(data.target, dtype=torch.long)
    # Normalize features
    X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)
    return X, y, "Tabular (Wine Classification: 13 feats, 3 classes)"

def get_image_data():
    data = load_digits()
    X = torch.tensor(data.data, dtype=torch.double) / 16.0 # Normalize 0-1
    y = torch.tensor(data.target, dtype=torch.long)
    return X, y, "Image (Digits Classification: 64 feats, 10 classes)"

def get_audio_data(n_samples=300):
    torch.manual_seed(42)
    t = torch.linspace(0, 1, n_samples, dtype=torch.double).unsqueeze(-1)
    # Sum of three sinusoids (multi-frequency audio-like wave)
    y = (torch.sin(2 * np.pi * 5 * t) + 0.5 * torch.sin(2 * np.pi * 12 * t) + 0.2 * torch.sin(2 * np.pi * 25 * t))
    return t, y, "Audio (Multi-frequency Waveform Regression: 1 feat, 1 output)"

def get_clean_function_data(n_samples=250):
    torch.manual_seed(42)
    X = (torch.rand(n_samples, 3) * 2.0 - 1.0).double()
    y = (torch.sin(X[:, 0] * np.pi) * torch.exp(X[:, 1]) + X[:, 2]**2).unsqueeze(-1)
    return X, y, "Clean Function Regression (3 feats, 1 output)"

def get_noisy_function_data(n_samples=250):
    X, y, _ = get_clean_function_data(n_samples)
    torch.manual_seed(42)
    noise = torch.randn_like(y) * 0.2
    return X, y + noise, "Noisy Function Regression (3 feats, 1 output, noise std=0.2)"

def train_eval(model, X, y, is_classification, epochs=150):
    criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Train test split (80/20)
    n_samples = X.shape[0]
    indices = torch.randperm(n_samples)
    split = int(0.8 * n_samples)
    
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        test_loss = criterion(test_out, y_test).item()
        if is_classification:
            preds = torch.argmax(test_out, dim=1)
            metric = (preds == y_test).double().mean().item() # Accuracy
        else:
            metric = test_loss # MSE
            
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return test_loss, metric, params

def run_sweeps():
    datasets = [
        ("Tabular", get_tabular_data(), True),
        ("Image", get_image_data(), True),
        ("Audio", get_audio_data(), False),
        ("Clean Function", get_clean_function_data(), False),
        ("Noisy Function", get_noisy_function_data(), False)
    ]
    
    # Structural candidates to sweep: [layers, K]
    # We will sweep small, medium, and deep/wide configs for each modality
    results = {}
    
    for name, (X, y, desc), is_clf in datasets:
        print(f"\nEvaluating: {desc}")
        in_dim = X.shape[1]
        out_dim = 3 if name == "Tabular" else (10 if name == "Image" else 1)
        
        configs = [
            # Shallow/Narrow
            ([in_dim, out_dim], 1),
            ([in_dim, out_dim], 2),
            # Medium
            ([in_dim, int(in_dim * 0.5) + 2, out_dim], 1),
            ([in_dim, int(in_dim * 0.5) + 2, out_dim], 2),
            # Deep/Wide
            ([in_dim, in_dim + 2, in_dim + 2, out_dim], 1),
        ]
        
        results[name] = []
        for layers, K in configs:
            # Skip excessively large image networks for rapid execution
            if name == "Image" and len(layers) > 2 and layers[1] > 30:
                layers = [in_dim, 16, 16, out_dim] # Downscale for digits speed
                
            model = EMLKAN(layers, num_eml_components=K).double()
            test_loss, metric, params = train_eval(model, X, y, is_clf)
            results[name].append({
                "layers": layers,
                "K": K,
                "params": params,
                "test_loss": test_loss,
                "metric": metric
            })
            metric_name = "Accuracy" if is_clf else "MSE"
            print(f"  Config: Layers={layers}, K={K} | Params: {params} | Test Loss: {test_loss:.4f} | {metric_name}: {metric:.4f}")
            
    # Write to a report file
    report_path = "strategiesForEMLKAN/multi_domain_results.md"
    with open(report_path, "w") as f:
        f.write("# Multi-Domain EML-KAN Sweep Results\n\n")
        f.write("This report presents empirical benchmarking of various EML-KAN structures across diverse data domains.\n\n")
        
        for name in results:
            f.write(f"## {name} Dataset sweeps\n\n")
            is_clf = name in ["Tabular", "Image"]
            metric_header = "Test Accuracy (%)" if is_clf else "Test MSE Loss"
            
            f.write(f"| Network Layers | K Components | Parameters | Test Loss | {metric_header} |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            
            for res in results[name]:
                val = f"{res['metric']*100:.2f}%" if is_clf else f"{res['metric']:.6f}"
                f.write(f"| `{res['layers']}` | {res['K']} | {res['params']} | {res['test_loss']:.5f} | {val} |\n")
            f.write("\n")
            
    print(f"\nSweep reports written to {report_path}")

if __name__ == "__main__":
    run_sweeps()

import os
import sys
# Injected path for root and core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from eml_network import EMLKAN
from kan_hopfield import RBFKAN

def target_function(x):
    """
    Complex target function:
    y = exp(sin(x1) - cos(x2)) - ln(softplus(x1*x2) + 0.1)
    """
    x1, x2 = x[:, 0], x[:, 1]
    val1 = torch.sin(x1) - torch.cos(x2)
    val2 = F.softplus(x1 * x2) + 0.1
    return torch.exp(val1) - torch.log(val2)

def main():
    print("Training [2, 6, 1] models on the complex math function to extract all 18 edge curves...")
    
    # 1. Dataset
    torch.manual_seed(42)
    np.random.seed(42)
    X_train = (torch.rand(300, 2) * 3.0 - 1.5) # range [-1.5, 1.5]
    y_train = target_function(X_train).unsqueeze(-1)
    
    # 2. Train Standard KAN [2, 6, 1]
    print("Training Standard RBF-KAN...")
    rbf_kan = RBFKAN(layers_hidden=[2, 6, 1], grid_size=10, grid_range=[-2.0, 2.0])
    optimizer_rbf = optim.AdamW(rbf_kan.parameters(), lr=0.015)
    for epoch in range(1, 1001):
        rbf_kan.train()
        optimizer_rbf.zero_grad()
        loss = F.mse_loss(rbf_kan(X_train), y_train)
        loss.backward()
        optimizer_rbf.step()
        if epoch % 500 == 0:
            print(f"  RBF-KAN Loss: {loss.item():.6f}")
            
    # 3. Train EML-KAN [2, 6, 1]
    print("Training EML-KAN...")
    eml_kan = EMLKAN(layers_hidden=[2, 6, 1], num_eml_components=4)
    optimizer_eml = optim.AdamW(eml_kan.parameters(), lr=0.015)
    for epoch in range(1, 1001):
        eml_kan.train()
        optimizer_eml.zero_grad()
        loss = F.mse_loss(eml_kan(X_train), y_train)
        loss.backward()
        optimizer_eml.step()
        if epoch % 500 == 0:
            print(f"  EML-KAN Loss: {loss.item():.6f}")
            
    # 4. Extract curves over evaluation range
    x_test_range = np.linspace(-1.5, 1.5, 100)
    x_eval = torch.tensor(x_test_range, dtype=torch.float32).unsqueeze(-1) # [100, 1]
    
    fig, axes = plt.subplots(6, 3, figsize=(18, 25), dpi=150)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    
    # Helper to compute EML edge output (sum of K components)
    def get_eml_edge_y(layer, i, j, x):
        # i: out_feature index, j: in_feature index
        x_expanded = x.unsqueeze(1).expand(-1, layer.num_eml_components) # [100, K]
        # Evaluate for specific edge (i, j)
        a = layer.a[i, j] # [K]
        b = layer.b[i, j]
        c = layer.c[i, j]
        d = layer.d[i, j]
        w = layer.weight_eml[i, j]
        
        arg_x = a * x_expanded + b
        arg_y = F.softplus(c * x_expanded + d) + layer.eps
        eml_val = torch.exp(torch.clamp(arg_x, -10, 10)) - torch.log(arg_y)
        edge_out = torch.sum(eml_val * w, dim=-1)
        return edge_out.detach().cpu().numpy()
        
    # Helper to compute Standard RBF-KAN edge output
    def get_rbf_edge_y(layer, i, j, x):
        x_expanded = x.unsqueeze(-1) # [100, 1, 1]
        diff = x_expanded - layer.grid
        rbf_act = torch.exp(-torch.square(diff) / (2 * (layer.sigma ** 2))) # [100, 1, grid_size]
        w_rbf = layer.weight_rbf[i, j] # [grid_size]
        y_rbf = torch.sum(rbf_act * w_rbf.unsqueeze(0).unsqueeze(0), dim=-1).squeeze().detach().cpu().numpy()
        return y_rbf

    edge_idx = 0
    # --- Layer 0: Input 2 -> Hidden 6 = 12 Edges ---
    for i in range(6): # Hidden nodes
        for j in range(2): # Input nodes
            row = edge_idx // 3
            col = edge_idx % 3
            ax = axes[row, col]
            
            y_rbf = get_rbf_edge_y(rbf_kan.layers[0], i, j, x_eval.squeeze())
            y_eml = get_eml_edge_y(eml_kan.layers[0], i, j, x_eval.squeeze())
            
            ax.plot(x_test_range, y_rbf, color='#E11D48', linewidth=2.0, label='Standard RBF-KAN')
            ax.plot(x_test_range, y_eml, color='#7C3AED', linewidth=2.0, label='EML-KAN')
            ax.set_title(f"Edge {edge_idx}: Input $x_{j+1} \\rightarrow$ Hidden $h_{i+1}$", fontsize=10, fontweight='bold')
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper left", fontsize=8)
            edge_idx += 1
            
    # --- Layer 1: Hidden 6 -> Output 1 = 6 Edges ---
    # Evaluate over typical hidden ranges [-1.5, 1.5]
    for j in range(6): # Hidden nodes
        row = edge_idx // 3
        col = edge_idx % 3
        ax = axes[row, col]
        
        y_rbf = get_rbf_edge_y(rbf_kan.layers[1], 0, j, x_eval.squeeze())
        y_eml = get_eml_edge_y(eml_kan.layers[1], 0, j, x_eval.squeeze())
        
        ax.plot(x_test_range, y_rbf, color='#E11D48', linewidth=2.0, label='Standard RBF-KAN')
        ax.plot(x_test_range, y_eml, color='#7C3AED', linewidth=2.0, label='EML-KAN')
        ax.set_title(f"Edge {edge_idx}: Hidden $h_{j+1} \\rightarrow$ Output $y_1$", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", fontsize=8)
        edge_idx += 1
        
    plt.suptitle("Grid Comparison of All 18 Learned Edge Curves on the Complex Math Function", fontsize=15, fontweight='bold', color='#0F172A', y=0.99)
    
    dest_path = os.path.join(os.path.dirname(__file__), "trained_curves_comparison.png")
    plt.savefig(dest_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"All 18 edge curves successfully generated and saved to: {dest_path}")

if __name__ == "__main__":
    main()

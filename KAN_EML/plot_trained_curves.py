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
    val = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for j in range(x.shape[1]):
        val_x = 1.2 * x[:, j] - 0.3
        val_y = F.softplus(0.8 * x[:, j] + 0.2) + 1e-6
        val += torch.exp(val_x) - torch.log(val_y)
    return val

def main():
    print("Training both Standard KAN and EML-KAN to extract actual learned curves...")
    
    # 1. Dataset (float32 for standard KAN training compatibility)
    torch.manual_seed(42)
    np.random.seed(42)
    X_train = (torch.rand(300, 2) * 2.0 - 1.0)
    y_train = target_function(X_train).unsqueeze(-1)
    
    # 2. Train Standard KAN (RBF-based)
    print("Training Standard RBF-KAN...")
    rbf_kan = RBFKAN(layers_hidden=[2, 1], grid_size=10, grid_range=[-1.2, 1.2])
    optimizer_rbf = optim.AdamW(rbf_kan.parameters(), lr=0.02)
    for epoch in range(1, 1001):
        rbf_kan.train()
        optimizer_rbf.zero_grad()
        loss = F.mse_loss(rbf_kan(X_train), y_train)
        loss.backward()
        optimizer_rbf.step()
        
    # 3. Train EML-KAN
    print("Training EML-KAN...")
    eml_kan = EMLKAN(layers_hidden=[2, 1], num_eml_components=1)
    # Freeze linear base for pure EML comparison
    with torch.no_grad():
        eml_kan.layers[0].weight_base.zero_()
        eml_kan.layers[0].weight_base.requires_grad = False
        eml_kan.layers[0].weight_eml.fill_(1.0)
        eml_kan.layers[0].weight_eml.requires_grad = False
        
    optimizer_eml = optim.AdamW(eml_kan.parameters(), lr=0.03)
    for epoch in range(1, 1001):
        eml_kan.train()
        optimizer_eml.zero_grad()
        loss = F.mse_loss(eml_kan(X_train), y_train)
        loss.backward()
        optimizer_eml.step()
        
    # Fine-tune EML-KAN with L-BFGS for exact match
    optimizer_lbfgs = optim.LBFGS(eml_kan.parameters(), lr=0.5, max_iter=100)
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = F.mse_loss(eml_kan(X_train), y_train)
        loss.backward()
        return loss
    optimizer_lbfgs.step(closure)
    
    # 4. Extract curves over domain [-1.0, 1.0]
    x_test_range = np.linspace(-1.0, 1.0, 100)
    x_eval = torch.tensor(x_test_range, dtype=torch.float32).unsqueeze(-1) # [100, 1]
    
    # Ground Truth Curve for a single feature
    y_gt = (torch.exp(1.2 * x_eval - 0.3) - torch.log(F.softplus(0.8 * x_eval + 0.2) + 1e-6)).squeeze().numpy()
    
    # Evaluate Standard KAN Edge 0 and Edge 1
    # We pass [x, 0] to evaluate feature 0, and [0, x] to evaluate feature 1
    rbf_kan.eval()
    with torch.no_grad():
        # Feature 0 curve (set feature 1 to zero)
        eval_f0 = torch.zeros(100, 2)
        eval_f0[:, 0] = x_eval.squeeze()
        # The output includes base and RBF: we extract the edge output
        y_rbf_f0 = rbf_kan(eval_f0).squeeze().numpy()
        
        # Feature 1 curve (set feature 0 to zero)
        eval_f1 = torch.zeros(100, 2)
        eval_f1[:, 1] = x_eval.squeeze()
        y_rbf_f1 = rbf_kan(eval_f1).squeeze().numpy()
        
    # Evaluate EML-KAN Edge 0 and Edge 1
    eml_kan.eval()
    with torch.no_grad():
        # Evaluate using the KAN layer EML formula directly on the inputs
        layer = eml_kan.layers[0]
        # Evaluate feature 0
        eml_arg_x0 = layer.a[0, 0, 0] * x_eval.squeeze() + layer.b[0, 0, 0]
        eml_arg_y0 = F.softplus(layer.c[0, 0, 0] * x_eval.squeeze() + layer.d[0, 0, 0]) + layer.eps
        y_eml_f0 = (torch.exp(eml_arg_x0) - torch.log(eml_arg_y0)).numpy()
        
        # Evaluate feature 1
        eml_arg_x1 = layer.a[0, 1, 0] * x_eval.squeeze() + layer.b[0, 1, 0]
        eml_arg_y1 = F.softplus(layer.c[0, 1, 0] * x_eval.squeeze() + layer.d[0, 1, 0]) + layer.eps
        y_eml_f1 = (torch.exp(eml_arg_x1) - torch.log(eml_arg_y1)).numpy()
        
    # 5. Plotting Comparison of the Learned Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    # Left Panel: Feature 1 Edge Curve
    axes[0].plot(x_test_range, y_gt, color='#0F172A', linestyle='--', linewidth=2.5, label='Ground Truth Target')
    axes[0].plot(x_test_range, y_rbf_f0, color='#E11D48', linewidth=2.0, label='Learned Standard RBF-KAN')
    axes[0].plot(x_test_range, y_eml_f0, color='#7C3AED', linewidth=2.0, label='Learned EML-KAN')
    axes[0].set_title("Edge 0 (Input $x_1 \\rightarrow$ Output $y_1$)\nLearned Univariate Function $\\phi_1(x_1)$", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Input Activation $x_1$", fontsize=10)
    axes[0].set_ylabel("Edge Output $\\phi(x_1)$", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left")
    
    # Right Panel: Feature 2 Edge Curve
    axes[1].plot(x_test_range, y_gt, color='#0F172A', linestyle='--', linewidth=2.5, label='Ground Truth Target')
    axes[1].plot(x_test_range, y_rbf_f1, color='#E11D48', linewidth=2.0, label='Learned Standard RBF-KAN')
    axes[1].plot(x_test_range, y_eml_f1, color='#7C3AED', linewidth=2.0, label='Learned EML-KAN')
    axes[1].set_title("Edge 1 (Input $x_2 \\rightarrow$ Output $y_1$)\nLearned Univariate Function $\\phi_2(x_2)$", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Input Activation $x_2$", fontsize=10)
    axes[1].set_ylabel("Edge Output $\\phi(x_2)$", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left")
    
    plt.suptitle("Empirical Comparison of Trained KAN Edge Curves: Splines vs. EML Basis", fontsize=15, fontweight='bold', color='#0F172A', y=1.0)
    
    dest_path = os.path.join(os.path.dirname(__file__), "trained_curves_comparison.png")
    plt.savefig(dest_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trained curves plot successfully saved to: {dest_path}")

if __name__ == "__main__":
    main()

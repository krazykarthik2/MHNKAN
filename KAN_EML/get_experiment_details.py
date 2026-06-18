import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from eml_network import EMLKAN
from kan_hopfield import RBFKAN

def target_function(x):
    x1, x2 = x[:, 0], x[:, 1]
    val1 = torch.sin(x1) - torch.cos(x2)
    val2 = F.softplus(x1 * x2) + 0.1
    return torch.exp(val1) - torch.log(val2)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    X_train = (torch.rand(300, 2) * 3.0 - 1.5)
    y_train = target_function(X_train).unsqueeze(-1)
    
    # RBF-KAN
    rbf_kan = RBFKAN(layers_hidden=[2, 6, 1], grid_size=10, grid_range=[-2.0, 2.0])
    optimizer_rbf = optim.AdamW(rbf_kan.parameters(), lr=0.015)
    for epoch in range(1, 1500):
        rbf_kan.train()
        optimizer_rbf.zero_grad()
        loss = F.mse_loss(rbf_kan(X_train), y_train)
        loss.backward()
        optimizer_rbf.step()
    final_rbf_loss = F.mse_loss(rbf_kan(X_train), y_train).item()
    
    # EML-KAN
    eml_kan = EMLKAN(layers_hidden=[2, 6, 1], num_eml_components=4)
    with torch.no_grad():
        for layer in eml_kan.layers:
            layer.weight_base.zero_()
            layer.weight_base.requires_grad = False
            layer.weight_eml.fill_(1.0)
            layer.weight_eml.requires_grad = False
            
    optimizer_eml = optim.AdamW(eml_kan.parameters(), lr=0.015)
    for epoch in range(1, 1500):
        eml_kan.train()
        optimizer_eml.zero_grad()
        loss = F.mse_loss(eml_kan(X_train), y_train)
        loss.backward()
        optimizer_eml.step()
        
    optimizer_lbfgs = optim.LBFGS(eml_kan.parameters(), lr=0.5, max_iter=200)
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = F.mse_loss(eml_kan(X_train), y_train)
        loss.backward()
        return loss
    optimizer_lbfgs.step(closure)
    
    final_eml_loss = F.mse_loss(eml_kan(X_train), y_train).item()
    
    print(f"FINAL_RBF_LOSS={final_rbf_loss:.12f}")
    print(f"FINAL_EML_LOSS={final_eml_loss:.12f}")
    
    # Let's print the learned formula parameters for EML-KAN Layer 1 (Hidden to Output: 6 edges)
    # The output is sum_{j=0..5} sum_{k=0..3} eml(a*h_j + b, softplus(c*h_j + d) + eps)
    layer1 = eml_kan.layers[1]
    print("\nEML-KAN Layer 1 (Hidden to Output) parameters:")
    for j in range(6):
        print(f"  Edge from h_{j+1} -> y_1:")
        for k in range(4):
            a = layer1.a[0, j, k].item()
            b = layer1.b[0, j, k].item()
            c = layer1.c[0, j, k].item()
            d = layer1.d[0, j, k].item()
            print(f"    Component {k+1}: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")

if __name__ == "__main__":
    main()

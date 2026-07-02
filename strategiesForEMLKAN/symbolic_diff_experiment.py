import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sympy as sp
import numpy as np

# Adjust path to import from KAN_EML
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'KAN_EML')))
from eml_network import EMLKAN

def softplus_sympy(x):
    return sp.log(1 + sp.exp(x))

def target_f1(x):
    return torch.exp(2.0 * x) - torch.log(F.softplus(x) + 1e-6)

def target_f2(x):
    return torch.exp(-1.5 * x) - torch.log(F.softplus(0.5 * x + 0.2) + 1e-6)

def target_f3(x):
    return 0.5 * x + 1.2 * (torch.exp(0.8 * x) - torch.log(F.softplus(x) + 1e-6))

def train_and_extract(target_fn, epochs=2500):
    torch.manual_seed(42)
    # Generate 150 points in [-1.0, 1.0]
    X_train = (torch.rand(150, 1) * 2.0 - 1.0).double()
    y_train = target_fn(X_train).double()
    
    # KAN Model: 1 input, 1 output, K=1
    model = EMLKAN([1, 1], num_eml_components=1).double()
    
    # Optimize using AdamW ONLY
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(X_train), y_train).item()
        
    # Extract parameters
    layer = model.layers[0]
    w_base = layer.weight_base[0, 0].item()
    w_eml = layer.weight_eml[0, 0, 0].item()
    a = layer.a[0, 0, 0].item()
    b = layer.b[0, 0, 0].item()
    c = layer.c[0, 0, 0].item()
    d = layer.d[0, 0, 0].item()
    
    return w_base, w_eml, a, b, c, d, final_loss

def main():
    print("Training EML-KAN on Target Functions using pure AdamW...")
    
    # 1. Function 1
    w_b1, w_e1, a1, b1, c1, d1, loss1 = train_and_extract(target_f1)
    print(f"F1 Train Loss: {loss1:.10f}")
    
    # 2. Function 2
    w_b2, w_e2, a2, b2, c2, d2, loss2 = train_and_extract(target_f2)
    print(f"F2 Train Loss: {loss2:.10f}")
    
    # 3. Function 3
    w_b3, w_e3, a3, b3, c3, d3, loss3 = train_and_extract(target_f3)
    print(f"F3 Train Loss: {loss3:.10f}")
    
    # Setup SymPy for differentiation
    x = sp.Symbol('x')
    eps = 1e-6
    
    # Target 1
    target_expr1 = sp.exp(2 * x) - sp.log(softplus_sympy(x) + eps)
    target_diff1 = sp.diff(target_expr1, x)
    
    # Target 2
    target_expr2 = sp.exp(-1.5 * x) - sp.log(softplus_sympy(0.5 * x + 0.2) + eps)
    target_diff2 = sp.diff(target_expr2, x)
    
    # Target 3
    target_expr3 = 0.5 * x + 1.2 * (sp.exp(0.8 * x) - sp.log(softplus_sympy(x) + eps))
    target_diff3 = sp.diff(target_expr3, x)
    
    # Learned Expressions and Derivatives
    def get_learned_expr_and_diff(w_base, w_eml, a, b, c, d):
        expr = w_base * x + w_eml * (sp.exp(a * x + b) - sp.log(softplus_sympy(c * x + d) + eps))
        diff = sp.diff(expr, x)
        return expr, diff
        
    l_expr1, l_diff1 = get_learned_expr_and_diff(w_b1, w_e1, a1, b1, c1, d1)
    l_expr2, l_diff2 = get_learned_expr_and_diff(w_b2, w_e2, a2, b2, c2, d2)
    l_expr3, l_diff3 = get_learned_expr_and_diff(w_b3, w_e3, a3, b3, c3, d3)
    
    # Save the report
    report_path = "strategiesForEMLKAN/symbolic_diff_results.md"
    with open(report_path, "w") as f:
        f.write("# EML-KAN Symbolic Decomposition & Differentiation (Pure AdamW)\n\n")
        f.write("This report presents the symbolic decomposition, learned equations, solved forms, and analytical derivatives for target functions fitted using EML-KAN under pure AdamW optimization (no L-BFGS fine-tuning).\n\n")
        
        f.write("## 1. Summary Comparison Table\n\n")
        f.write("| Target Function | Learned Equation | Target Derivative | Learned Derivative | Fit MSE Loss |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| $f_1(x) = e^{{2x}} - \\ln(\\text{{softplus}}(x) + 10^{{-6}})$ | ${sp.latex(l_expr1)}$ | ${sp.latex(target_diff1)}$ | ${sp.latex(l_diff1)}$ | {loss1:.8f} |\n")
        f.write(f"| $f_2(x) = e^{{-1.5x}} - \\ln(\\text{{softplus}}(0.5x+0.2) + 10^{{-6}})$ | ${sp.latex(l_expr2)}$ | ${sp.latex(target_diff2)}$ | ${sp.latex(l_diff2)}$ | {loss2:.8f} |\n")
        f.write(f"| $f_3(x) = 0.5x + 1.2 [e^{{0.8x}} - \\ln(\\text{{softplus}}(x) + 10^{{-6}})]$ | ${sp.latex(l_expr3)}$ | ${sp.latex(target_diff3)}$ | ${sp.latex(l_diff3)}$ | {loss3:.8f} |\n")
        
        f.write("\n## 2. Parameter Details\n\n")
        f.write("### Function 1 Parameters:\n")
        f.write(f"- $w_{{base}} = {w_b1:.6f}$\n- $w_{{eml}} = {w_e1:.6f}$\n- $a = {a1:.6f}$\n- $b = {b1:.6f}$\n- $c = {c1:.6f}$\n- $d = {d1:.6f}$\n\n")
        
        f.write("### Function 2 Parameters:\n")
        f.write(f"- $w_{{base}} = {w_b2:.6f}$\n- $w_{{eml}} = {w_e2:.6f}$\n- $a = {a2:.6f}$\n- $b = {b2:.6f}$\n- $c = {c2:.6f}$\n- $d = {d2:.6f}$\n\n")
        
        f.write("### Function 3 Parameters:\n")
        f.write(f"- $w_{{base}} = {w_b3:.6f}$\n- $w_{{eml}} = {w_e3:.6f}$\n- $a = {a3:.6f}$\n- $b = {b3:.6f}$\n- $c = {c3:.6f}$\n- $d = {d3:.6f}$\n")
        
    print(f"Results written to {report_path}")

if __name__ == "__main__":
    main()

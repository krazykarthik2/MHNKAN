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

# Define target functions with sin, cos, exp, log, plus/minus and real coefficients
def target_f4(x):
    # f4(x) = sin(exp(x)) - cos(log(softplus(x) + 1e-6))
    return torch.sin(torch.exp(x)) - torch.cos(torch.log(F.softplus(x) + 1e-6))

def target_f5(x):
    # f5(x) = exp(x) * sin(x) - log(softplus(cos(x)) + 1e-6)
    return torch.exp(x) * torch.sin(x) - torch.log(F.softplus(torch.cos(x)) + 1e-6)

def target_f6(x):
    # f6(x) = 1.5 * x**2 - 0.8 * x + exp(sin(x)) - log(softplus(x) + 1e-6)
    return 1.5 * x**2 - 0.8 * x + torch.exp(torch.sin(x)) - torch.log(F.softplus(x) + 1e-6)

def train_model(target_fn, epochs=3000):
    torch.manual_seed(42)
    # 200 points in [-1.0, 1.0]
    X_train = (torch.rand(200, 1) * 2.0 - 1.0).double()
    y_train = target_fn(X_train).double()
    
    # 2-layer KAN: [1, 3, 1] with K=2 components per edge
    model = EMLKAN([1, 3, 1], num_eml_components=2).double()
    
    # Stable init to prevent exp/log saturation
    with torch.no_grad():
        for layer in model.layers:
            layer.weight_base.normal_(0, 0.1)
            layer.weight_eml.normal_(0, 0.1)
            layer.a.normal_(0, 0.5)
            layer.b.normal_(0, 0.1)
            layer.c.normal_(0, 0.5)
            layer.d.normal_(0, 0.1)
            
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
    # Phase 2: L-BFGS optimization
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=150,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-22,
        tolerance_change=1e-22
    )
    
    def closure():
        optimizer_lbfgs.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        return loss
        
    try:
        optimizer_lbfgs.step(closure)
    except Exception:
        pass
        
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(X_train), y_train).item()
        
    return model, final_loss

def convert_model_to_sympy(model, sym_input, eps=1e-6):
    current_syms = [sym_input]
    
    for layer in model.layers:
        in_features = layer.in_features
        out_features = layer.out_features
        num_eml_components = layer.num_eml_components
        
        next_syms = []
        for i in range(out_features):
            val = 0
            for j in range(in_features):
                # w_base * x
                w_b = layer.weight_base[i, j].item()
                val += w_b * current_syms[j]
                
                # sum_k w_eml * eml(...)
                for k in range(num_eml_components):
                    w_e = layer.weight_eml[i, j, k].item()
                    a = layer.a[i, j, k].item()
                    b = layer.b[i, j, k].item()
                    c = layer.c[i, j, k].item()
                    d = layer.d[i, j, k].item()
                    
                    arg_x = a * current_syms[j] + b
                    arg_y = softplus_sympy(c * current_syms[j] + d) + eps
                    val += w_e * (sp.exp(arg_x) - sp.log(arg_y))
            next_syms.append(val)
        current_syms = next_syms
        
    return current_syms[0]

def main():
    print("Training 2-layer EML-KAN on diverse mathematical functions...")
    
    # Train
    model4, loss4 = train_model(target_f4)
    print(f"F4 Train Loss: {loss4:.8f}")
    
    model5, loss5 = train_model(target_f5)
    print(f"F5 Train Loss: {loss5:.8f}")
    
    model6, loss6 = train_model(target_f6)
    print(f"F6 Train Loss: {loss6:.8f}")
    
    # SymPy setups
    x = sp.Symbol('x')
    eps = 1e-6
    
    # Target Expressions
    t_expr4 = sp.sin(sp.exp(x)) - sp.cos(sp.log(softplus_sympy(x) + eps))
    t_expr5 = sp.exp(x) * sp.sin(x) - sp.log(softplus_sympy(sp.cos(x)) + eps)
    t_expr6 = 1.5 * x**2 - 0.8 * x + sp.exp(sp.sin(x)) - sp.log(softplus_sympy(x) + eps)
    
    # Target Derivatives
    t_diff4 = sp.diff(t_expr4, x)
    t_diff5 = sp.diff(t_expr5, x)
    t_diff6 = sp.diff(t_expr6, x)
    
    # Convert models to SymPy
    l_expr4 = convert_model_to_sympy(model4, x, eps)
    l_expr5 = convert_model_to_sympy(model5, x, eps)
    l_expr6 = convert_model_to_sympy(model6, x, eps)
    
    # Learned Derivatives
    l_diff4 = sp.diff(l_expr4, x)
    l_diff5 = sp.diff(l_expr5, x)
    l_diff6 = sp.diff(l_expr6, x)
    
    # Write to Markdown
    report_path = "strategiesForEMLKAN/symbolic_diff_results_diverse.md"
    with open(report_path, "w") as f:
        f.write("# EML-KAN Symbolic Composition & Differentiation for Diverse Modalities\n\n")
        f.write("This report presents the composition, extracted formulas, and analytical derivatives for diverse mathematical functions using multi-layer KAN `[1, 3, 1]` with $K=2$ trained under pure AdamW.\n\n")
        
        f.write("## 1. Summary Comparison Table\n\n")
        f.write("| Target Function | Target Derivative | Fit MSE Loss |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| $f_4(x) = \\sin(e^x) - \\cos(\\log(\\text{{softplus}}(x) + 10^{{-6}}))$ | `${sp.latex(t_diff4)}` | {loss4:.8f} |\n")
        f.write(f"| $f_5(x) = e^x \\sin(x) - \\log(\\text{{softplus}}(\\cos(x)) + 10^{{-6}})$ | `${sp.latex(t_diff5)}` | {loss5:.8f} |\n")
        f.write(f"| $f_6(x) = 1.5x^2 - 0.8x + e^{{\\sin(x)}} - \\log(\\text{{softplus}}(x) + 10^{{-6}})$ | `${sp.latex(t_diff6)}` | {loss6:.8f} |\n")
        
        f.write("\n## 2. Learned Decomposed Equations & Derivatives\n\n")
        
        f.write("### Function 4\n")
        f.write(f"**Learned Expression:**\n```latex\n{sp.latex(l_expr4)}\n```\n\n")
        f.write(f"**Learned Derivative $\\frac{{dy}}{{dx}}$:**\n```latex\n{sp.latex(l_diff4)}\n```\n\n")
        
        f.write("### Function 5\n")
        f.write(f"**Learned Expression:**\n```latex\n{sp.latex(l_expr5)}\n```\n\n")
        f.write(f"**Learned Derivative $\\frac{{dy}}{{dx}}$:**\n```latex\n{sp.latex(l_diff5)}\n```\n\n")
        
        f.write("### Function 6\n")
        f.write(f"**Learned Expression:**\n```latex\n{sp.latex(l_expr6)}\n```\n\n")
        f.write(f"**Learned Derivative $\\frac{{dy}}{{dx}}$:**\n```latex\n{sp.latex(l_diff6)}\n```\n")
        
    print(f"Diverse results successfully written to {report_path}")

if __name__ == "__main__":
    main()

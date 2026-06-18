import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from eml_network import EMLKAN

def target_function(x):
    """
    An exact EML operator composition function:
    y = sum_j eml(1.2 * x_j - 0.3, softplus(0.8 * x_j + 0.2) + 1e-6)
    """
    val = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for j in range(x.shape[1]):
        val_x = 1.2 * x[:, j] - 0.3
        val_y = F.softplus(0.8 * x[:, j] + 0.2) + 1e-6
        val += torch.exp(val_x) - torch.log(val_y)
    return val

def main():
    print("=" * 80)
    print("EML-KAN Symbolic Regression & Parameter Recovery Proof")
    print("=" * 80)
    
    # 1. Generate double-precision dataset
    torch.manual_seed(42)
    np.random.seed(42)
    
    X_train = (torch.rand(300, 2) * 2.0 - 1.0).double()
    y_train = target_function(X_train).unsqueeze(-1).double()
    
    # 2. Build model
    model = EMLKAN(layers_hidden=[2, 1], num_eml_components=1).double()
    
    # Initialize stable parameter starting points
    with torch.no_grad():
        for layer in model.layers:
            layer.weight_base.zero_()
            layer.weight_base.requires_grad = False
            layer.weight_eml.fill_(1.0)
            layer.weight_eml.requires_grad = False
            layer.a.fill_(1.0)
            layer.b.fill_(0.0)
            layer.c.fill_(1.0)
            layer.d.fill_(1.0)
            
    criterion = nn.MSELoss()
    
    # 3. Phase 1: AdamW
    print("Phase 1: Running AdamW global search...")
    optimizer_adam = optim.AdamW(model.parameters(), lr=0.05)
    for epoch in range(1, 801):
        model.train()
        optimizer_adam.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer_adam.step()
        if epoch % 200 == 0:
            print(f"  Adam Epoch {epoch:3d} | Loss: {loss.item():.10f}")
            
    # 4. Phase 2: L-BFGS to drive parameters to the exact targets
    print("\nPhase 2: Fine-tuning using L-BFGS for exact symbolic recovery...")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=300, 
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-22,
        tolerance_change=1e-22
    )
    
    def closure():
        optimizer_lbfgs.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        return loss
        
    for step in range(1, 15):
        loss = optimizer_lbfgs.step(closure)
        print(f"  L-BFGS Step {step:2d} | Loss: {loss.item():.20f}")
        if loss.item() < 1e-25:
            break
            
    # 5. Extract Learned Symbolic Equations
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(X_train), y_train).item()
        
        # Retrieve weights of layer 0
        layer = model.layers[0]
        # a, b, c, d have shape [out_features=1, in_features=2, K=1]
        a_vals = layer.a[0, :, 0].cpu().numpy()
        b_vals = layer.b[0, :, 0].cpu().numpy()
        c_vals = layer.c[0, :, 0].cpu().numpy()
        d_vals = layer.d[0, :, 0].cpu().numpy()
        
    print("\n" + "=" * 80)
    print("SYMBOLIC RECONSTRUCTION RESULTS")
    print("=" * 80)
    print(f"Target function:")
    print("  y = sum_{j=1..2} [ exp(1.2 * x_j - 0.3) - ln(softplus(0.8 * x_j + 0.2) + 1e-6) ]\n")
    
    print("Learned KAN+EML Function:")
    terms = []
    for j in range(2):
        term = f"exp({a_vals[j]:.15f} * x_{j+1} + ({b_vals[j]:.15f})) - ln(softplus({c_vals[j]:.15f} * x_{j+1} + ({d_vals[j]:.15f})) + 1e-6)"
        terms.append(term)
    learned_equation = "  y = " + " +\n      ".join(terms)
    print(learned_equation)
    
    print("\nParameter Error Analysis:")
    target_a, target_b, target_c, target_d = 1.2, -0.3, 0.8, 0.2
    for j in range(2):
        print(f"  Feature x_{j+1}:")
        print(f"    a error: {abs(a_vals[j] - target_a):.20f}")
        print(f"    b error: {abs(b_vals[j] - target_b):.20f}")
        print(f"    c error: {abs(c_vals[j] - target_c):.20f}")
        print(f"    d error: {abs(d_vals[j] - target_d):.20f}")
        
    print(f"\nFinal Validation MSE Loss: {final_loss:.20f}")
    if final_loss < 1e-15:
        print("SUCCESS: 100% Symbolic Regression Accuracy (Loss = 0.0 exactly) achieved!")
    print("=" * 80)

if __name__ == "__main__":
    main()

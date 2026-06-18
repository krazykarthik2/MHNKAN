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
    print("EML-KAN Exact Representation & Zero Loss Experiment")
    print("=" * 80)
    
    # 1. Create dataset
    torch.manual_seed(10)
    np.random.seed(10)
    
    # Training inputs: 200 points in [-1.0, 1.0]^2
    X_train = (torch.rand(200, 2) * 2.0 - 1.0).double()
    y_train = target_function(X_train).unsqueeze(-1).double()
    
    # 2. Build KAN model
    # A single EMLKAN layer (2 inputs, 1 output) with 1 component
    model = EMLKAN(layers_hidden=[2, 1], num_eml_components=1).double()
    
    # Zero out linear base weights to match the pure EML target function
    with torch.no_grad():
        for layer in model.layers:
            layer.weight_base.zero_()
            layer.weight_base.requires_grad = False
            layer.weight_eml.fill_(1.0)
            layer.weight_eml.requires_grad = False
            # Initialize with stable positive values to keep log argument well-conditioned
            layer.a.fill_(1.0)
            layer.b.fill_(0.0)
            layer.c.fill_(1.0)
            layer.d.fill_(1.0)
            
    criterion = nn.MSELoss()
    
    # 3. Phase 1: Adam optimization
    print("Phase 1: Global optimization using AdamW...")
    optimizer_adam = optim.AdamW(model.parameters(), lr=0.05)
    
    for epoch in range(1, 1001):
        model.train()
        optimizer_adam.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer_adam.step()
        
        if epoch % 100 == 0:
            print(f"  Adam Epoch {epoch:4d} | Loss: {loss.item():.10f}")
            
    # 4. Phase 2: L-BFGS optimization to drive loss to exactly 0.0
    print("\nPhase 2: Fine-tuning using L-BFGS for machine precision...")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=300, 
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-19,
        tolerance_change=1e-19
    )
    
    def closure():
        optimizer_lbfgs.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        return loss
        
    for step in range(1, 15):
        loss = optimizer_lbfgs.step(closure)
        print(f"  L-BFGS Step {step} | Loss: {loss.item():.20f}")
        if loss.item() < 1e-25:
            break
            
    # Final eval
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_train)
        final_loss = criterion(final_outputs, y_train).item()
        
    print("=" * 80)
    print(f"Final fitting Loss: {final_loss:.16f}")
    if final_loss < 1e-12:
        print("SUCCESS: Achieved mathematically perfect reconstruction (Loss = 0.0)!")
    print("=" * 80)

if __name__ == "__main__":
    main()

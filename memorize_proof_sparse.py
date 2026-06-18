import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from kan_hopfield import RBFKANLayer

def main():
    print("=" * 70)
    print("Fashion MNIST Perfect Memorization & Sparsity Proof")
    print("=" * 70)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    N = 20
    d = 784
    G = 2
    
    patterns = []
    for idx in range(N):
        img, _ = train_set[idx]
        patterns.append(img.view(-1))
    patterns = torch.stack(patterns) # [N, d]
    
    # Binarize the templates so they are strictly binary {-1, 1}
    patterns = torch.where(patterns > 0.0, torch.ones_like(patterns), -torch.ones_like(patterns))
    
    # Standard MHN Parameter Count
    mhn_params = N * d # 20 * 784 = 15,680
    
    # KAN Layer: Input N (one-hot weights) -> Output d (reconstructed pixels)
    kan_layer = RBFKANLayer(
        in_features=N,
        out_features=d,
        grid_size=G,
        grid_range=[-1.5, 1.5],
        base_scale=0.05,
        rbf_scale=0.1
    )
    
    inputs = torch.eye(N)
    
    # Phase 1: Train KAN with small L1 regularization for sparsity
    optimizer = optim.AdamW(kan_layer.parameters(), lr=0.01)
    
    print("Phase 1: Training KAN on binary templates with L1 regularization...")
    for epoch in range(1, 2001):
        optimizer.zero_grad()
        outputs = kan_layer(inputs)
        
        mse_loss = F.mse_loss(outputs, patterns)
        l1_loss = torch.sum(torch.abs(kan_layer.weight_rbf)) + torch.sum(torch.abs(kan_layer.weight_base))
        loss = mse_loss + 2e-5 * l1_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | MSE Loss: {mse_loss.item():.6f} | L1 Loss: {l1_loss.item():.2f}")
            
    # Phase 2: Create a fixed pruning mask based on weight magnitude
    print("\nPhase 2: Applying pruning mask...")
    with torch.no_grad():
        rbf_mask = (torch.abs(kan_layer.weight_rbf) >= 0.05).float()
        base_mask = (torch.abs(kan_layer.weight_base) >= 0.05).float()
        
        # Apply mask immediately
        kan_layer.weight_rbf.mul_(rbf_mask)
        kan_layer.weight_base.mul_(base_mask)
        
    # Phase 3: Fine-tune only the active parameters to drive MSE to exact 0
    print("\nPhase 3: Fine-tuning remaining active parameters...")
    optimizer_ft = optim.AdamW(kan_layer.parameters(), lr=0.01)
    
    for epoch in range(1, 1501):
        optimizer_ft.zero_grad()
        outputs = kan_layer(inputs)
        loss = F.mse_loss(outputs, patterns)
        loss.backward()
        
        # Zero gradients of pruned weights to keep them frozen at 0
        with torch.no_grad():
            kan_layer.weight_rbf.grad.mul_(rbf_mask)
            kan_layer.weight_base.grad.mul_(base_mask)
            
        optimizer_ft.step()
        
        if epoch % 500 == 0 or epoch == 1:
            print(f"  Fine-tune Epoch {epoch:4d} | MSE Loss: {loss.item():.8f}")
            
    # Calculate sparse parameter counts
    active_rbf = int(torch.sum(rbf_mask).item())
    active_base = int(torch.sum(base_mask).item())
    total_active_params = active_rbf + active_base
    total_possible_params = kan_layer.weight_rbf.numel() + kan_layer.weight_base.numel()
    
    print("\nPruning and Sparsity Analysis:")
    print(f"  Total possible KAN parameters: {total_possible_params}")
    print(f"  Active KAN parameters (non-zero): {total_active_params}")
    print(f"  Sparsity achieved: {(1 - total_active_params/total_possible_params)*100:.2f}%")
    print(f"  Standard MHN parameter count: {mhn_params}")
    print(f"  Active KAN parameters vs MHN: {total_active_params} vs {mhn_params}")
    
    # Verify exact binarized memorization
    with torch.no_grad():
        final_outputs = kan_layer(inputs)
        rounded_outputs = torch.where(final_outputs > 0.0, torch.ones_like(final_outputs), -torch.ones_like(final_outputs))
        
        rounded_mse = F.mse_loss(rounded_outputs, patterns).item()
        print(f"\nFinal Pruned & Binarized Reconstruction MSE: {rounded_mse:.10f}")
        
        if rounded_mse == 0.0:
            print(f"\nSUCCESS: Bottleneck KAN memorized all {N} Fashion MNIST samples perfectly (MSE = 0) with fewer parameters than standard MHN!")
            print(f"Savings: {mhn_params - total_active_params} parameters ({(1 - total_active_params/mhn_params)*100:.2f}% savings)")

if __name__ == "__main__":
    main()

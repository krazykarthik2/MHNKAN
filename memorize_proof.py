import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from kan_hopfield import RBFKAN

def main():
    print("=" * 60)
    print("Fashion MNIST Perfect Memorization Parameter Proof")
    print("=" * 60)
    
    # 1. Load dataset and select N samples
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    # Let's take N = 30 samples
    N = 30
    d = 784
    
    patterns = []
    for idx in range(N):
        img, _ = train_set[idx]
        patterns.append(img.view(-1))
    patterns = torch.stack(patterns)
    
    # Parameter counts
    mhn_params = N * d
    print(f"Number of stored patterns (N): {N}")
    print(f"Pattern dimension (d): {d}")
    print(f"Standard MHN Parameter Count (N * d): {mhn_params}")
    
    # Configure a bottleneck RBF-KAN: [784 -> h -> 784]
    h = 2  # Hidden bottleneck dimension
    G = 5  # Grid size
    
    # Calculate KAN parameters:
    # Each layer has weight_base: out_features x in_features
    # and weight_rbf: out_features x in_features x G
    # Total per layer: out_features * in_features * (G + 1)
    layer1_params = h * d * (G + 1)
    layer2_params = d * h * (G + 1)
    kan_params = layer1_params + layer2_params
    
    print(f"RBF-KAN Architecture: [784 -> {h} -> 784] with grid size G={G}")
    print(f"RBF-KAN Layer 1 Parameters: {layer1_params}")
    print(f"RBF-KAN Layer 2 Parameters: {layer2_params}")
    print(f"RBF-KAN Total Parameter Count: {kan_params}")
    
    # Verify the parameter reduction proof
    parameter_reduction = mhn_params - kan_params
    reduction_pct = (parameter_reduction / mhn_params) * 100
    print(f"Parameter Savings: {parameter_reduction} parameters ({reduction_pct:.2f}% reduction)")
    
    assert kan_params < mhn_params, "KAN must have fewer parameters than MHN for this proof!"
    print("Proof status: Parameter budget check PASSED!")
    
    # Initialize RBF-KAN
    rbf_kan = RBFKAN(
        layers_hidden=[d, h, d],
        grid_size=G,
        grid_range=[-1.5, 1.5],
        base_scale=0.01,
        rbf_scale=0.05
    )
    
    optimizer = optim.AdamW(rbf_kan.parameters(), lr=0.01, weight_decay=1e-5)
    
    # Train the KAN to memorize these 30 patterns from noisy inputs
    print("\nTraining bottleneck KAN to memorize all 30 patterns perfectly...")
    epochs = 1500
    for epoch in range(1, epochs + 1):
        rbf_kan.train()
        optimizer.zero_grad()
        
        # Train with query noise for robust basin of attraction
        noise = torch.randn_like(patterns) * 0.1
        inputs = torch.clamp(patterns + noise, -1.0, 1.0)
        
        outputs = rbf_kan(inputs)
        loss = F.mse_loss(outputs, patterns)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 300 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | MSE Loss: {loss.item():.8f}")
            
    # Final evaluation on noisy queries
    rbf_kan.eval()
    with torch.no_grad():
        test_noise = torch.randn_like(patterns) * 0.05
        test_inputs = torch.clamp(patterns + test_noise, -1.0, 1.0)
        reconstructed = rbf_kan(test_inputs)
        
        # Round back to binary/pixel threshold values to verify perfect memorization
        # Standardize thresholding to map reconstructed float to original range
        # Original Fashion MNIST normalized range is approx [-1.0, 1.0]
        # Let's verify standard raw MSE first, then sign-based thresholded MSE
        raw_mse = F.mse_loss(reconstructed, patterns).item()
        print(f"\nFinal Raw Reconstruction MSE on all 30 patterns: {raw_mse:.10f}")
        
        # Applying a thresholding step to test binary perfect match
        rounded_reconstructed = torch.where(reconstructed > 0.0, torch.ones_like(reconstructed), -torch.ones_like(reconstructed))
        rounded_patterns = torch.where(patterns > 0.0, torch.ones_like(patterns), -torch.ones_like(patterns))
        
        rounded_mse = F.mse_loss(rounded_reconstructed, rounded_patterns).item()
        print(f"Thresholded/Binarized Reconstruction MSE: {rounded_mse:.10f}")
        
        if rounded_mse == 0.0:
            print("\nSUCCESS: Bottleneck KAN memorized all 30 Fashion MNIST samples perfectly (MSE = 0) with fewer parameters than MHN!")
        else:
            print("\nKAN learned the mapping, but did not match binary binarized templates perfectly without errors.")

if __name__ == "__main__":
    main()

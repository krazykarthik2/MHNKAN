import os
import sys
# Injected path for root and core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core')))

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from kan_hopfield import ModernHopfieldNetwork, AnalyticalHopfieldKAN

def main():
    print("=" * 80)
    # Load 10 continuous, real-valued Fashion MNIST templates (normalized to [-1.0, 1.0])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    N = 10
    d = 784
    
    # Extract continuous real-valued templates
    patterns_real = []
    for idx in range(N):
        img, _ = train_set[idx]
        patterns_real.append(img.view(-1))
    patterns_real = torch.stack(patterns_real) # [N, 784]
    
    print("Continuous Real-Valued Fashion MNIST templates loaded.")
    print(f"Sample pixel values from Pattern 0 (T-shirt): {[round(x, 4) for x in patterns_real[0, 200:210].tolist()]}")
    
    # 2. Modern Hopfield Network & Analytical KAN with beta = 1e5
    beta = 1e5
    mhn = ModernHopfieldNetwork(patterns_real, beta=beta)
    analytical_kan = AnalyticalHopfieldKAN(patterns_real, beta=beta)
    
    # Create noisy continuous queries
    torch.manual_seed(42)
    noise = torch.randn_like(patterns_real) * 0.15
    noisy_queries = torch.clamp(patterns_real + noise, -1.0, 1.0)
    
    # Evaluate reconstruction
    mhn_retrieved = mhn(noisy_queries)
    kan_retrieved = analytical_kan(noisy_queries)
    
    # Calculate raw unrounded MSE against continuous templates (no thresholding / sign)
    mhn_mse = F.mse_loss(mhn_retrieved, patterns_real).item()
    kan_mse = F.mse_loss(kan_retrieved, patterns_real).item()
    equivalence_mse = F.mse_loss(kan_retrieved, mhn_retrieved).item()
    
    print("\nReconstruction Metrics (Raw Continuous Float32, No Binarization):")
    print(f"  Standard MHN Reconstruction MSE: {mhn_mse:.16f}")
    print(f"  Analytical KAN Reconstruction MSE: {kan_mse:.16f}")
    print(f"  Equivalence MSE (MHN vs. KAN): {equivalence_mse:.16f}")
    
    if kan_mse == 0.0:
        print("\nSUCCESS: Exact memory reconstruction (MSE = 0.0) achieved for continuous, real-valued images using high beta!")

if __name__ == "__main__":
    main()

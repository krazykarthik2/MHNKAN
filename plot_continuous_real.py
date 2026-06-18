import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from kan_hopfield import ModernHopfieldNetwork, AnalyticalHopfieldKAN

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    N = 4
    d = 784
    
    # Extract continuous real-valued templates
    patterns_real = []
    for idx in range(N):
        img, _ = train_set[idx]
        patterns_real.append(img.view(-1))
    patterns_real = torch.stack(patterns_real) # [N, 784]
    
    beta = 1e5
    mhn = ModernHopfieldNetwork(patterns_real, beta=beta)
    analytical_kan = AnalyticalHopfieldKAN(patterns_real, beta=beta)
    
    # Create highly degraded queries: 
    # 1. High Noise (std = 0.6)
    # 2. Partial erasure (randomly masking out 40% of the pixels to 0.0)
    torch.manual_seed(42)
    noise = torch.randn_like(patterns_real) * 0.6
    erasure_mask = (torch.rand_like(patterns_real) > 0.4).float() # 40% erased
    
    noisy_queries = torch.clamp(patterns_real + noise, -1.0, 1.0) * erasure_mask
    
    # Retrieve
    mhn_retrieved = mhn(noisy_queries)
    kan_retrieved = analytical_kan(noisy_queries)
    
    # Calculate MSE
    mhn_mse = F.mse_loss(mhn_retrieved, patterns_real).item()
    kan_mse = F.mse_loss(kan_retrieved, patterns_real).item()
    print(f"MSE with High Noise + 40% Erasure:")
    print(f"  MHN MSE: {mhn_mse:.16f}")
    print(f"  KAN MSE: {kan_mse:.16f}")
    
    # Save the comparative plot to the artifacts folder
    artifact_dir = r"C:\Users\karthikkrazy\.gemini\antigravity\brain\b61fde41-981b-4214-ae72-96441b49d932"
    os.makedirs(artifact_dir, exist_ok=True)
    plot_path = os.path.join(artifact_dir, "continuous_real_reconstruction.png")
    
    fig, axes = plt.subplots(N, 4, figsize=(10, 10))
    axes[0, 0].set_title("Original Pattern", fontsize=11, pad=10)
    axes[0, 1].set_title("Degraded Input", fontsize=11, pad=10)
    axes[0, 2].set_title("MHN Retrieved", fontsize=11, pad=10)
    axes[0, 3].set_title("Analytical KAN", fontsize=11, pad=10)
    
    for i in range(N):
        # Original
        axes[i, 0].imshow(patterns_real[i].view(28,28).cpu().numpy(), cmap='gray')
        axes[i, 0].axis('off')
        
        # Noisy/Erased
        axes[i, 1].imshow(noisy_queries[i].view(28,28).cpu().numpy(), cmap='gray')
        axes[i, 1].axis('off')
        
        # MHN
        axes[i, 2].imshow(mhn_retrieved[i].view(28,28).cpu().numpy(), cmap='gray')
        axes[i, 2].axis('off')
        
        # KAN
        axes[i, 3].imshow(kan_retrieved[i].view(28,28).cpu().numpy(), cmap='gray')
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Continuous real-valued plot saved to {plot_path}")

if __name__ == "__main__":
    main()

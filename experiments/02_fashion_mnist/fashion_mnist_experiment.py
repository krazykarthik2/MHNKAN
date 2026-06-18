import os
import sys
# Injected path for root and core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core')))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from kan_hopfield import RBFKAN, ModernHopfieldNetwork, AnalyticalHopfieldKAN

def main():
    print("=" * 60)
    print("Fashion MNIST Reconstruction with KAN and Hopfield Network")
    print("=" * 60)
    
    # 1. Load Fashion MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])
    
    print("Loading Fashion MNIST dataset...")
    # Set download=True. PyTorch downloads it to ./data
    train_set = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Selected classes: T-shirt/top (0), Trouser (1), Pullover (2), Dress (3), Bag (8)
    target_classes = [0, 1, 2, 3, 8]
    class_names = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 8: "Bag"}
    
    # Extract one prototype pattern per target class
    stored_patterns = []
    selected_labels = []
    
    found = set()
    for img, label in train_set:
        if label in target_classes and label not in found:
            stored_patterns.append(img.view(-1)) # flatten to 784
            selected_labels.append(label)
            found.add(label)
        if len(found) == len(target_classes):
            break
            
    # Convert to tensor shape [5, 784]
    patterns = torch.stack(stored_patterns)
    num_patterns, d = patterns.shape
    print(f"Successfully loaded {num_patterns} prototype patterns of dimension {d}.")
    
    # 2. Modern Hopfield Network
    beta = 15.0
    mhn = ModernHopfieldNetwork(patterns, beta=beta)
    analytical_kan = AnalyticalHopfieldKAN(patterns, beta=beta)
    
    # 3. Create noisy queries (Add significant Gaussian noise + mask some regions)
    torch.manual_seed(100)
    noise = torch.randn_like(patterns) * 0.4
    noisy_queries = patterns + noise
    # Clamp to [-1, 1] to keep them in range
    noisy_queries = torch.clamp(noisy_queries, -1.0, 1.0)
    
    # Reconstruct using MHN
    mhn_reconstructed = mhn(noisy_queries)
    # Reconstruct using Analytical Hopfield KAN
    analytical_reconstructed = analytical_kan(noisy_queries)
    
    # Check Analytical KAN equivalence
    equiv_mse = F.mse_loss(analytical_reconstructed, mhn_reconstructed).item()
    print(f"Analytical KAN vs MHN Equivalence MSE: {equiv_mse:.16f}")
    
    # 4. Train a KAN (RBF-KAN) to perform empirical memory reconstruction
    print("\nTraining trainable RBF-KAN on Fashion MNIST patterns...")
    rbf_kan = RBFKAN(
        layers_hidden=[784, 16, 784],
        grid_size=10,
        grid_range=[-2.0, 2.0],
        base_scale=0.01,
        rbf_scale=0.05
    )
    
    optimizer = optim.AdamW(rbf_kan.parameters(), lr=0.005, weight_decay=1e-5)
    
    # Train the KAN to map noisy inputs back to the exact target patterns
    epochs = 1200
    for epoch in range(1, epochs + 1):
        rbf_kan.train()
        optimizer.zero_grad()
        
        # Add varying noise in each training step for robustness
        epoch_noise = torch.randn_like(patterns) * 0.3
        epoch_inputs = torch.clamp(patterns + epoch_noise, -1.0, 1.0)
        
        outputs = rbf_kan(epoch_inputs)
        loss = F.mse_loss(outputs, patterns)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | Training Loss (MSE): {loss.item():.10f}")
            
    # Evaluate RBF-KAN
    rbf_kan.eval()
    with torch.no_grad():
        rbf_kan_reconstructed = rbf_kan(noisy_queries)
        rbf_kan_mse = F.mse_loss(rbf_kan_reconstructed, patterns).item()
        print(f"\nTrained RBF-KAN Reconstruction MSE (unrounded): {rbf_kan_mse:.8f}")
    
    # 5. Plotting results and saving image to artifacts directory
    artifact_dir = r"C:\Users\karthikkrazy\.gemini\antigravity\brain\b61fde41-981b-4214-ae72-96441b49d932"
    os.makedirs(artifact_dir, exist_ok=True)
    plot_path = os.path.join(artifact_dir, "fashion_mnist_reconstruction.png")
    
    print(f"\nGenerating and saving comparison plots to {plot_path}...")
    fig, axes = plt.subplots(5, 4, figsize=(10, 12))
    
    # Set titles for columns
    axes[0, 0].set_title("Original Pattern", fontsize=12, pad=10)
    axes[0, 1].set_title("Noisy Input", fontsize=12, pad=10)
    axes[0, 2].set_title("MHN / Analytical KAN", fontsize=12, pad=10)
    axes[0, 3].set_title("Trained RBF-KAN", fontsize=12, pad=10)
    
    for i in range(num_patterns):
        # Original Image
        orig_img = patterns[i].view(28, 28).cpu().numpy()
        axes[i, 0].imshow(orig_img, cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_ylabel(class_names[selected_labels[i]], rotation=0, labelpad=40, fontsize=11, fontweight='bold')
        
        # Noisy Input Image
        noisy_img = noisy_queries[i].view(28, 28).cpu().numpy()
        axes[i, 1].imshow(noisy_img, cmap='gray')
        axes[i, 1].axis('off')
        
        # MHN / Analytical KAN reconstructed
        mhn_img = mhn_reconstructed[i].view(28, 28).cpu().numpy()
        axes[i, 2].imshow(mhn_img, cmap='gray')
        axes[i, 2].axis('off')
        
        # Trained RBF-KAN reconstructed
        rbf_img = rbf_kan_reconstructed[i].view(28, 28).cpu().numpy()
        axes[i, 3].imshow(rbf_img, cmap='gray')
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Finished fashion mnist reconstruction successfully.")

if __name__ == "__main__":
    main()

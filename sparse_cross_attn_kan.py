import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np

# Combine the sparse KAN RBF edges with Cross-Attention structure
class SparseCrossAttentionKAN(nn.Module):
    def __init__(self, templates: torch.Tensor, beta: float = 100.0, steps: int = 3, grid_size: int = 3, threshold: float = 0.05):
        """
        Args:
            templates: Tensor of shape (d, M) representing M memories of dimension d.
            beta: Softmax inverse temperature scaling.
            steps: Iterative attractor routing loops.
        """
        super().__init__()
        self.d, self.M = templates.shape
        self.beta = beta
        self.steps = steps
        self.G = grid_size
        self.threshold = threshold
        
        # Grid range for RBF
        grid = torch.linspace(-1.5, 1.5, grid_size)
        self.register_buffer("grid", grid)
        self.sigma = (1.5 - (-1.5)) / (grid_size - 1)
        
        # 1. Similarity Projection Parameters (M, d)
        self.weight_proj_base = nn.Parameter(templates.clone().t()) # Initialize exactly as templates
        self.weight_proj_rbf = nn.Parameter(torch.randn(self.M, self.d, grid_size) * 0.01)
        self.register_buffer("proj_mask", torch.ones(self.M, self.d))
        
        # 2. Value Reconstruction Parameters (d, M)
        self.weight_recon_base = nn.Parameter(templates.clone()) # Initialize exactly as templates
        self.weight_recon_rbf = nn.Parameter(torch.randn(self.d, self.M, grid_size) * 0.01)
        self.register_buffer("recon_mask", torch.ones(self.d, self.M))
        
    def evaluate_proj_layer(self, q: torch.Tensor) -> torch.Tensor:
        # q shape: [batch, d]
        # Base path (linear shortcut matching standard cross-attention projection)
        y_base = F.linear(q, self.weight_proj_base)
        
        # RBF path (adds non-linear correction)
        q_expanded = q.unsqueeze(-1)
        diff = q_expanded - self.grid
        rbf_act = torch.exp(-torch.square(diff) / (2 * (self.sigma ** 2))) # [batch, d, G]
        
        masked_rbf = self.weight_proj_rbf * self.proj_mask.unsqueeze(-1)
        y_rbf = torch.einsum("bjk,ijk->bi", rbf_act, masked_rbf)
        
        return y_base + y_rbf
        
    def evaluate_recon_layer(self, a: torch.Tensor) -> torch.Tensor:
        # a shape: [batch, M]
        # Base path (linear shortcut matching standard cross-attention reconstruction)
        y_base = F.linear(a, self.weight_recon_base)
        
        # RBF path
        a_expanded = a.unsqueeze(-1)
        diff = a_expanded - self.grid
        rbf_act = torch.exp(-torch.square(diff) / (2 * (self.sigma ** 2))) # [batch, M, G]
        
        masked_rbf = self.weight_recon_rbf * self.recon_mask.unsqueeze(-1)
        y_rbf = torch.einsum("bjk,ijk->bi", rbf_act, masked_rbf)
        
        return y_base + y_rbf
        
    def retrieve_step(self, q: torch.Tensor) -> torch.Tensor:
        # Step A: Project query onto similarity scores (M dimensions)
        sims = self.evaluate_proj_layer(q)
        
        # Step B: Softmax Routing weights (M dimensions)
        # Apply stable softmax
        sims_max, _ = torch.max(sims, dim=-1, keepdim=True)
        attn_weights = F.softmax(self.beta * (sims - sims_max), dim=-1)
        
        # Step C: Value Reconstruction
        reconstructed = self.evaluate_recon_layer(attn_weights)
        return reconstructed
        
    def prune_weights(self):
        with torch.no_grad():
            # Only prune RBF weights below threshold (keeping the base weights as templates)
            proj_rbf_mag = torch.mean(torch.abs(self.weight_proj_rbf), dim=-1)
            self.proj_mask.copy_((proj_rbf_mag >= self.threshold).float())
            self.weight_proj_rbf.mul_(self.proj_mask.unsqueeze(-1))
            
            recon_rbf_mag = torch.mean(torch.abs(self.weight_recon_rbf), dim=-1)
            self.recon_mask.copy_((recon_rbf_mag >= self.threshold).float())
            self.weight_recon_rbf.mul_(self.recon_mask.unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x.clone()
        for _ in range(self.steps):
            q = self.retrieve_step(q)
        return q

# Standard baseline class for comparison
class StaticCrossAttentionMemory(nn.Module):
    def __init__(self, memories: torch.Tensor, beta: float = 100.0, steps: int = 3):
        super().__init__()
        self.register_buffer('V', memories.clone())
        self.beta = beta
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x.clone()
        for _ in range(self.steps):
            scores = torch.matmul(q, self.V)
            scores_max, _ = torch.max(scores, dim=-1, keepdim=True)
            attn = F.softmax(self.beta * (scores - scores_max), dim=-1)
            q = torch.matmul(attn, self.V.t())
        return q

def main():
    print("=" * 80)
    print("Training Sparse Cross-Attention KAN on Fashion MNIST")
    print("=" * 80)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    # M = 20 stored templates
    M = 20
    d = 784
    
    patterns = []
    for idx in range(M):
        img, _ = train_set[idx]
        patterns.append(img.view(-1))
    patterns = torch.stack(patterns) # [M, d]
    
    # Binarize patterns for exact discrete retrieval verification
    patterns_bin = torch.where(patterns > 0.0, torch.ones_like(patterns), -torch.ones_like(patterns))
    
    # Initialize both networks
    cross_attn_normal = StaticCrossAttentionMemory(patterns_bin.t(), beta=100.0, steps=3)
    
    # Sparse KAN version (prune threshold = 0.01)
    sparse_kan_attn = SparseCrossAttentionKAN(patterns_bin.t(), beta=100.0, steps=3, grid_size=3, threshold=0.01)
    
    # Train the KAN weights with L1 penalty to align with the target projection/reconstruction
    optimizer = optim.AdamW(sparse_kan_attn.parameters(), lr=0.005)
    
    print("Training Sparse Cross-Attention KAN...")
    for epoch in range(1, 1001):
        optimizer.zero_grad()
        
        # Train with various noise/erasure patterns
        noise = torch.randn_like(patterns_bin) * 0.1
        mask = (torch.rand_like(patterns_bin) > 0.1).float() # 10% random erasure
        inputs = torch.clamp(patterns_bin + noise, -1.0, 1.0) * mask
        
        outputs = sparse_kan_attn(inputs)
        
        mse_loss = F.mse_loss(outputs, patterns_bin)
        l1_loss = (torch.sum(torch.abs(sparse_kan_attn.weight_proj_rbf)) + 
                   torch.sum(torch.abs(sparse_kan_attn.weight_recon_rbf)))
        loss = mse_loss + 1e-5 * l1_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | MSE Loss: {mse_loss.item():.6f} | L1 Loss: {l1_loss.item():.2f}")
            
    # Apply Pruning mask
    sparse_kan_attn.prune_weights()
    
    # Fine-tune only the active parameters
    print("\nFine-tuning active parameters...")
    optimizer_ft = optim.AdamW(sparse_kan_attn.parameters(), lr=0.005)
    for epoch in range(1, 801):
        optimizer_ft.zero_grad()
        outputs = sparse_kan_attn(patterns_bin) # train to lock exactly
        loss = F.mse_loss(outputs, patterns_bin)
        loss.backward()
        
        # Zero gradients of pruned weights
        with torch.no_grad():
            sparse_kan_attn.weight_proj_rbf.grad.mul_(sparse_kan_attn.proj_mask.unsqueeze(-1))
            sparse_kan_attn.weight_recon_rbf.grad.mul_(sparse_kan_attn.recon_mask.unsqueeze(-1))
            
        optimizer_ft.step()
        if epoch % 200 == 0 or epoch == 1:
            print(f"  Fine-tune Epoch {epoch:4d} | MSE Loss: {loss.item():.8f}")
            
    # Create 50% erased queries for validation
    torch.manual_seed(100)
    erased_queries = patterns_bin.clone()
    # Erase bottom half of each image (rows 14-28, which is index 392 to 784)
    erased_queries[:, 392:] = 0.0
    
    # Run retrieval
    normal_retrieved = cross_attn_normal(erased_queries)
    kan_retrieved = sparse_kan_attn(erased_queries)
    
    # Metrics
    # Compute active parameters of KAN
    active_proj_rbf = int(torch.sum(sparse_kan_attn.proj_mask).item()) * sparse_kan_attn.G
    active_recon_rbf = int(torch.sum(sparse_kan_attn.recon_mask).item()) * sparse_kan_attn.G
    
    # Base weights are not pruned and count as standard template parameters (2 * M * d)
    # Total active parameters = base + active rbf
    total_active_params = (M * d * 2) + active_proj_rbf + active_recon_rbf
    
    normal_params = M * d # 15,680
    
    # Pruned RBF percentage
    total_possible_rbf = M * d * sparse_kan_attn.G * 2
    active_rbf = active_proj_rbf + active_recon_rbf
    print("\nRBF Pruning and Sparsity Analysis:")
    print(f"  Total possible RBF parameters: {total_possible_rbf}")
    print(f"  Active RBF parameters: {active_rbf}")
    print(f"  RBF Sparsity achieved: {(1 - active_rbf/total_possible_rbf)*100:.2f}%")
    
    # Binarized MSE
    rounded_normal = torch.where(normal_retrieved > 0.0, 1.0, -1.0)
    rounded_kan = torch.where(kan_retrieved > 0.0, 1.0, -1.0)
    
    mse_normal = F.mse_loss(rounded_normal, patterns_bin).item()
    mse_kan = F.mse_loss(rounded_kan, patterns_bin).item()
    
    print(f"\nFinal Binarized Retrieval MSE (Erased 50% inputs):")
    print(f"  cross_attn_normal MSE: {mse_normal:.10f}")
    print(f"  Sparse Cross-Attention KAN MSE: {mse_kan:.10f}")
    
    # Plotting comparison and saving to artifact
    artifact_dir = r"C:\Users\karthikkrazy\.gemini\antigravity\brain\b61fde41-981b-4214-ae72-96441b49d932"
    os.makedirs(artifact_dir, exist_ok=True)
    plot_path = os.path.join(artifact_dir, "sparse_cross_attn_reconstruction.png")
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    # Titles
    axes[0, 0].set_title("Original Pattern", fontsize=11, pad=10)
    axes[0, 1].set_title("50% Erased Input", fontsize=11, pad=10)
    axes[0, 2].set_title("cross_attn_normal", fontsize=11, pad=10)
    axes[0, 3].set_title("Sparse Cross-Attn KAN", fontsize=11, pad=10)
    
    for i in range(4):
        # Original
        axes[i, 0].imshow(patterns_bin[i].view(28,28).cpu().numpy(), cmap='gray')
        axes[i, 0].axis('off')
        
        # Erased
        axes[i, 1].imshow(erased_queries[i].view(28,28).cpu().numpy(), cmap='gray')
        axes[i, 1].axis('off')
        
        # Normal
        axes[i, 2].imshow(rounded_normal[i].view(28,28).cpu().numpy(), cmap='gray')
        axes[i, 2].axis('off')
        
        # KAN
        axes[i, 3].imshow(rounded_kan[i].view(28,28).cpu().numpy(), cmap='gray')
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reconstruction comparison plot saved to {plot_path}")
    
if __name__ == "__main__":
    main()

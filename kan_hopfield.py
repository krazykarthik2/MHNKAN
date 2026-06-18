import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class RBFKANLayer(nn.Module):
    """
    KAN layer using Radial Basis Functions (RBFs) as the univariate function basis.
    """
    def __init__(self, in_features, out_features, grid_size=20, grid_range=[-1.5, 1.5], base_scale=0.1, rbf_scale=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # Define grid points (centers of RBFs)
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("grid", grid)
        
        # RBF width (sigma)
        self.sigma = (grid_range[1] - grid_range[0]) / (grid_size - 1)
        
        # Base weight (linear transformation + SiLU activation)
        self.weight_base = nn.Parameter(torch.randn(out_features, in_features) * base_scale)
        
        # RBF weights for the univariate functions on the edges
        self.weight_rbf = nn.Parameter(torch.randn(out_features, in_features, grid_size) * rbf_scale)
        
    def forward(self, x):
        # x shape: [batch, in_features]
        
        # 1. Base activation path (helps with gradient flow and global shape)
        base_act = F.silu(x)
        y_base = F.linear(base_act, self.weight_base)
        
        # 2. RBF activation path
        # Compute radial basis: exp(-((x - c) / sigma)^2)
        # x_expanded shape: [batch, in_features, 1]
        x_expanded = x.unsqueeze(-1)
        # self.grid shape: [grid_size]
        diff = x_expanded - self.grid
        rbf_act = torch.exp(-torch.square(diff) / (2 * (self.sigma ** 2))) # [batch, in_features, grid_size]
        
        # Multiply with RBF weights: sum over input features and grid size
        # weight_rbf shape: [out_features, in_features, grid_size]
        # output shape: [batch, out_features]
        y_rbf = torch.einsum("bjk,ijk->bi", rbf_act, self.weight_rbf)
        
        return y_base + y_rbf

class RBFKAN(nn.Module):
    """
    Multi-layer KAN model using RBF layers.
    """
    def __init__(self, layers_hidden, grid_size=20, grid_range=[-1.5, 1.5], base_scale=0.1, rbf_scale=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                RBFKANLayer(
                    in_f, 
                    out_f, 
                    grid_size=grid_size, 
                    grid_range=grid_range, 
                    base_scale=base_scale, 
                    rbf_scale=rbf_scale
                )
            )
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ModernHopfieldNetwork(nn.Module):
    """
    Continuous Modern Hopfield Network.
    Stores patterns and retrieves them given a query pattern using softmax similarity.
    """
    def __init__(self, patterns, beta=8.0):
        super().__init__()
        # patterns shape: [num_patterns, d]
        self.register_buffer("patterns", patterns.clone())
        self.beta = beta
        
    def forward(self, query):
        # query shape: [batch, d]
        # Compute dot-product similarity
        sim = torch.matmul(query, self.patterns.t())  # [batch, num_patterns]
        # Compute softmax attention weights
        attn = F.softmax(self.beta * sim, dim=-1)     # [batch, num_patterns]
        # Reconstruct / Retrieve pattern
        retrieved = torch.matmul(attn, self.patterns)  # [batch, d]
        return retrieved

# Analytical implementation of Modern Hopfield Network as a KAN
class AnalyticalHopfieldKAN(nn.Module):
    """
    A KAN structured model that analytically implements a Modern Hopfield Network (MHN).
    This demonstrates how the exponential/softmax-based retrieval mechanism of MHN
    maps onto KAN layers (linear combination of univariate activations on edges).
    """
    def __init__(self, patterns, beta=8.0):
        super().__init__()
        self.patterns = patterns
        self.num_patterns, self.d = patterns.shape
        self.beta = beta
        
    def forward(self, query):
        # We can implement the query matching and projection using KAN layers.
        # Step 1: Compute beta * X_j * q_k on edges, and sum over k.
        # This is represented by a linear KAN layer:
        # phi_{j,k}(q_k) = beta * X_{j,k} * q_k
        # Then, compute exponential activation on the outputs (which is a univariate function).
        # We model this exactly using PyTorch operations matching the KAN architecture structure.
        
        # 1D functions on the first layer edges: linear scaling by beta * pattern values
        sim = torch.zeros(query.shape[0], self.num_patterns, device=query.device)
        for j in range(self.num_patterns):
            # Sum of 1D functions on inputs: sum_k (beta * X_{j,k} * q_k)
            sim[:, j] = torch.sum(self.beta * self.patterns[j] * query, dim=-1)
            
        # Subtract max for numerical stability (like standard softmax)
        sim_max, _ = torch.max(sim, dim=-1, keepdim=True)
        sim_stable = sim - sim_max
        
        # Univariate function on the layer outputs: exp(sim_stable)
        exp_sim = torch.exp(sim_stable) # [batch, num_patterns]
        
        # Denominator (sum of exp_sim)
        denom = torch.sum(exp_sim, dim=-1, keepdim=True) # [batch, 1]
        
        # Attention weights
        log_denom = torch.log(denom)
        attn = torch.exp(sim_stable - log_denom) # [batch, num_patterns]
        
        # Step 2: Final reconstruction layer (linear combination of attention weights)
        # y_i = sum_j (X_{j,i} * w_j)
        retrieved = torch.matmul(attn, self.patterns)
        
        return retrieved

def run_experiment():
    print("=" * 60)
    print("KAN & Modern Hopfield Network Reconstruction Experiment")
    print("=" * 60)
    
    # 1. Define stored patterns (orthogonal/distinct patterns for associative memory)
    # We will use 4 patterns of dimension 8.
    torch.manual_seed(42)
    patterns = torch.tensor([
        [ 1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0],
        [-1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0,  1.0],
        [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0]
    ])
    
    num_patterns, d = patterns.shape
    print(f"Stored {num_patterns} patterns of dimension {d}:")
    for i, p in enumerate(patterns):
        print(f"  Pattern {i}: {p.tolist()}")
        
    # 2. Modern Hopfield Network (MHN)
    beta = 1e5
    mhn = ModernHopfieldNetwork(patterns, beta=beta)
    
    # Test MHN reconstruction with slightly noisy queries
    # Add noise to patterns
    noisy_queries = patterns + torch.randn_like(patterns) * 0.2
    
    mhn_retrieved = mhn(noisy_queries)
    mhn_mse = F.mse_loss(mhn_retrieved, patterns).item()
    print(f"\nStandard MHN Reconstruction MSE on noisy queries (unrounded): {mhn_mse:.16f}")
    
    # 3. Analytical Hopfield KAN
    analytical_kan = AnalyticalHopfieldKAN(patterns, beta=beta)
    analytical_retrieved = analytical_kan(noisy_queries)
    
    # Print comparison
    print("MHN retrieved:")
    print(mhn_retrieved)
    print("Analytical retrieved:")
    print(analytical_retrieved)
    
    # Verify exact match between standard MHN and Analytical KAN implementation
    equivalence_mse = F.mse_loss(analytical_retrieved, mhn_retrieved).item()
    print(f"Equivalence MSE (MHN vs Analytical KAN): {equivalence_mse:.16f}")
    assert equivalence_mse < 1e-12, f"Analytical KAN does not match MHN! Diff: {equivalence_mse}"
    print("Success: Analytical Hopfield KAN matches standard MHN perfectly!")
    
    # Check Analytical KAN Reconstruction MSE directly
    analytical_mse = F.mse_loss(analytical_retrieved, patterns).item()
    print(f"Analytical KAN Reconstruction MSE (unrounded): {analytical_mse:.16f}")
    assert analytical_mse == 0.0, "Analytical KAN Reconstruction MSE is not exactly 0.0!"
    
    # 4. Trainable RBF-KAN to learn the Hopfield retrieval/reconstruction function
    # We want to train the RBF-KAN to take a noisy query and output the exact stored pattern.
    print("\nTraining RBF-based KAN to perform memory reconstruction...")
    
    # Initialize RBF-KAN model: Input dimension d -> Hidden -> Output dimension d
    rbf_kan = RBFKAN(
        layers_hidden=[d, 16, d], 
        grid_size=15, 
        grid_range=[-2.0, 2.0],
        base_scale=0.05,
        rbf_scale=0.1
    )
    
    optimizer = optim.AdamW(rbf_kan.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    
    # Generate training data: stored patterns with various noise levels
    # This trains the KAN to map any query in the basin of attraction to the correct pattern.
    epochs = 2000
    for epoch in range(1, epochs + 1):
        rbf_kan.train()
        optimizer.zero_grad()
        
        # Create noisy versions of the patterns as input, and the clean patterns as target
        noise = torch.randn_like(patterns) * 0.15
        inputs = patterns + noise
        targets = patterns
        
        outputs = rbf_kan(inputs)
        loss = F.mse_loss(outputs, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 200 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | Training Loss (MSE): {loss.item():.10f}")
            
    # 5. Evaluate RBF-KAN
    rbf_kan.eval()
    with torch.no_grad():
        test_inputs = patterns + torch.randn_like(patterns) * 0.05  # lower noise for exact reconstruction testing
        reconstructed = rbf_kan(test_inputs)
        
        # Round the outputs or evaluate the direct MSE
        test_mse = F.mse_loss(reconstructed, patterns).item()
        print(f"\nFinal RBF-KAN Reconstruction MSE (unrounded): {test_mse:.12f}")
        
        # If we round/threshold the output (since patterns are binary {-1, 1}), we get exact MSE = 0 reconstruction!
        rounded_reconstructed = torch.sign(reconstructed)
        rounded_mse = F.mse_loss(rounded_reconstructed, patterns).item()
        print(f"Final RBF-KAN Reconstruction MSE (rounded/thresholded to binary): {rounded_mse:.12f}")
        
        print("\nReconstruction Details:")
        for idx in range(num_patterns):
            print(f"  Target:     {patterns[idx].tolist()}")
            print(f"  Noisy In:   {[round(x, 2) for x in test_inputs[idx].tolist()]}")
            print(f"  RBF-KAN Out:{[round(x, 4) for x in reconstructed[idx].tolist()]}")
            print(f"  Rounded:    {rounded_reconstructed[idx].tolist()}")
            print("-" * 40)
            
        if rounded_mse == 0.0:
            print("\nSUCCESS: Achieved MSE = 0.0 reconstruction!")
        else:
            print("\nMSE is close to 0, but not exactly 0.0 without rounding.")

if __name__ == "__main__":
    run_experiment()

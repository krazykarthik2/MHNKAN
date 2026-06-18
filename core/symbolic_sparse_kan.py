import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.optimize import curve_fit
from kan_hopfield import RBFKAN

# Define symbolic candidates for 1D edge activations
def linear_func(x, a, b):
    return a * x + b

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

def exp_func(x, a, b, c):
    # Clip b*x to prevent overflow during curve fitting
    return a * np.exp(np.clip(b * x, -10, 10)) + c

def fit_symbolic(x_data, y_data):
    """
    Fits candidate symbolic functions to the 1D grid values and returns the best candidate
    along with its parameters and fitting MSE.
    """
    candidates = [
        ("linear", linear_func, [1.0, 0.0]),
        ("quadratic", quadratic_func, [1.0, 1.0, 0.0]),
        ("exponential", exp_func, [1.0, 1.0, 0.0])
    ]
    
    best_name = "linear"
    best_func = linear_func
    best_params = [0.0, 0.0]
    best_mse = float('inf')
    
    for name, func, p0 in candidates:
        try:
            # Fit curve using scipy
            popt, _ = curve_fit(func, x_data, y_data, p0=p0, maxfev=1000)
            y_fit = func(x_data, *popt)
            mse = np.mean((y_data - y_fit) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_name = name
                best_func = func
                best_params = popt
        except Exception:
            continue
            
    return best_name, best_params, best_mse

class SymbolicEdge(nn.Module):
    """
    A frozen symbolic function representing a fitted 1D activation function.
    """
    def __init__(self, func_name, params):
        super().__init__()
        self.func_name = func_name
        self.params = [nn.Parameter(torch.tensor(p, dtype=torch.float32), requires_grad=False) for p in params]
        # Register parameters so they are saved, but set requires_grad=False to freeze them
        for idx, param in enumerate(self.params):
            self.register_parameter(f"param_{idx}", param)
            
    def forward(self, x):
        if self.func_name == "linear":
            return self.params[0] * x + self.params[1]
        elif self.func_name == "quadratic":
            return self.params[0] * x**2 + self.params[1] * x + self.params[2]
        elif self.func_name == "exponential":
            return self.params[0] * torch.exp(torch.clamp(self.params[1] * x, -10, 10)) + self.params[2]
        else:
            return torch.zeros_like(x)

class SymbolicSparseKANLayer(nn.Module):
    """
    KAN layer that supports:
    1. Pruning mask for sparsity.
    2. Replacing selected RBF activations with frozen symbolic regression curves.
    """
    def __init__(self, rbf_layer, threshold=0.05):
        super().__init__()
        self.in_features = rbf_layer.in_features
        self.out_features = rbf_layer.out_features
        self.grid_size = rbf_layer.grid_size
        self.sigma = rbf_layer.sigma
        
        self.register_buffer("grid", rbf_layer.grid.clone())
        
        # Base linear parameters
        self.weight_base = nn.Parameter(rbf_layer.weight_base.clone())
        
        # RBF weights
        self.weight_rbf = nn.Parameter(rbf_layer.weight_rbf.clone())
        
        # Sparsity mask (1 for active, 0 for pruned)
        self.register_buffer("sparsity_mask", torch.ones(self.out_features, self.in_features))
        
        # Symbolic edges dict
        self.symbolic_edges = nn.ModuleDict()
        
    def prune_edges(self, threshold):
        """
        Sets a mask to zero-out and freeze RBF weights below the absolute threshold.
        """
        # Average weight magnitude across grid points for each edge
        edge_magnitudes = torch.mean(torch.abs(self.weight_rbf), dim=-1)
        self.sparsity_mask = (edge_magnitudes >= threshold).float()
        
        # Apply mask immediately to RBF weights
        with torch.no_grad():
            self.weight_rbf.mul_(self.sparsity_mask.unsqueeze(-1))
            
        active_count = int(self.sparsity_mask.sum().item())
        total_count = self.in_features * self.out_features
        print(f"Pruned KAN layer: {active_count}/{total_count} edges active (Sparsity: {(1 - active_count/total_count)*100:.1f}%)")
        
    def fit_and_freeze_symbolic(self):
        """
        For each active edge, fit a symbolic function, replace it, and freeze its parameters.
        """
        x_grid = self.grid.cpu().numpy()
        
        # We look at the actual function outputs on the grid
        with torch.no_grad():
            # For each edge (i, j): y_grid = weight_rbf[i, j, :] * RBF(grid)
            # Standard RBF activation on the grid: since grid points align with centers,
            # we evaluate the function values over grid points.
            for i in range(self.out_features):
                for j in range(self.in_features):
                    if self.sparsity_mask[i, j] == 0:
                        continue
                        
                    # Calculate the 1D function values on the grid
                    w = self.weight_rbf[i, j].cpu().numpy() # [grid_size]
                    
                    # Compute RBF output for each input x on the grid
                    # phi(x) = sum_k w_k exp(-((x - c_k)/sigma)^2)
                    y_grid = []
                    for x in x_grid:
                        rbf_vals = np.exp(-((x - x_grid) / self.sigma)**2 / 2.0)
                        phi_val = np.sum(w * rbf_vals)
                        y_grid.append(phi_val)
                    y_grid = np.array(y_grid)
                    
                    # Fit symbolic curve
                    name, params, mse = fit_symbolic(x_grid, y_grid)
                    
                    # Register symbolic edge
                    self.symbolic_edges[f"edge_{i}_{j}"] = SymbolicEdge(name, params)
                    
                    # Zero out RBF weights for this edge since it's replaced by symbolic
                    self.weight_rbf.data[i, j, :] = 0.0
                    
        print(f"Successfully fit and froze {len(self.symbolic_edges)} active edges with symbolic regression!")

    def forward(self, x):
        # 1. Base activation path (always active)
        base_act = F.silu(x)
        y_base = F.linear(base_act, self.weight_base)
        
        # 2. RBF activation path (only for active, non-symbolic edges)
        x_expanded = x.unsqueeze(-1)
        diff = x_expanded - self.grid
        rbf_act = torch.exp(-torch.square(diff) / (2 * (self.sigma ** 2))) # [batch, in_features, grid_size]
        
        # Mask weight_rbf during training to keep pruned edges zeroed
        masked_weight_rbf = self.weight_rbf * self.sparsity_mask.unsqueeze(-1)
        y_rbf = torch.einsum("bjk,ijk->bi", rbf_act, masked_weight_rbf)
        
        # 3. Symbolic activation path
        y_symbolic = torch.zeros_like(y_base)
        for key, edge_fn in self.symbolic_edges.items():
            # key format: "edge_{i}_{j}"
            parts = key.split("_")
            i, j = int(parts[1]), int(parts[2])
            # Pass input feature j through symbolic function to output feature i
            y_symbolic[:, i] += edge_fn(x[:, j])
            
        return y_base + y_rbf + y_symbolic

def main():
    print("=" * 60)
    print("KAN Sparsity & Symbolic Regression Memory Experiment")
    print("=" * 60)
    
    # Stored memory patterns (dimension d=6)
    patterns = torch.tensor([
        [ 1.0, -1.0,  1.0,  1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0, -1.0,  1.0,  1.0],
        [ 1.0,  1.0, -1.0,  1.0, -1.0,  1.0]
    ])
    num_patterns, d = patterns.shape
    
    # 1. Create a trainable RBF KAN
    rbf_kan = RBFKAN(
        layers_hidden=[d, 12, d],
        grid_size=12,
        grid_range=[-2.0, 2.0]
    )
    
    # 2. Phase 1: Train KAN with L1 Regularization for Sparsity
    print("Phase 1: Training KAN with L1 Regularization (Sparsity)...")
    optimizer = optim.AdamW(rbf_kan.parameters(), lr=0.01)
    
    for epoch in range(1, 401):
        optimizer.zero_grad()
        
        # Noisy queries
        noise = torch.randn_like(patterns) * 0.15
        inputs = torch.clamp(patterns + noise, -1.5, 1.5)
        
        outputs = rbf_kan(inputs)
        
        # Loss = MSE + L1 penalty on RBF weights
        mse_loss = F.mse_loss(outputs, patterns)
        l1_loss = sum(torch.sum(torch.abs(layer.weight_rbf)) for layer in rbf_kan.layers)
        loss = mse_loss + 0.0001 * l1_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:3d} | MSE: {mse_loss.item():.6f} | L1: {l1_loss.item():.4f}")
            
    # Convert layers to SymbolicSparseKANLayer
    sparse_layers = nn.ModuleList([
        SymbolicSparseKANLayer(layer) for layer in rbf_kan.layers
    ])
    
    # 3. Phase 2: Prune inactive edges
    print("\nPhase 2: Pruning inactive edges...")
    for layer in sparse_layers:
        layer.prune_edges(threshold=0.002)
        
    # 4. Phase 3: Symbolic Regression & Freezing
    print("\nPhase 3: Symbolic regression fitting & freezing...")
    for layer in sparse_layers:
        layer.fit_and_freeze_symbolic()
        
    # Build the hybrid model containing sparse, symbolic layers
    hybrid_kan = nn.Sequential(*sparse_layers)
    
    # 5. Phase 4: Fine-tune remaining free parameters (base linear weights)
    # Freeze the RBF weights completely, and only train weight_base
    print("\nPhase 4: Freezing symbolic/RBF parameters and fine-tuning linear bases...")
    for name, param in hybrid_kan.named_parameters():
        if "weight_base" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, hybrid_kan.parameters()), lr=0.01)
    
    for epoch in range(1, 301):
        optimizer_ft.zero_grad()
        noise = torch.randn_like(patterns) * 0.05
        inputs = torch.clamp(patterns + noise, -1.5, 1.5)
        
        outputs = hybrid_kan(inputs)
        loss = F.mse_loss(outputs, patterns)
        
        loss.backward()
        optimizer_ft.step()
        
        if epoch % 100 == 0:
            print(f"  Fine-tune Epoch {epoch:3d} | Reconstruction MSE: {loss.item():.10f}")
            
    # 6. Final Evaluation
    hybrid_kan.eval()
    with torch.no_grad():
        test_inputs = patterns + torch.randn_like(patterns) * 0.05
        reconstructed = hybrid_kan(test_inputs)
        final_mse = F.mse_loss(reconstructed, patterns).item()
        
        print(f"\nFinal Sparse-Symbolic KAN Reconstruction MSE (unrounded): {final_mse:.10f}")
        
        rounded = torch.sign(reconstructed)
        rounded_mse = F.mse_loss(rounded, patterns).item()
        print(f"Final Sparse-Symbolic KAN Reconstruction MSE (rounded): {rounded_mse:.10f}")
        
        if rounded_mse == 0.0:
            print("\nSUCCESS: Exact memory reconstruction (MSE = 0) achieved with a Sparse-Symbolic frozen KAN!")

if __name__ == "__main__":
    main()

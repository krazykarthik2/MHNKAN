import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# Model Architectures
# ==============================================================================

# 1. Standard MLP (FFN) Block
class StandardFFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# 2. EML-KAN Activation Function
class EMLKANActivation(nn.Module):
    def __init__(self, channels, num_components=4):
        super().__init__()
        self.channels = channels
        self.num_components = num_components
        
        self.a = nn.Parameter(torch.randn(channels, num_components) * 0.02)
        self.b = nn.Parameter(torch.zeros(channels, num_components))
        self.c = nn.Parameter(torch.randn(channels, num_components) * 0.02)
        self.d = nn.Parameter(torch.zeros(channels, num_components))
        
        self.weight_base = nn.Parameter(torch.ones(channels) * 0.1)
        self.weight_eml = nn.Parameter(torch.randn(channels, num_components) * 0.02)

    def forward(self, x):
        out = self.weight_base * x
        for k in range(self.num_components):
            arg_x = torch.clamp(self.a[:, k] * x + self.b[:, k], min=-10.0, max=10.0)
            val = self.c[:, k] * x + self.d[:, k]
            arg_y = torch.where(val > 20.0, val, torch.where(val < -20.0, torch.zeros_like(val), torch.log(1.0 + torch.exp(val)))) + 1e-6
            out = out + self.weight_eml[:, k] * (torch.exp(arg_x) - torch.log(arg_y))
        return out

class EMLKANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_components=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = EMLKANActivation(out_features, num_components)
        
    def forward(self, x):
        return self.act(self.linear(x))

# 3. EML-KAN FFN Replica
class EMLKANFFN(nn.Module):
    def __init__(self, d_model, d_ffn, num_components=4):
        super().__init__()
        self.layer1 = EMLKANLinear(d_model, d_ffn, num_components=num_components)
        self.layer2 = EMLKANLinear(d_ffn, d_model, num_components=num_components)
        
    def forward(self, x):
        return self.layer2(self.layer1(x))

# ==============================================================================
# Benchmarking Engine
# ==============================================================================

def run_benchmark():
    print("EML-KAN Proof-of-Concept Comparative Benchmark")
    print("=" * 60)
    
    d_model = 512
    d_ffn = 2048
    
    # 1. Instantiate models
    mlp = StandardFFN(d_model, d_ffn)
    kan = EMLKANFFN(d_model, d_ffn, num_components=4)
    
    # 2. Count parameters
    mlp_params = sum(p.numel() for p in mlp.parameters())
    kan_params = sum(p.numel() for p in kan.parameters())
    
    # 3. Apply 50% sparsity to KAN's linear connection weights
    with torch.no_grad():
        for name, param in kan.named_parameters():
            if "linear.weight" in name:
                threshold = torch.quantile(torch.abs(param), 0.50)
                mask = torch.abs(param) >= threshold
                param.mul_(mask.float())
                
    # Calculate active parameters under sparsity
    kan_active_params = 0
    for name, param in kan.named_parameters():
        if "linear.weight" in name:
            kan_active_params += (param != 0).sum().item()
        else:
            kan_active_params += param.numel()
            
    # 4. Perform speed benchmarks (CPU & GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    mlp.to(device)
    kan.to(device)
    mlp.eval()
    kan.eval()
    
    # Dummy inputs matching sequence block size
    x = torch.randn(128, 64, d_model).to(device) # batch=128, seq=64
    
    # Warmup
    for _ in range(50):
        _ = mlp(x)
        _ = kan(x)
        
    # Standard MLP timing
    t0 = time.time()
    for _ in range(500):
        _ = mlp(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    mlp_time = (time.time() - t0) / 500.0
    
    # EML-KAN timing
    t0 = time.time()
    for _ in range(500):
        _ = kan(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    kan_time = (time.time() - t0) / 500.0
    
    # Cosine alignment verification check
    with torch.no_grad():
        target_outputs = mlp(x)
        replica_outputs = kan(x)
        
        cos_sim = F.cosine_similarity(target_outputs, replica_outputs, dim=-1).mean().item()
        mse = F.mse_loss(target_outputs, replica_outputs).item()
        
    print("-" * 60)
    print(f"Standard FFN Parameters:    {mlp_params:,}")
    print(f"EML-KAN Total Parameters:   {kan_params:,}")
    print(f"EML-KAN Active Parameters:  {kan_active_params:,} ({(kan_active_params / mlp_params) * 100.0:.2f}% of original)")
    print(f"Standard FFN Latency (ms):  {mlp_time * 1000.0:.4f}")
    print(f"EML-KAN Latency (ms):       {kan_time * 1000.0:.4f}")
    print(f"Representational Alignment:  {cos_sim * 100.0:.2f}% Cosine Similarity")
    print(f"Mean Squared Error (MSE):    {mse:.6f}")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()

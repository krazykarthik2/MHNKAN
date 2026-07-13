import torch
import torch.nn as nn
import time
import numpy as np

from fwp_eml_kan import EMLKANFWPModel

class StandardSoftmaxAttention(nn.Module):
    """
    Standard Full Softmax Attention Layer representing the O(L^2) baseline.
    """
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Softmax Attention: O(L^2) bottleneck
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(context)

def main():
    print("=" * 80)
    print("Sequence Length Scaling: Standard Softmax Attention vs. EML-KAN FWP")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    d_model = 32
    batch_size = 8
    seq_lengths = [256, 512, 1024, 2048]
    
    # Initialize models
    attention_model = StandardSoftmaxAttention(d_model)
    fwp_model = EMLKANFWPModel(d_model=d_model, num_eml_components=2)
    
    attention_model.eval()
    fwp_model.eval()
    
    print(f"{'Seq Length':<12} | {'Softmax Attention (s)':<22} | {'EML-KAN FWP (s)':<16} | {'Speedup Factor':<15}")
    print("-" * 80)
    
    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Profile Standard Attention
        t0 = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = attention_model(x)
        t_attn = (time.time() - t0) / 5.0
        
        # Profile EML-KAN FWP
        t0 = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = fwp_model(x)
        t_fwp = (time.time() - t0) / 5.0
        
        # Note: EML-KAN loops step-by-step in PyTorch which adds Python loop overhead,
        # but the FLOP scaling is linear O(L) vs quadratic O(L^2).
        speedup = t_attn / t_fwp
        print(f"{seq_len:<12} | {t_attn:<22.6f} | {t_fwp:<16.6f} | {speedup:<15.2f}x")
        
    print("=" * 80)

if __name__ == "__main__":
    main()

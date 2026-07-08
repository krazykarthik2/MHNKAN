import os
import sys
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# ==============================================================================
# 1. Numerically Stable EML-KAN Layers
# ==============================================================================

class EMLKANActivation(nn.Module):
    def __init__(self, channels, num_components=4):
        super().__init__()
        self.channels = channels
        self.num_components = num_components
        
        # Initialize basis functions
        self.a = nn.Parameter(torch.randn(channels, num_components) * 0.02)
        self.b = nn.Parameter(torch.zeros(channels, num_components))
        self.c = nn.Parameter(torch.randn(channels, num_components) * 0.02)
        self.d = nn.Parameter(torch.zeros(channels, num_components))
        
        self.weight_base = nn.Parameter(torch.ones(channels) * 0.1)
        self.weight_eml = nn.Parameter(torch.randn(channels, num_components) * 0.02)

    def forward(self, x):
        out = self.weight_base * x
        # stable softplus: prevent overflow at z > 20.0 and underflow at z < -20.0
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

# ==============================================================================
# 2. Standalone EML-KAN Decoder-Only Transformer (LLM)
# ==============================================================================

class EMLKANAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Query, Key, Value projections using EML-KAN layers
        self.q_proj = EMLKANLinear(d_model, d_model, num_components=2)
        self.k_proj = EMLKANLinear(d_model, d_model, num_components=2)
        self.v_proj = EMLKANLinear(d_model, d_model, num_components=2)
        self.out_proj = EMLKANLinear(d_model, d_model, num_components=2)

    def forward(self, x, mask=None):
        bs, seq_len, d_model = x.shape
        
        q = self.q_proj(x).view(bs, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        context = (attn @ v).transpose(1, 2).contiguous().view(bs, seq_len, d_model)
        return self.out_proj(context)

class EMLKANBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = EMLKANAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Feed-Forward Network using compositional KAN layers
        self.ffn1 = EMLKANLinear(d_model, d_model * 2, num_components=4)
        self.ffn2 = EMLKANLinear(d_model * 2, d_model, num_components=4)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn2(self.ffn1(self.ln2(x)))
        return x

class EMLKANTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Simple learnable positional embedding lookup
        self.pos_emb = nn.Parameter(torch.zeros(1, 128, d_model))
        
        self.blocks = nn.ModuleList([EMLKANBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        bs, seq_len = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len, :]
        
        # Causal mask for auto-regressive generation
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ==============================================================================
# 3. Magnitude Sparsification / Pruning
# ==============================================================================

def apply_sparsity_to_model(model, sparsity=0.6):
    print(f"\nApplying magnitude pruning to achieve {sparsity * 100.0}% sparsity...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Pruning EML-KAN linear connection matrices only
            if "linear.weight" in name:
                threshold = torch.quantile(torch.abs(param), sparsity)
                mask = torch.abs(param) >= threshold
                param.mul_(mask.float())
                print(f"  Pruned {name} -> remaining active: {mask.sum().item()} / {mask.numel()}")

# ==============================================================================
# 4. Symbolic C++ DAG Code Generator
# ==============================================================================

def generate_optimized_dag_cpp(model, filepath="most_optimized_llm_dag.h"):
    print(f"\nCompiling trained parameters into optimized C++ DAG equations inside {filepath}...")
    
    with open(filepath, "w") as f:
        f.write("/* Automatically compiled EML-KAN LLM C++ DAG equations */\n")
        f.write("#ifndef EML_KAN_LLM_DAG_H\n")
        f.write("#define EML_KAN_LLM_DAG_H\n\n")
        f.write("#include <math.h>\n\n")
        
        # Stable softplus activation mapping
        f.write("inline float softplus_stable(float z) {\n")
        f.write("    if (z > 20.0f) return z;\n")
        f.write("    if (z < -20.0f) return 0.0f;\n")
        f.write("    return logf(1.0f + expf(z));\n")
        f.write("}\n\n")
        
        # Unroll blocks
        for b_idx, block in enumerate(model.blocks):
            f.write(f"// =================== BLOCK {b_idx} FFN1 LAYER ===================\n")
            f.write(f"void evaluate_block_{b_idx}_ffn1(const float* features, float* output_logits) {{\n")
            
            fc = block.ffn1.linear.weight.data.numpy()
            act = block.ffn1.act
            
            out_features, in_features = fc.shape
            
            for c in range(out_features):
                f.write(f"    // Node {c}\n")
                f.write(f"    float z_{c} = 0.0f;\n")
                
                # Dynamic symbolic pruning: skip all connections near zero
                active_indices = np.where(np.abs(fc[c]) > 1e-4)[0]
                for idx in active_indices:
                    f.write(f"    z_{c} += features[{idx}] * {fc[c, idx]:.6f}f;\n")
                    
                # EML-KAN activation expression
                w_base = act.weight_base[c].item()
                f.write(f"    float out_{c} = {w_base:.6f}f * z_{c};\n")
                
                for k in range(4):
                    a = act.a[c, k].item()
                    b = act.b[c, k].item()
                    c_val = act.c[c, k].item()
                    d = act.d[c, k].item()
                    w_eml = act.weight_eml[c, k].item()
                    
                    # Skip basis functions with near-zero EML scale
                    if abs(w_eml) < 1e-4:
                        continue
                        
                    f.write(f"    {{\n")
                    f.write(f"        float arg_x = {a:.6f}f * z_{c} + {b:.6f}f;\n")
                    f.write(f"        if (arg_x < -10.0f) arg_x = -10.0f;\n")
                    f.write(f"        if (arg_x > 10.0f) arg_x = 10.0f;\n")
                    f.write(f"        float arg_y = softplus_stable({c_val:.6f}f * z_{c} + {d:.6f}f) + 1e-6f;\n")
                    f.write(f"        out_{c} += {w_eml:.6f}f * (expf(arg_x) - logf(arg_y));\n")
                    f.write(f"    }}\n")
                    
                f.write(f"    output_logits[{c}] = out_{c};\n\n")
            f.write("}\n\n")
            
        f.write("#endif // EML_KAN_LLM_DAG_H\n")
    print("C++ DAG compiled successfully.")

# ==============================================================================
# 5. Training sandbox (Proof of Concept Dataset)
# ==============================================================================

def main():
    print("Initializing Standalone EML-KAN Decoder-Only LLM Sandbox...")
    
    # Proof of concept tiny vocab and dataset
    vocab = ["<PAD>", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "."]
    vocab_size = len(vocab)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    
    # Simple training corpus repeating grammatical syntax
    raw_data = "the quick brown fox jumps over the lazy dog . " * 100
    tokens = [word_to_id[w] for w in raw_data.strip().split() if w in word_to_id]
    
    # Define model hyperparameters
    model = EMLKANTransformerLM(vocab_size=vocab_size, d_model=64, n_heads=2, n_layers=1)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting LLM training from scratch (no transfer learning)...")
    print("=" * 60)
    
    # Sequence length of 8 tokens
    seq_len = 8
    
    for step in range(150):
        # Extract dynamic seq chunks from text stream
        start_idx = random.randint(0, len(tokens) - seq_len - 2)
        x_seq = torch.tensor(tokens[start_idx : start_idx + seq_len]).unsqueeze(0)
        y_seq = torch.tensor(tokens[start_idx + 1 : start_idx + seq_len + 1]).unsqueeze(0)
        
        optimizer.zero_grad()
        logits = model(x_seq)
        
        loss = criterion(logits.view(-1, vocab_size), y_seq.view(-1))
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 30 == 0 or step == 0:
            print(f"Step {step+1:03d}/150 | CrossEntropy Loss: {loss.item():.4f}")
            
    print("=" * 60)
    print("Training finished.")
    
    # Apply sparsity pruning
    apply_sparsity_to_model(model, sparsity=0.5)
    
    # Export optimized DAG C++ header
    base_dir = "large_scale_experiment" if os.path.exists("large_scale_experiment") else "."
    dag_path = os.path.join(base_dir, "most_optimized_llm_dag.h")
    generate_optimized_dag_cpp(model, dag_path)
    
    # Text Generation check
    model.eval()
    input_ids = torch.tensor([[word_to_id["the"]]])
    print("\nGenerative text decoding check:")
    print(f"  Seed: 'the'")
    generated = ["the"]
    
    with torch.no_grad():
        for _ in range(7):
            logits = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :]).item()
            generated.append(vocab[next_token])
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            
    print(f"  Predicted sequence: {' '.join(generated)}")

if __name__ == "__main__":
    main()

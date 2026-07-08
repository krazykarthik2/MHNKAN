import os
import sys
import math
import time
import random
import urllib.request
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

# ==============================================================================
# 2. Standalone EML-KAN Decoder-Only Transformer (LLM)
# ==============================================================================

class EMLKANAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
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
        
        self.ffn1 = EMLKANLinear(d_model, d_model * 2, num_components=4)
        self.ffn2 = EMLKANLinear(d_model * 2, d_model, num_components=4)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn2(self.ffn1(self.ln2(x)))
        return x

class EMLKANTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 256, d_model))
        
        self.blocks = nn.ModuleList([EMLKANBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        bs, seq_len = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:, :seq_len, :]
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ==============================================================================
# 3. Magnitude Sparsification / Pruning
# ==============================================================================

def apply_sparsity_to_model(model, sparsity=0.5):
    print(f"\nApplying magnitude pruning to achieve {sparsity * 100.0}% sparsity...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "linear.weight" in name:
                threshold = torch.quantile(torch.abs(param), sparsity)
                mask = torch.abs(param) >= threshold
                param.mul_(mask.float())
                print(f"  Pruned {name} -> remaining active: {mask.sum().item()} / {mask.numel()}")

# ==============================================================================
# 4. Symbolic PyTorch DAG Code Generator
# ==============================================================================

def generate_optimized_dag_pytorch(model, filepath="most_optimized_llm_dag.py"):
    print(f"\nCompiling trained parameters into optimized PyTorch DAG equations inside {filepath}...")
    
    with open(filepath, "w") as f:
        f.write("import torch\n")
        f.write("import torch.nn as nn\n\n")
        f.write("class OptimizedEMLKANFFN1(nn.Module):\n")
        f.write("    def __init__(self):\n")
        f.write("        super().__init__()\n\n")
        f.write("    def forward(self, features):\n")
        f.write("        # features shape: [batch_size, seq_len, d_model]\n")
        f.write("        bs, seq_len, d_model = features.shape\n")
        f.write("        flat_features = features.view(-1, d_model)\n")
        f.write("        flat_out = torch.zeros(flat_features.shape[0], 256, device=features.device)\n\n") # d_ffn = d_model * 2 = 256
        
        for b_idx, block in enumerate(model.blocks):
            fc = block.ffn1.linear.weight.data.cpu().numpy()
            act = block.ffn1.act
            
            out_features, in_features = fc.shape
            
            f.write(f"        # --- Block {b_idx} FFN1 Symbolic Mapping ---\n")
            for c in range(out_features):
                f.write(f"        # Node {c}\n")
                f.write(f"        z_{c} = ")
                
                active_indices = np.where(np.abs(fc[c]) > 1e-4)[0]
                if len(active_indices) == 0:
                    f.write("torch.zeros(flat_features.shape[0], device=features.device)\n")
                else:
                    terms = []
                    for idx in active_indices:
                        terms.append(f"flat_features[:, {idx}] * {fc[c, idx]:.6f}")
                    f.write(" + ".join(terms) + "\n")
                    
                w_base = act.weight_base[c].item()
                f.write(f"        out_{c} = z_{c} * {w_base:.6f}\n")
                
                for k in range(4):
                    a = act.a[c, k].item()
                    b = act.b[c, k].item()
                    c_val = act.c[c, k].item()
                    d = act.d[c, k].item()
                    w_eml = act.weight_eml[c, k].item()
                    
                    if abs(w_eml) < 1e-4:
                        continue
                        
                    f.write(f"        # Basis {k}\n")
                    f.write(f"        arg_x_{c}_{k} = torch.clamp(z_{c} * {a:.6f} + {b:.6f}, min=-10.0, max=10.0)\n")
                    f.write(f"        val_{c}_{k} = z_{c} * {c_val:.6f} + {d:.6f}\n")
                    f.write(f"        arg_y_{c}_{k} = torch.where(val_{c}_{k} > 20.0, val_{c}_{k}, torch.where(val_{c}_{k} < -20.0, torch.zeros_like(val_{c}_{k}), torch.log(1.0 + torch.exp(val_{c}_{k})))) + 1e-6\n")
                    f.write(f"        out_{c} = out_{c} + {w_eml:.6f} * (torch.exp(arg_x_{c}_{k}) - torch.log(arg_y_{c}_{k}))\n")
                    
                f.write(f"        flat_out[:, {c}] = out_{c}\n\n")
                
        f.write("        return flat_out.view(bs, seq_len, -1)\n")
        
    print("PyTorch DAG script compiled successfully.")

# ==============================================================================
# 5. Dataset Loader & Sandbox Training
# ==============================================================================

def main():
    print("Downloading Tiny Shakespeare Corpus for training...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "shakespeare.txt"
    
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)
        print("Shakespeare corpus downloaded successfully.")
    else:
        print("Using cached Shakespeare corpus.")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Character-level vocab
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Dataset Characters: {len(text)} | Unique Vocabulary Size: {vocab_size}")
    
    # Tokenize full corpus
    data_tokens = [char_to_id[ch] for ch in text]
    
    # Build model (EML-KAN Transformer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")
    
    model = EMLKANTransformerLM(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=1).to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    seq_len = 64
    batch_size = 32
    steps = 150
    
    print("\nTraining EML-KAN Decoder LLM on Tiny Shakespeare...")
    print("=" * 60)
    
    for step in range(steps):
        # Extract random batch chunks
        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            start_idx = random.randint(0, len(data_tokens) - seq_len - 2)
            x_batch.append(data_tokens[start_idx : start_idx + seq_len])
            y_batch.append(data_tokens[start_idx + 1 : start_idx + seq_len + 1])
            
        x_tensor = torch.tensor(x_batch, dtype=torch.long).to(device)
        y_tensor = torch.tensor(y_batch, dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        logits = model(x_tensor)
        
        loss = criterion(logits.view(-1, vocab_size), y_tensor.view(-1))
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 30 == 0 or step == 0:
            print(f"Step {step+1:03d}/{steps} | CrossEntropy Loss: {loss.item():.4f}")
            
    print("=" * 60)
    print("Training finished.")
    
    # Apply sparsity pruning
    apply_sparsity_to_model(model, sparsity=0.5)
    
    # Export optimized PyTorch DAG model
    base_dir = "large_scale_experiment" if os.path.exists("large_scale_experiment") else "."
    dag_path = os.path.join(base_dir, "most_optimized_llm_dag.py")
    generate_optimized_dag_pytorch(model, dag_path)
    
    # Text Generation check
    model.eval()
    seed_str = "ROMEO:"
    input_ids = torch.tensor([[char_to_id[ch] for ch in seed_str]], dtype=torch.long).to(device)
    
    print(f"\nGenerative text decoding check:")
    print(f"  Seed: '{seed_str}'")
    
    generated_chars = list(seed_str)
    with torch.no_grad():
        for _ in range(30):
            logits = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :]).item()
            generated_chars.append(id_to_char[next_token])
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
            
    print(f"  Predicted sequence:\n{''.join(generated_chars)}")

if __name__ == "__main__":
    main()

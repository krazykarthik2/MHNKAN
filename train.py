import os
import sys
import math
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==============================================================================
# 1. Standalone KAN-LLaMA Architecture Blocks
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

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqss = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqss, freqss), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x):
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len):
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1)
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot

class EMLKANGQAAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, rope):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_queries_per_kv = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.rope = rope
        
        self.q_proj = EMLKANLinear(d_model, d_model, num_components=2)
        self.k_proj = EMLKANLinear(d_model, n_kv_heads * self.d_head, num_components=2)
        self.v_proj = EMLKANLinear(d_model, n_kv_heads * self.d_head, num_components=2)
        self.out_proj = EMLKANLinear(d_model, d_model, num_components=2)

    def forward(self, x, mask=None):
        bs, seq_len, d_model = x.shape
        q = self.q_proj(x).view(bs, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        
        q, k = self.rope(q, k, seq_len)
        
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -65500.0)
            
        attn = F.softmax(scores, dim=-1)
        context = (attn @ v).transpose(1, 2).contiguous().view(bs, seq_len, d_model)
        return self.out_proj(context)

class EMLKANLLaMABlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ffn, rope):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = EMLKANGQAAttention(d_model, n_heads, n_kv_heads, rope)
        self.ln2 = RMSNorm(d_model)
        
        self.ffn1 = EMLKANLinear(d_model, d_ffn, num_components=4)
        self.ffn2 = EMLKANLinear(d_ffn, d_model, num_components=4)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn2(self.ffn1(self.ln2(x)))
        return x

class EMLKANLLaMA(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(vocab_size, config['d_model'])
        self.rope = RotaryEmbedding(dim=config['d_model'] // config['n_heads'])
        self.blocks = nn.ModuleList([
            EMLKANLLaMABlock(
                d_model=config['d_model'], 
                n_heads=config['n_heads'], 
                n_kv_heads=config['n_kv_heads'], 
                d_ffn=config['d_ffn'], 
                rope=self.rope
            ) for _ in range(config['n_layers'])
        ])
        self.ln_f = RMSNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], vocab_size, bias=False)

    def forward(self, input_ids):
        bs, seq_len = input_ids.shape
        x = self.token_emb(input_ids)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ==============================================================================
# 2. Conversational Tokenization & Dataset Handling
# ==============================================================================

class ConversationalDataset(Dataset):
    def __init__(self, raw_conversations, tokenizer, max_length=256):
        self.examples = []
        
        user_tag = "<|user|>\n"
        assistant_tag = "<|assistant|>\n"
        end_tag = "<|end|>\n"
        
        for dialogue in raw_conversations:
            full_ids = []
            target_mask = []
            
            for i, utterance in enumerate(dialogue):
                prefix = user_tag if i % 2 == 0 else assistant_tag
                text = prefix + utterance + end_tag
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                full_ids.extend(tokens)
                if i % 2 == 1:
                    target_mask.extend(tokens)
                else:
                    target_mask.extend([-100] * len(tokens))
                    
            if len(full_ids) > 1:
                full_ids = full_ids[:max_length]
                target_mask = target_mask[:max_length]
                
                pad_len = max_length - len(full_ids)
                if pad_len > 0:
                    input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
                    targets = target_mask + [-100] * pad_len
                else:
                    input_ids = full_ids
                    targets = target_mask
                    
                self.examples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "targets": torch.tensor(targets, dtype=torch.long)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# ==============================================================================
# 3. Model Configuration Scales
# ==============================================================================

CONFIGS = {
    "llama-7b-equivalent-kan": {
        "d_model": 2048,
        "n_heads": 16,
        "n_kv_heads": 4,
        "d_ffn": 5120,
        "n_layers": 24
    },
    "poc-llama-kan": {
        "d_model": 256,
        "n_heads": 8,
        "n_kv_heads": 2,
        "d_ffn": 512,
        "n_layers": 4
    }
}

# ==============================================================================
# 4. Learning Rate Schedule with Warmup
# ==============================================================================

def get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==============================================================================
# 5. Checkpoint Manager
# ==============================================================================

def save_checkpoint(state, checkpoint_dir="checkpoints", filename="checkpoint_latest.pt"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    print(f"\n[CHECKPOINTING] Saving model weights to {filepath}...")
    torch.save(state, filepath)

def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_dir="checkpoints", filename="checkpoint_latest.pt", device="cpu"):
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(filepath):
        print(f"\n[CHECKPOINTING] Loading weights from {filepath}...")
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint.get('epoch', 0), checkpoint.get('loss', 999.0)
    print("\n[CHECKPOINTING] No checkpoint found. Initializing model randomly.")
    return 0, 999.0

# ==============================================================================
# 6. Interactive Conversational CLI Chat Loop
# ==============================================================================

def run_conversational_chat(model, tokenizer, device, max_gen_len=64):
    model.eval()
    print("\n" + "=" * 60)
    print("      Interactive EML-KAN LLaMA Assistant CLI Mode")
    print("  Type 'exit' or 'quit' to end the dialogue conversation.")
    print("=" * 60 + "\n")
    
    dialogue_history = []
    
    while True:
        try:
            user_input = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        dialogue_history.append(user_input)
        
        full_context = ""
        for i, utterance in enumerate(dialogue_history):
            prefix = "<|user|>\n" if i % 2 == 0 else "<|assistant|>\n"
            full_context += prefix + utterance + "<|end|>\n"
        full_context += "<|assistant|>\n"
        
        input_ids = torch.tensor([tokenizer.encode(full_context, add_special_tokens=False)], dtype=torch.long).to(device)
        
        response_tokens = []
        print("Assistant > ", end="", flush=True)
        
        with torch.no_grad():
            for _ in range(max_gen_len):
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(input_ids)
                next_logits = logits[0, -1, :] / 0.7
                
                val, idx = torch.topk(next_logits, 10)
                probs = F.softmax(val, dim=-1)
                next_token = idx[torch.multinomial(probs, 1)].item()
                
                if next_token == tokenizer.encode("<|end|>\n", add_special_tokens=False)[0] or next_token == tokenizer.eos_token_id:
                    break
                    
                response_tokens.append(next_token)
                token_str = tokenizer.decode([next_token])
                print(token_str, end="", flush=True)
                
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
                
        print()
        assistant_response = tokenizer.decode(response_tokens).strip()
        dialogue_history.append(assistant_response)

# ==============================================================================
# 7. Main Execution: Train or Chat Mode
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Professional KAN-LLaMA Dialogue Assistant")
    parser.add_argument("--profile", type=str, default="poc-llama-kan",
                        choices=["poc-llama-kan", "llama-7b-equivalent-kan"],
                        help="Configuration scaling profile (default: poc-llama-kan)")
    parser.add_argument("--chat", action="store_true",
                        help="Enter interactive conversation mode using saved checkpoints")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Base learning rate (default: 2e-4)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")
    
    print("\nLoading state-of-the-art Qwen-2.5 BPE Tokenizer...")
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    except Exception:
        # Open source non-gated fallback configuration
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Qwen-2.5 BPE Vocabulary Size: {vocab_size}")
    
    config = CONFIGS[args.profile]
    print(f"\nBuilding EML-KAN LLaMA model under {args.profile.upper()} profile...")
    model = EMLKANLLaMA(vocab_size=vocab_size, config=config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    if args.chat:
        start_epoch, _ = load_checkpoint(model, checkpoint_dir="checkpoints", device=device)
        if start_epoch == 0:
            print("[WARNING] Starting chat with an untrained model. Generative output may be garbage.")
        run_conversational_chat(model, tokenizer, device)
        return
        
    print("\nLoading DailyDialog Dialogue Dataset from Hugging Face...")
    try:
        from datasets import load_dataset
        raw_dataset = load_dataset("daily_dialog")
        train_conversations = raw_dataset["train"]["dialog"]
        val_conversations = raw_dataset["validation"]["dialog"]
    except Exception as data_err:
        print(f"Failed to load DailyDialog ({data_err}). Building mock conversational data stream...")
        train_conversations = [
            ["Hello, how are you?", "I am doing well, how can I assist you?", "Tell me a joke.", "Why did the neural net cross the road? To minimize loss!"],
            ["What is your name?", "I am the EML-KAN LLaMA conversational assistant.", "That is a cool name.", "Thank you, I was optimized for language generation."]
        ] * 100
        val_conversations = train_conversations[:10]
        
    train_dataset = ConversationalDataset(train_conversations, tokenizer, max_length=128)
    val_dataset = ConversationalDataset(val_conversations, tokenizer, max_length=128)
    
    is_large = (args.profile == "llama-7b-equivalent-kan")
    batch_size = 2 if is_large else 16
    accumulation_steps = 16 if is_large else 1
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)
    
    start_epoch, _ = load_checkpoint(model, optimizer, scheduler, device=device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    
    print(f"\nStarting conversational pre-training pipeline...")
    print(f"Batch size: {batch_size} (Effective batch: {batch_size * accumulation_steps}) | Max Epochs: {args.epochs}")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        step_idx = 0
        
        optimizer.zero_grad()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accumulation_steps
            
            if (step_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
            step_idx += 1
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                targets = batch["targets"].to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(input_ids)
                    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Live generation sampler to verify conversational capability progress
        print(f"--- Epoch {epoch+1:02d} Conversational Alignment Sample ---")
        sample_prompt = "<|user|>\nHello, what is your name?\n<|assistant|>\n"
        sample_input = torch.tensor([tokenizer.encode(sample_prompt, add_special_tokens=False)], dtype=torch.long).to(device)
        model.eval()
        with torch.no_grad():
            for _ in range(20):
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(sample_input)
                next_token = torch.argmax(logits[0, -1, :]).item()
                if next_token == tokenizer.eos_token_id or next_token == tokenizer.encode("<|end|>\n", add_special_tokens=False)[0]:
                    break
                sample_input = torch.cat([sample_input, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        decoded_sample = tokenizer.decode(sample_input[0].tolist())
        print(f"Generated text:\n{decoded_sample}")
        print("-" * 60)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': avg_val_loss
        })
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': best_loss
            }, filename="checkpoint_best.pt")
            
    print("\nTraining completed successfully! Use 'python train.py --chat' to enter interactive assistant mode.")

if __name__ == "__main__":
    main()

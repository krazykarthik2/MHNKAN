import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def compare_distributions():
    print("Loading BERT-small model...")
    model_name = "prajjwal1/bert-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name)
    bert.eval()
    
    # Real-world test sentences
    sentences = [
        "In machine learning, model compression is critical for edge device deployment.",
        "Kolmogorov-Arnold Networks replace linear weights with learnable activation functions.",
        "We are evaluating the distillation performance of BERT feed-forward layers.",
        "Deep learning models require optimization to run efficiently on low-power microcontrollers.",
        "Let's measure the cosine similarity of the distilled EML-KAN model replica."
    ]
    
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    for layer_idx in range(4):
        print(f"\n--- Layer {layer_idx} ---")
        
        # 1. Capture real FFN inputs
        ffn_inputs = []
        def hook_fn(module, input_tensor, output_tensor):
            ffn_inputs.append(input_tensor[0].detach())
            
        hook_handle = bert.encoder.layer[layer_idx].intermediate.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = bert(**inputs)
        hook_handle.remove()
        
        real_ffn_input = torch.cat(ffn_inputs, dim=0).view(-1, 512)
        
        # 2. Capture LayerNorm parameters
        ln_weight = bert.encoder.layer[layer_idx].attention.output.LayerNorm.weight.data
        ln_bias = bert.encoder.layer[layer_idx].attention.output.LayerNorm.bias.data
        
        # 3. Generate our matched inputs
        Z = torch.randn(2000, 512)
        synthetic_matched = Z * ln_weight + ln_bias
        
        # Print Stats
        print(f"Real hidden states  | Mean: {real_ffn_input.mean().item():.4f} | Std: {real_ffn_input.std().item():.4f} | Min: {real_ffn_input.min().item():.4f} | Max: {real_ffn_input.max().item():.4f}")
        print(f"Synthetic matched   | Mean: {synthetic_matched.mean().item():.4f} | Std: {synthetic_matched.std().item():.4f} | Min: {synthetic_matched.min().item():.4f} | Max: {synthetic_matched.max().item():.4f}")

if __name__ == "__main__":
    compare_distributions()

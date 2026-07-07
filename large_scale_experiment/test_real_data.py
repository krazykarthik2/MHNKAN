import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Define EML-KAN Layers to match distill_test.py
class EMLKANActivation(nn.Module):
    def __init__(self, channels, num_components=4):
        super().__init__()
        self.channels = channels
        self.num_components = num_components
        
        self.a = nn.Parameter(torch.randn(channels, num_components) * 0.01)
        self.b = nn.Parameter(torch.zeros(channels, num_components))
        self.c = nn.Parameter(torch.randn(channels, num_components) * 0.01)
        self.d = nn.Parameter(torch.zeros(channels, num_components))
        
        self.weight_base = nn.Parameter(torch.ones(channels) * 0.1)
        self.weight_eml = nn.Parameter(torch.randn(channels, num_components) * 0.01)

    def forward(self, x):
        out = self.weight_base * x
        # stable softplus
        for k in range(self.num_components):
            arg_x = torch.clamp(self.a[:, k] * x + self.b[:, k], min=-10.0, max=10.0)
            # stable softplus
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

class EMLKANFFNReplica(nn.Module):
    def __init__(self, d_model, num_components=4):
        super().__init__()
        self.layer1 = EMLKANLinear(d_model, d_model, num_components=num_components)
        self.ln = nn.LayerNorm(d_model)
        self.layer2 = EMLKANLinear(d_model, d_model, num_components=num_components)
        
    def forward(self, x):
        return self.layer2(self.ln(self.layer1(x)))

def test_real_data():
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        print(f"Error: {temp_dir} directory not found.")
        return
        
    print("Loading BERT-small model and tokenizer...")
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
    
    # Tokenize input text
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    print("\nStarting Evaluation on Real Text Data:")
    print("=" * 80)
    
    # Check for layer files inside the temp folder
    for file_name in sorted(os.listdir(temp_dir)):
        if not file_name.endswith(".pth"):
            continue
            
        try:
            layer_idx = int(file_name.split("_")[1].split(".")[0])
        except Exception:
            continue
            
        weights_path = os.path.join(temp_dir, file_name)
        print(f"\nEvaluating Layer {layer_idx} using weights: {weights_path}")
        
        # Capture real intermediate FFN inputs using PyTorch forward hooks
        ffn_inputs = []
        def hook_fn(module, input_tensor, output_tensor):
            # Input to the intermediate dense layer
            ffn_inputs.append(input_tensor[0].detach())
            
        hook_handle = bert.encoder.layer[layer_idx].intermediate.register_forward_hook(hook_fn)
        
        # Run BERT forward pass to populate hooks with real features
        with torch.no_grad():
            _ = bert(**inputs)
            
        hook_handle.remove()
        
        # Flatten sequence length and batch dimensions to run evaluations per token
        real_ffn_input = torch.cat(ffn_inputs, dim=0).view(-1, 512)
        
        # Get targets from original FFN layer
        ffn_intermediate = bert.encoder.layer[layer_idx].intermediate
        ffn_output = bert.encoder.layer[layer_idx].output.dense
        with torch.no_grad():
            real_ffn_output = ffn_output(ffn_intermediate(real_ffn_input))
            
        # Load EML-KAN Replica model and apply trained weights
        kan_replica = EMLKANFFNReplica(d_model=512, num_components=4)
        kan_replica.load_state_dict(torch.load(weights_path, map_location="cpu"))
        kan_replica.eval()
        
        # Evaluate replica outputs
        with torch.no_grad():
            predicted_output = kan_replica(real_ffn_input)
            
        # Compute metrics
        mse = F.mse_loss(predicted_output, real_ffn_output).item()
        cos_sims = F.cosine_similarity(predicted_output, real_ffn_output, dim=-1)
        mean_cos_sim = cos_sims.mean().item()
        min_cos_sim = cos_sims.min().item()
        
        print(f"  Real Samples Evaluated (Tokens): {real_ffn_input.shape[0]}")
        print(f"  Mean Squared Error (MSE):       {mse:.6f}")
        print(f"  Mean Cosine Similarity:         {mean_cos_sim * 100.0:.2f}%")
        print(f"  Min Cosine Similarity:          {min_cos_sim * 100.0:.2f}%")
        
    print("=" * 80)
    
    # Delete the temp folder as requested
    print(f"\nDeleting the {temp_dir} folder...")
    try:
        shutil.rmtree(temp_dir)
        print("Folder deleted successfully.")
    except Exception as e:
        print(f"Error deleting folder: {e}")

if __name__ == "__main__":
    test_real_data()

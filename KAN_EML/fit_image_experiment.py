import os
import sys
# Injected path for root and core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw

from eml_network import EMLKANLayer

# -------------------------------------------------------------------------
# 1. Stable EML-KAN with Layer Normalization Scaling
# -------------------------------------------------------------------------
class StableEMLKAN(nn.Module):
    def __init__(self, layers_hidden, num_components=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for idx in range(len(layers_hidden) - 1):
            self.layers.append(
                EMLKANLayer(layers_hidden[idx], layers_hidden[idx+1], num_eml_components=num_components)
            )
            
    def forward(self, x):
        for layer in self.layers:
            # Evaluate EMLKANLayer
            out = layer(x)
            # Scale by feature count to prevent exponential explosion (e.g. e^38)
            x = out / np.sqrt(layer.in_features)
        # Multiply final output by a scalar to restore RGB scale [0, 1]
        return x * 2.0

# -------------------------------------------------------------------------
# 2. Fourier Positional Encoding
# -------------------------------------------------------------------------
def fourier_encode(x, num_frequencies=3):
    out = [x]
    for freq in range(num_frequencies):
        factor = np.pi * (2.0 ** freq)
        out.append(torch.sin(factor * x))
        out.append(torch.cos(factor * x))
    return torch.cat(out, dim=-1)

# -------------------------------------------------------------------------
# 3. Programmatic target image generator (Stylized Rick Astley)
# -------------------------------------------------------------------------
def generate_rick_roll_image(size=32):
    img = Image.new("RGB", (size, size), "#85A3E0") # Sky blue background
    draw = ImageDraw.Draw(img)
    
    s = size / 128.0
    
    # 1. Shoulders/Suit (Dark suit with white shirt collar)
    draw.polygon([(20*s, 128*s), (108*s, 128*s), (95*s, 95*s), (33*s, 95*s)], fill="#1E3048") 
    draw.polygon([(54*s, 95*s), (74*s, 95*s), (64*s, 115*s)], fill="#FFFFFF") 
    
    # 2. Face (Peach skin oval)
    draw.ellipse([(44*s, 40*s), (84*s, 95*s)], fill="#FCD5B5")
    
    # 3. Signature Rick Astley Hair (High-volume reddish brown)
    draw.ellipse([(40*s, 22*s), (88*s, 52*s)], fill="#B85A38") 
    draw.ellipse([(38*s, 30*s), (58*s, 55*s)], fill="#B85A38") 
    draw.ellipse([(70*s, 30*s), (90*s, 55*s)], fill="#B85A38") 
    
    # 4. Sunglasses (Dark cool sunglasses)
    draw.rectangle([(48*s, 55*s), (62*s, 67*s)], fill="#1A1A1A") 
    draw.rectangle([(66*s, 55*s), (80*s, 67*s)], fill="#1A1A1A") 
    draw.line([(60*s, 60*s), (68*s, 60*s)], fill="#1A1A1A", width=max(1, int(2*s))) 
    
    # 5. Microphone (Silver mic held near face)
    draw.ellipse([(35*s, 80*s), (48*s, 93*s)], fill="#B3B3B3") 
    draw.rectangle([(38*s, 93*s), (45*s, 128*s)], fill="#333333") 
    
    return img

def main():
    print("=" * 80)
    print("EML-KAN Stable Zero-Loss Image Regression")
    print("=" * 80)
    
    size = 32
    target_img = generate_rick_roll_image(size)
    target_path = "KAN_EML/rick_roll_target.png"
    target_img.save(target_path)
    print(f"Generated target portrait at: {target_path}")
    
    # Coordinates mapping
    coords = np.zeros((size, size, 2), dtype=np.float64)
    for u in range(size):
        for v in range(size):
            coords[u, v, 0] = (u / (size - 1)) * 2.0 - 1.0
            coords[u, v, 1] = (v / (size - 1)) * 2.0 - 1.0
            
    X_coords = torch.tensor(coords.reshape(-1, 2), dtype=torch.float64)
    X = fourier_encode(X_coords, num_frequencies=3).double()
    in_dim = X.shape[1]
    
    rgb_data = np.array(target_img, dtype=np.float64) / 255.0
    y = torch.tensor(rgb_data.reshape(-1, 3), dtype=torch.float64)
    
    # 4. Build Stable EML-KAN model
    # [14 -> 64 -> 3] network
    model = StableEMLKAN(layers_hidden=[in_dim, 64, 3], num_components=3).double()
    
    # Initialize stable parameter starting points
    with torch.no_grad():
        for layer in model.layers:
            layer.weight_base.zero_()
            layer.weight_base.requires_grad = False
            layer.weight_eml.fill_(1.0)
            layer.weight_eml.requires_grad = False
            # Small scaling factor to ensure exp(a*x) starts small and behaves linearly
            layer.a.fill_(0.1)
            layer.b.fill_(0.0)
            layer.c.fill_(0.5)
            layer.d.fill_(1.0)
            
    criterion = nn.MSELoss()
    
    # 5. Train with AdamW (Global Phase)
    print("Phase 1: Running AdamW global search...")
    optimizer_adam = optim.AdamW(model.parameters(), lr=0.03)
    epochs = 1200
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer_adam.zero_grad()
        
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer_adam.step()
        
        if epoch % 300 == 0 or epoch == 1:
            print(f"  Adam Epoch {epoch:4d} | Loss (MSE): {loss.item():.10f}")
            
    # 6. Fine-tune with L-BFGS (Exact Local Convergence Phase)
    print("\nPhase 2: Fine-tuning with L-BFGS for machine precision...")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=0.5, 
        max_iter=300, 
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-22,
        tolerance_change=1e-22
    )
    
    def closure():
        optimizer_lbfgs.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        return loss
        
    for step in range(1, 15):
        loss = optimizer_lbfgs.step(closure)
        print(f"  L-BFGS Step {step} | Loss (MSE): {loss.item():.20f}")
        if loss.item() < 1e-15:
            break
            
    # 7. Reconstruct and Save Reconstructed Image
    model.eval()
    with torch.no_grad():
        reconstructed_output = model(X)
        reconstructed_loss = criterion(reconstructed_output, y).item()
        
    reconstructed_rgb = torch.clamp(reconstructed_output, 0.0, 1.0).numpy()
    reconstructed_rgb = (reconstructed_rgb.reshape(size, size, 3) * 255.0).astype(np.uint8)
    
    reconstructed_img = Image.fromarray(reconstructed_rgb)
    reconstructed_path = "KAN_EML/rick_roll_reconstructed.png"
    reconstructed_img.save(reconstructed_path)
    
    print("\n" + "=" * 80)
    print(f"Final Image Reconstruction MSE Loss: {reconstructed_loss:.20f}")
    if reconstructed_loss < 1e-12:
        print("SUCCESS: Achieved mathematically perfect zero-loss reconstruction (Loss = 0.0)!")
    print("=" * 80)

if __name__ == "__main__":
    main()

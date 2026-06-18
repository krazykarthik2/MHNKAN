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
# 1. Residual EML-KAN with Skip Connections
# -------------------------------------------------------------------------
class ResidualEMLKAN(nn.Module):
    def __init__(self, in_features=2, hidden_dim=16, out_features=3, num_layers=3, num_components=3):
        super().__init__()
        self.input_layer = EMLKANLayer(in_features, hidden_dim, num_eml_components=num_components)
        self.hidden_layers = nn.ModuleList([
            EMLKANLayer(hidden_dim, hidden_dim, num_eml_components=num_components)
            for _ in range(num_layers - 2)
        ])
        self.output_layer = EMLKANLayer(hidden_dim, out_features, num_eml_components=num_components)
        
    def forward(self, x):
        # Input coordinate mapping
        h = self.input_layer(x)
        # Hidden layers with skip connections (Residual)
        for layer in self.hidden_layers:
            h = layer(h) + h
        # Output color mapping
        return self.output_layer(h)

# -------------------------------------------------------------------------
# 2. Programmatic target image generator (Stylized Rick Astley)
# -------------------------------------------------------------------------
def generate_rick_roll_image(size=64):
    img = Image.new("RGB", (size, size), "#85A3E0") # Sky blue background
    draw = ImageDraw.Draw(img)
    
    # Scale coordinates to fit image size
    s = size / 128.0
    
    # 1. Shoulders/Suit (Dark suit with white shirt collar)
    draw.polygon([(20*s, 128*s), (108*s, 128*s), (95*s, 95*s), (33*s, 95*s)], fill="#1E3048") # Dark Blue Suit
    draw.polygon([(54*s, 95*s), (74*s, 95*s), (64*s, 115*s)], fill="#FFFFFF") # White shirt collar
    
    # 2. Face (Peach skin oval)
    draw.ellipse([(44*s, 40*s), (84*s, 95*s)], fill="#FCD5B5")
    
    # 3. Signature Rick Astley Hair (High-volume reddish brown)
    draw.ellipse([(40*s, 22*s), (88*s, 52*s)], fill="#B85A38") # Top volume
    draw.ellipse([(38*s, 30*s), (58*s, 55*s)], fill="#B85A38") # Left volume
    draw.ellipse([(70*s, 30*s), (90*s, 55*s)], fill="#B85A38") # Right volume
    
    # 4. Sunglasses (Dark cool sunglasses)
    draw.rectangle([(48*s, 55*s), (62*s, 67*s)], fill="#1A1A1A") # Left lens
    draw.rectangle([(66*s, 55*s), (80*s, 67*s)], fill="#1A1A1A") # Right lens
    draw.line([(60*s, 60*s), (68*s, 60*s)], fill="#1A1A1A", width=int(2*s)) # Bridge
    
    # 5. Microphone (Silver mic held near face)
    draw.ellipse([(35*s, 80*s), (48*s, 93*s)], fill="#B3B3B3") # Mesh head
    draw.rectangle([(38*s, 93*s), (45*s, 128*s)], fill="#333333") # Stand / shaft
    
    return img

def main():
    print("=" * 80)
    print("EML-KAN Image Regression: Modeling Rick Astley Portrait")
    print("=" * 80)
    
    size = 64
    # Generate and save target image
    target_img = generate_rick_roll_image(size)
    target_path = "KAN_EML/rick_roll_target.png"
    target_img.save(target_path)
    print(f"Generated target portrait at: {target_path}")
    
    # 3. Prepare data coordinates
    # Normalized coordinates x, y in [-1.0, 1.0]
    coords = np.zeros((size, size, 2), dtype=np.float32)
    for u in range(size):
        for v in range(size):
            coords[u, v, 0] = (u / (size - 1)) * 2.0 - 1.0
            coords[u, v, 1] = (v / (size - 1)) * 2.0 - 1.0
            
    # Reshape coordinates to [N, 2]
    X = torch.tensor(coords.reshape(-1, 2), dtype=torch.float32)
    
    # Get target RGB values normalized to [0, 1]
    rgb_data = np.array(target_img, dtype=np.float32) / 255.0
    y = torch.tensor(rgb_data.reshape(-1, 3), dtype=torch.float32)
    
    # 4. Build EML-KAN model with Skip Connections
    # [2 -> 24 -> 24 -> 3] network
    model = ResidualEMLKAN(in_features=2, hidden_dim=24, out_features=3, num_layers=3, num_components=3)
    
    # 5. Layer-wise Learning Rates
    # Set smaller learning rates for inner scale params (a, c) to avoid log domain failures
    inner_params = []
    outer_params = []
    
    for name, param in model.named_parameters():
        if ".a" in name or ".c" in name:
            inner_params.append(param)
        else:
            outer_params.append(param)
            
    optimizer_adam = optim.AdamW([
        {'params': outer_params, 'lr': 0.02, 'weight_decay': 1e-5},
        {'params': inner_params, 'lr': 0.005, 'weight_decay': 0.0}
    ])
    
    criterion = nn.MSELoss()
    
    # 6. Train with AdamW (Global Phase)
    print("Phase 1: Running AdamW global coordinate fitting...")
    epochs = 400
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer_adam.zero_grad()
        
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer_adam.step()
        
        if epoch % 100 == 0 or epoch == 1:
            print(f"  Adam Epoch {epoch:3d} | Loss (MSE): {loss.item():.8f}")
            
    # 7. Fine-tune with L-BFGS (Exact Local Convergence Phase)
    print("\nPhase 2: Fine-tuning with L-BFGS to capture fine details...")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=0.5, 
        max_iter=100, 
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-12
    )
    
    def closure():
        optimizer_lbfgs.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        return loss
        
    for step in range(1, 4):
        loss = optimizer_lbfgs.step(closure)
        print(f"  L-BFGS Step {step} | Loss (MSE): {loss.item():.12f}")
        
    # 8. Reconstruct and Save Reconstructed Image
    model.eval()
    with torch.no_grad():
        reconstructed_output = model(X)
        reconstructed_loss = criterion(reconstructed_output, y).item()
        
    # Clip values to [0, 1] and map to [0, 255]
    reconstructed_rgb = torch.clamp(reconstructed_output, 0.0, 1.0).numpy()
    reconstructed_rgb = (reconstructed_rgb.reshape(size, size, 3) * 255.0).astype(np.uint8)
    
    reconstructed_img = Image.fromarray(reconstructed_rgb)
    reconstructed_path = "KAN_EML/rick_roll_reconstructed.png"
    reconstructed_img.save(reconstructed_path)
    
    print("\n" + "=" * 80)
    print(f"Final Image Reconstruction MSE Loss: {reconstructed_loss:.12f}")
    print(f"Saved reconstructed image to: {reconstructed_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

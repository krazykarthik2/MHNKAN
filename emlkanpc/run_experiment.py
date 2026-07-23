import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np
from eml_flow_matching import EMLFlowMatchingModel

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

def get_mnist_subset(num_samples_per_digit=60):
    """
    Loads MNIST digits using torchvision.
    Uses tensor slicing to load instantly without loops.
    """
    try:
        import torchvision
        import torchvision.transforms as transforms
        
        print("Attempting to load MNIST dataset via torchvision...", flush=True)
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        
        images_list = []
        labels_list = []
        
        for digit in range(10):
            indices = (dataset.targets == digit).nonzero(as_tuple=True)[0][:num_samples_per_digit]
            digit_imgs = dataset.data[indices].float() / 255.0
            images_list.append(digit_imgs.view(-1, 784))
            labels_list.append(dataset.targets[indices])
            
        images = torch.cat(images_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        print(f"Loaded {images.shape[0]} real MNIST images successfully.", flush=True)
        return images, labels
        
    except Exception as e:
        print(f"Torchvision MNIST load failed: {e}", flush=True)
        print("Falling back to generating high-quality synthetic digit patterns...", flush=True)
        return generate_synthetic_digits(num_samples_per_digit)

def generate_synthetic_digits(num_samples_per_digit=60):
    """Generates synthetic 28x28 digit patterns representing classes 0-9."""
    images = []
    labels = []
    
    for digit in range(10):
        for _ in range(num_samples_per_digit):
            img = torch.zeros(1, 28, 28)
            if digit == 0:
                img[0, 5:23, 5] = 1.0
                img[0, 5:23, 22] = 1.0
                img[0, 5, 5:23] = 1.0
                img[0, 22, 5:23] = 1.0
            elif digit == 1:
                img[0, 5:23, 14] = 1.0
            elif digit == 2:
                img[0, 5, 5:23] = 1.0
                img[0, 5:14, 22] = 1.0
                img[0, 14, 5:23] = 1.0
                img[0, 14:23, 5] = 1.0
                img[0, 22, 5:23] = 1.0
            elif digit == 3:
                img[0, 5, 5:23] = 1.0
                img[0, 5:23, 22] = 1.0
                img[0, 14, 5:23] = 1.0
                img[0, 22, 5:23] = 1.0
            elif digit == 4:
                img[0, 5:14, 5] = 1.0
                img[0, 14, 5:23] = 1.0
                img[0, 5:23, 22] = 1.0
            elif digit == 5:
                img[0, 5, 5:23] = 1.0
                img[0, 5:14, 5] = 1.0
                img[0, 14, 5:23] = 1.0
                img[0, 14:23, 22] = 1.0
                img[0, 22, 5:23] = 1.0
            elif digit == 6:
                img[0, 5:23, 5] = 1.0
                img[0, 14, 5:23] = 1.0
                img[0, 14:23, 22] = 1.0
                img[0, 22, 5:23] = 1.0
            elif digit == 7:
                img[0, 5, 5:23] = 1.0
                img[0, 5:23, 22] = 1.0
            elif digit == 8:
                img[0, 5:23, 5] = 1.0
                img[0, 5:23, 22] = 1.0
                img[0, 5, 5:23] = 1.0
                img[0, 14, 5:23] = 1.0
                img[0, 22, 5:23] = 1.0
            else: # 9
                img[0, 5:14, 5] = 1.0
                img[0, 5:23, 22] = 1.0
                img[0, 5, 5:23] = 1.0
                img[0, 14, 5:23] = 1.0
                
            img += torch.randn(28, 28) * 0.05
            img = torch.clamp(img, 0.0, 1.0)
            images.append(img.view(-1))
            labels.append(digit)
            
    return torch.stack(images), torch.tensor(labels)

def print_ascii_digit(image_flat, threshold=0.35):
    grid = image_flat.detach().view(28, 28).numpy()
    for r in range(28):
        row_chars = []
        for c in range(28):
            val = grid[r, c]
            if val > 0.7:
                row_chars.append("#")
            elif val > threshold:
                row_chars.append("+")
            else:
                row_chars.append(" ")
        print("".join(row_chars))

def main():
    print("=" * 70, flush=True)
    print("   EML Optimal Transport Flow Matching (OT-FM) Demonstration", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load MNIST subset
    images, labels = get_mnist_subset(num_samples_per_digit=60)
    num_samples = images.shape[0]
    one_hot_labels = F.one_hot(labels, 10).float()
    
    # 2. Define Model
    model = EMLFlowMatchingModel(num_components=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    print(f"Flow Matching Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("-" * 70, flush=True)
    
    # 3. Train OT Flow Matching
    epochs = 100
    batch_size = 128
    noise_std = 0.05
    
    print(f"Training EML Flow Matching on CPU for {epochs} epochs...", flush=True)
    model.train()
    
    for epoch in range(epochs):
        indices = torch.randperm(num_samples)
        epoch_loss = 0.0
        batches = 0
        
        for start_idx in range(0, num_samples, batch_size):
            optimizer.zero_grad()
            
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            x_1 = images[batch_indices]
            y_one_hot = one_hot_labels[batch_indices]
            sz = x_1.shape[0]
            
            # Sample scaled noise x_0
            x_0 = torch.randn_like(x_1) * noise_std
            
            # Sample t in [0, 1]
            t = torch.rand(sz, 1)
            
            # Linear trajectory (OT path)
            x_t = (1.0 - t) * x_0 + t * x_1
            
            # Predict vector field
            v_pred = model(x_t, t, y_one_hot)
            
            # Target is u_t = x_1 - x_0
            v_target = x_1 - x_0
            
            # Loss
            loss = F.mse_loss(v_pred, v_target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} Completed... Avg MSE Loss: {epoch_loss/batches:.6f}", flush=True)
            
    print("-" * 70, flush=True)
    
    # 4. Generate and display digits conditionally
    print("Generating digits via 15-step Euler Flow integration...", flush=True)
    for digit in [0, 1, 2, 3, 4, 7]:
        print(f"\nGenerated Class Digit: {digit}", flush=True)
        print("-" * 35, flush=True)
        generated_flat = model.sample(digit, steps=15, noise_std=noise_std)
        print_ascii_digit(generated_flat, threshold=0.15)
        print("-" * 35, flush=True)
        
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()

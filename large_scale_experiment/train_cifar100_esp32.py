import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 1. Define Muon Optimizer helper

def newton_schulz5(G, steps=3):
    # Newton-Schulz iteration for orthogonalization (steps=3 is fast and accurate)
    a, b = G.shape
    X = G / (G.norm() + 1e-7)
    if a > b:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = 1.5 * X - 0.5 * B
    if a > b:
        X = X.T
    return X

class Muon(optim.Optimizer):
    """
    Muon optimizer: Multi-variate orthogonalization optimizer.
    Applies standard momentum followed by Newton-Schulz orthogonalization on 2D parameters.
    """
    def __init__(self, params, lr=0.02, momentum=0.9, ns_steps=3):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                
                # Reshape N-D tensor to 2D matrix
                shape = p.shape
                flat_p = p.view(shape[0], -1)
                flat_buf = buf.view(shape[0], -1)
                
                # Apply Newton-Schulz orthogonal step
                u = newton_schulz5(flat_buf, steps=ns_steps)
                
                # Update weights
                p.add_(u.view(shape), alpha=-lr)
                
        return loss

# 2. Define EML-KAN Layers

class EMLKANActivation(nn.Module):
    def __init__(self, channels, num_components=2):
        super().__init__()
        self.channels = channels
        self.num_components = num_components
        
        # EML activation parameters for each channel
        self.a = nn.Parameter(torch.randn(channels, num_components) * 0.01)
        self.b = nn.Parameter(torch.zeros(channels, num_components))
        self.c = nn.Parameter(torch.randn(channels, num_components) * 0.01)
        self.d = nn.Parameter(torch.zeros(channels, num_components))
        
        self.weight_base = nn.Parameter(torch.ones(channels) * 0.1)
        self.weight_eml = nn.Parameter(torch.randn(channels, num_components) * 0.01)

    def forward(self, x):
        is_4d = len(x.shape) == 4
        if is_4d:
            x = x.permute(0, 2, 3, 1) # [B, H, W, C]
            
        out = self.weight_base * x
        for k in range(self.num_components):
            arg_x = torch.clamp(self.a[:, k] * x + self.b[:, k], min=-10.0, max=10.0)
            arg_y = F.softplus(self.c[:, k] * x + self.d[:, k]) + 1e-6
            out = out + self.weight_eml[:, k] * (torch.exp(arg_x) - torch.log(arg_y))
            
        if is_4d:
            out = out.permute(0, 3, 1, 2) # [B, C, H, W]
        return out

class EMLKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_components=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = EMLKANActivation(out_channels, num_components)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class EMLKANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_components=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = EMLKANActivation(out_features, num_components)
        
    def forward(self, x):
        return self.act(self.linear(x))

# 3. Complete EML-KAN CIFAR-100 Classifier Architecture

class EMLKANCifar100(nn.Module):
    def __init__(self, num_components=2):
        super().__init__()
        self.features = nn.Sequential(
            EMLKANConv2d(3, 32, kernel_size=3, padding=1, num_components=num_components),
            nn.MaxPool2d(2, 2), # 16x16
            EMLKANConv2d(32, 64, kernel_size=3, padding=1, num_components=num_components),
            nn.MaxPool2d(2, 2), # 8x8
            nn.AdaptiveAvgPool2d((1, 1)) # 1x1x64
        )
        self.classifier = EMLKANLinear(64, 100, num_components=num_components)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 4. Export to ESP32 C++ Code Generator

def generate_esp32_header(model, filepath="esp32_cifar100_inference.h"):
    print(f"Generating ESP32 inference header at {filepath}...")
    
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    conv1 = model.features[0].conv.weight.data.numpy()
    bn1 = model.features[0].bn
    mean1 = bn1.running_mean.data.numpy()
    var1 = bn1.running_var.data.numpy()
    weight1 = bn1.weight.data.numpy()
    bias1 = bn1.bias.data.numpy()
    scale1 = weight1 / np.sqrt(var1 + bn1.eps)
    offset1 = bias1 - mean1 * scale1
    act1 = model.features[0].act
    
    conv2 = model.features[2].conv.weight.data.numpy()
    bn2 = model.features[2].bn
    mean2 = bn2.running_mean.data.numpy()
    var2 = bn2.running_var.data.numpy()
    weight2 = bn2.weight.data.numpy()
    bias2 = bn2.bias.data.numpy()
    scale2 = weight2 / np.sqrt(var2 + bn2.eps)
    offset2 = bias2 - mean2 * scale2
    act2 = model.features[2].act
    
    fc = model.classifier.linear.weight.data.numpy()
    act3 = model.classifier.act
    
    with open(filepath, "w") as f:
        f.write("/* Automatically generated EML-KAN C++ ESP32 inference header */\n")
        f.write("#ifndef EML_KAN_CIFAR100_H\n")
        f.write("#define EML_KAN_CIFAR100_H\n\n")
        f.write("#include <math.h>\n\n")
        
        f.write("inline float softplus(float x) {\n")
        f.write("    return logf(1.0f + expf(x));\n")
        f.write("}\n\n")
        
        def write_array(name, arr):
            f.write(f"const float {name}[] PROGMEM = {{\n    ")
            flat = arr.flatten()
            for idx, val in enumerate(flat):
                f.write(f"{val:.6f}f")
                if idx < len(flat) - 1:
                    f.write(", ")
                if (idx + 1) % 8 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")
            
        write_array("CONV1_WEIGHTS", conv1)
        write_array("BN1_SCALE", scale1)
        write_array("BN1_OFFSET", offset1)
        write_array("ACT1_A", act1.a.data.numpy())
        write_array("ACT1_B", act1.b.data.numpy())
        write_array("ACT1_C", act1.c.data.numpy())
        write_array("ACT1_D", act1.d.data.numpy())
        write_array("ACT1_W_BASE", act1.weight_base.data.numpy())
        write_array("ACT1_W_EML", act1.weight_eml.data.numpy())
        
        write_array("CONV2_WEIGHTS", conv2)
        write_array("BN2_SCALE", scale2)
        write_array("BN2_OFFSET", offset2)
        write_array("ACT2_A", act2.a.data.numpy())
        write_array("ACT2_B", act2.b.data.numpy())
        write_array("ACT2_C", act2.c.data.numpy())
        write_array("ACT2_D", act2.d.data.numpy())
        write_array("ACT2_W_BASE", act2.weight_base.data.numpy())
        write_array("ACT2_W_EML", act2.weight_eml.data.numpy())
        
        write_array("FC_WEIGHTS", fc)
        write_array("ACT3_A", act3.a.data.numpy())
        write_array("ACT3_B", act3.b.data.numpy())
        write_array("ACT3_C", act3.c.data.numpy())
        write_array("ACT3_D", act3.d.data.numpy())
        write_array("ACT3_W_BASE", act3.weight_base.data.numpy())
        write_array("ACT3_W_EML", act3.weight_eml.data.numpy())
        
        f.write("#endif // EML_KAN_CIFAR100_H\n")
    print("ESP32 header generated successfully.")

# 5. Training Pipeline Loop

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    print("Loading CIFAR-100 dataset...")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    model = EMLKANCifar100().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Split parameters: 2D (Muon) and 1D (AdamW)
    params_2d = []
    params_1d = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim >= 2:
                params_2d.append(p)
            else:
                params_1d.append(p)
                
    # Optimize using Muon for structural matrices and AdamW for 1D weights
    opt_muon = Muon(params_2d, lr=0.03, momentum=0.9, ns_steps=3)
    opt_adam = optim.AdamW(params_1d, lr=0.003, weight_decay=1e-4)
    
    # Cosine Annealing Schedulers for 100 epochs
    scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(opt_muon, T_max=100)
    scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=100)
    
    print("Starting training with Muon + AdamW (100 epochs)...")
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            opt_muon.zero_grad()
            opt_adam.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            l1_reg = 0.0
            for name, param in model.named_parameters():
                if "weight_eml" in name or "weight_base" in name:
                    l1_reg += torch.sum(torch.abs(param))
            loss = loss + 1e-5 * l1_reg
            
            loss.backward()
            opt_muon.step()
            opt_adam.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler_muon.step()
        scheduler_adam.step()
        
        acc = 100.0 * correct / total
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/100 | Loss: {running_loss/len(trainloader):.4f} | Accuracy: {acc:.2f}% | Muon LR: {scheduler_muon.get_last_lr()[0]:.6f}")
        
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
    print(f"Test Accuracy after 100 epochs: {100.0 * test_correct / test_total:.2f}%")
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight_eml" in name or "weight_base" in name:
                param[torch.abs(param) < 0.05] = 0.0
                
    model.to("cpu")
    generate_esp32_header(model, "large_scale_experiment/esp32_cifar100_inference.h")

if __name__ == "__main__":
    main()

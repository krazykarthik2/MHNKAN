import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import numpy as np

# 1. Define Muon Optimizer helper

def newton_schulz5(G, steps=3):
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
                
                shape = p.shape
                flat_p = p.view(shape[0], -1)
                flat_buf = buf.view(shape[0], -1)
                
                u = newton_schulz5(flat_buf, steps=ns_steps)
                p.add_(u.view(shape), alpha=-lr)
                
        return loss

# 2. Define EML-KAN Layers

class EMLKANActivation(nn.Module):
    def __init__(self, channels, num_components=2):
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

class EMLKANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_components=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = EMLKANActivation(out_features, num_components)
        
    def forward(self, x):
        return self.act(self.linear(x))

# 3. EML-KAN MobileNetCifar Classifier

class EMLKANMobileNetCifar(nn.Module):
    def __init__(self, num_classes=100, num_components=2):
        super().__init__()
        # Load pre-trained MobileNetV3 small backbone
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.backbone = mobilenet_v3_small(weights=weights).features
        
        # Keep backbone trainable for fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = EMLKANLinear(576, num_classes, num_components=num_components)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 4. Export to ESP32 C++ Code Generator & ONNX

def generate_esp32_header(model, filepath="esp32_cifar100_inference.h"):
    print(f"Generating C++ DAG inference header at {filepath}...")
    
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    fc = model.classifier.linear.weight.data.numpy()
    act3 = model.classifier.act
    
    with open(filepath, "w") as f:
        f.write("/* Automatically generated EML-KAN Classifier C++ ESP32 DAG inference header */\n")
        f.write("#ifndef EML_KAN_CLASSIFIER_H\n")
        f.write("#define EML_KAN_CLASSIFIER_H\n\n")
        f.write("#include <math.h>\n\n")
        
        # Stable softplus
        f.write("inline float softplus_stable(float z) {\n")
        f.write("    if (z > 20.0f) return z;\n")
        f.write("    if (z < -20.0f) return 0.0f;\n")
        f.write("    return logf(1.0f + expf(z));\n")
        f.write("}\n\n")
        
        # Write the DAG evaluation function
        f.write("inline void evaluate_eml_kan_classifier(const float* features, float* output_logits) {\n")
        
        for c in range(100):
            f.write(f"    // Class {c}\n")
            f.write(f"    float z_{c} = 0.0f;\n")
            
            # Find non-zero indices (skip pruned weights to speed up execution)
            non_zero_indices = np.where(np.abs(fc[c]) >= 0.05)[0]
            for idx in non_zero_indices:
                f.write(f"    z_{c} += features[{idx}] * {fc[c, idx]:.6f}f;\n")
                
            # Evaluate EML KAN activation for class c
            w_base = act3.weight_base[c].item()
            f.write(f"    float out_{c} = {w_base:.6f}f * z_{c};\n")
            
            for k in range(2):
                a = act3.a[c, k].item()
                b = act3.b[c, k].item()
                c_param = act3.c[c, k].item()
                d = act3.d[c, k].item()
                w_eml = act3.weight_eml[c, k].item()
                
                f.write(f"    {{\n")
                f.write(f"        float arg_x = {a:.6f}f * z_{c} + {b:.6f}f;\n")
                f.write(f"        if (arg_x < -10.0f) arg_x = -10.0f;\n")
                f.write(f"        if (arg_x > 10.0f) arg_x = 10.0f;\n")
                f.write(f"        float arg_y = softplus_stable({c_param:.6f}f * z_{c} + {d:.6f}f) + 1e-6f;\n")
                f.write(f"        out_{c} += {w_eml:.6f}f * (expf(arg_x) - logf(arg_y));\n")
                f.write(f"    }}\n")
                
            f.write(f"    output_logits[{c}] = out_{c};\n\n")
            
        f.write("}\n\n")
        f.write("#endif // EML_KAN_CLASSIFIER_H\n")
    print("ESP32 DAG classifier header generated successfully.")

# 5. Training Pipeline Loop

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)), # Resize for ImageNet filters
        transforms.RandomCrop(128, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Scale batch size to 256 for optimal update steps on 128x128 input sizes
    batch_size = 256 if torch.cuda.is_available() else 64
    num_workers = 2
    
    print("Loading CIFAR-100 dataset...")
    data_root = "./data"
    torchvision.datasets.CIFAR100.mirrors = [
        "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/",
        "https://raw.githubusercontent.com/uoip/cifar-mirror/master/",
        "https://www.cs.toronto.edu/~kriz/"
    ]
    full_trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
    c100_trainloader = torch.utils.data.DataLoader(full_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    full_testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
    c100_testloader = torch.utils.data.DataLoader(full_testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Instantiate MobileNetV3 + EML-KAN
    model = EMLKANMobileNetCifar(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Disable torch.compile to avoid compilation overhead on first epoch
    compiled_model = model
        
    # Group parameters for fine-tuning
    params_classifier_2d = [p for p in model.classifier.parameters() if p.requires_grad and p.ndim >= 2]
    params_classifier_1d = [p for p in model.classifier.parameters() if p.requires_grad and p.ndim < 2]
    params_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    
    opt_muon = Muon(params_classifier_2d, lr=0.03)
    # Train backbone weights with a very low learning rate to preserve features
    opt_adam = optim.AdamW([
        {'params': params_classifier_1d, 'lr': 0.003},
        {'params': params_backbone, 'lr': 0.0003}
    ], weight_decay=1e-4)
    
    scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(opt_muon, T_max=30)
    scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=30)
    
    scaler = torch.amp.GradScaler('cuda')
    
    print("\n--- Training EML-KAN MobileNet Classifier (30 epochs) ---")
    for epoch in range(30):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in c100_trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt_muon.zero_grad()
            opt_adam.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = compiled_model(inputs)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            
            scaler.unscale_(opt_muon)
            scaler.unscale_(opt_adam)
            
            scaler.step(opt_muon)
            scaler.step(opt_adam)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler_muon.step()
        scheduler_adam.step()
        print(f"Fine-Tuning Epoch {epoch+1}/30 | Loss: {running_loss/len(c100_trainloader):.4f} | Accuracy: {100.0*correct/total:.2f}%")
        
    # Evaluate CIFAR-100 model
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in c100_testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
    print(f"\nFinal CIFAR-100 Test Accuracy: {100.0 * test_correct / test_total:.2f}%")
    
    # Prune classifier weights
    with torch.no_grad():
        for name, param in model.classifier.named_parameters():
            if "weight_eml" in name or "weight_base" in name:
                param[torch.abs(param) < 0.05] = 0.0
                
    model.to("cpu")
    torch.save(model.state_dict(), "large_scale_experiment/eml_kan_model.pth")
    print("Saved trained model weights to large_scale_experiment/eml_kan_model.pth")
    
    header_path = "large_scale_experiment/esp32_project/esp32_cifar100_inference.h"
    if not os.path.exists("large_scale_experiment/esp32_project"):
        header_path = "esp32_project/esp32_cifar100_inference.h"
        
    generate_esp32_header(model, header_path)
    
    # Export full model to ONNX
    try:
        print("Exporting full model to ONNX...")
        dummy_input = torch.randn(1, 3, 128, 128)
        torch.onnx.export(model, dummy_input, "large_scale_experiment/eml_kan_mobilenet.onnx", 
                          input_names=['input'], output_names=['output'],
                          opset_version=12) # Use basic opset 12 for maximum compatibility
        print("ONNX model exported successfully to large_scale_experiment/eml_kan_mobilenet.onnx")
    except Exception as e:
        print(f"ONNX export failed: {e}")

if __name__ == "__main__":
    main()

import os
import csv
import sys
import struct
import random
import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# 1. Re-declare Model Class to Load Weights

class EMLKANActivation(nn.Module):
    def __init__(self, channels, num_components=2):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(channels, num_components))
        self.b = nn.Parameter(torch.zeros(channels, num_components))
        self.c = nn.Parameter(torch.zeros(channels, num_components))
        self.d = nn.Parameter(torch.zeros(channels, num_components))
        self.weight_base = nn.Parameter(torch.zeros(channels))
        self.weight_eml = nn.Parameter(torch.zeros(channels, num_components))

    def forward(self, x):
        out = self.weight_base * x
        for k in range(2):
            arg_x = torch.clamp(self.a[:, k] * x + self.b[:, k], min=-10.0, max=10.0)
            arg_y = torch.log(1.0 + torch.exp(self.c[:, k] * x + self.d[:, k])) + 1e-6
            out = out + self.weight_eml[:, k] * (torch.exp(arg_x) - torch.log(arg_y))
        return out

class EMLKANLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = EMLKANActivation(out_features)
        
    def forward(self, x):
        return self.act(self.linear(x))

class EMLKANMobileNetCifar(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None).features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = EMLKANLinear(576, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    parser = argparse.ArgumentParser(description="Serial Hardware-in-the-loop ESP32 Verification test pipeline")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port of the ESP32 (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate for Serial communication")
    parser.add_argument("--samples", type=int, default=20, help="Number of test samples to evaluate")
    args = parser.parse_args()
    
    # Try importing serial
    try:
        import serial
    except ImportError:
        print("Error: 'pyserial' module is not installed. Please run: pip install pyserial")
        sys.exit(1)
        
    print(f"Connecting to ESP32 on port {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=5)
        time.sleep(2) # Wait for ESP32 reboot handshake
        ser.reset_input_buffer()
        print("Connected successfully!")
    except Exception as e:
        print(f"Failed to connect to Serial port: {e}")
        print("Make sure your ESP32 is plugged in and you specified the correct port.")
        sys.exit(1)
        
    # Load PyTorch model weights
    print("Loading local trained model weights...")
    model = EMLKANMobileNetCifar(num_classes=100)
    model_path = "large_scale_experiment/eml_kan_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("Trained weights loaded successfully.")
    else:
        print(f"Warning: {model_path} not found. Running with random weights.")
    model.eval()
    
    # Load CIFAR-100 test dataset
    print("Loading CIFAR-100 test dataset...")
    # Override official Toronto CS mirrors with robust GitHub and HuggingFace mirrors to prevent 503 HTTP errors
    torchvision.datasets.CIFAR100.mirrors = [
        "https://raw.githubusercontent.com/uoip/cifar-mirror/master/",
        "https://huggingface.co/datasets/cifar100/resolve/main/",
        "https://www.cs.toronto.edu/~kriz/"
    ]
    # Add random transformations: slight shift, rotation, horizontal flip
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    # Output file setup
    csv_file = "large_scale_experiment/output.csv"
    csv_header = ["Sample_Idx", "True_Label", "PyTorch_Pred", "ESP32_Pred", "ESP32_Latency_us", "Match"]
    
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        
    print(f"\nEvaluating {args.samples} random augmented samples...")
    print("-" * 80)
    print(f"{'Sample':<8} | {'True Label':<12} | {'PyTorch Pred':<12} | {'ESP32 Pred':<12} | {'Latency (us)':<12} | {'Match?'}")
    print("-" * 80)
    
    matches = 0
    
    for idx in range(args.samples):
        # Pick a random sample
        sample_idx = random.randint(0, len(testset) - 1)
        image, target = testset[sample_idx]
        
        # Extract features locally using PyTorch backbone
        with torch.no_grad():
            features = model.backbone(image.unsqueeze(0))
            features = model.gap(features)
            features = torch.flatten(features, 1).squeeze(0).numpy()
            
            # Local PyTorch prediction
            logits = model.classifier(torch.tensor(features).unsqueeze(0))
            pytorch_pred = int(torch.argmax(logits, dim=1).item())
            
        # Transmit 576 floats (2304 bytes) to ESP32
        # Pack floats into little-endian binary bytes
        byte_data = struct.pack(f"<{len(features)}f", *features)
        
        # Send start sync signal
        ser.write(b"SYNC")
        ser.flush()
        time.sleep(0.01)
        
        # Send binary data
        ser.write(byte_data)
        ser.flush()
        
        # Read prediction and latency back from ESP32
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        
        esp32_pred = -1
        esp32_time = 0
        
        # Parse output from ESP32 (formatted as: "PRED: <class> TIME: <micros>")
        if "PRED:" in line and "TIME:" in line:
            try:
                parts = line.split()
                esp32_pred = int(parts[1])
                esp32_time = int(parts[3])
            except ValueError:
                pass
                
        match = "YES" if pytorch_pred == esp32_pred else "NO"
        if match == "YES":
            matches += 1
            
        # Print metrics
        print(f"{idx+1:<8} | {target:<12} | {pytorch_pred:<12} | {esp32_pred:<12} | {esp32_time:<12} | {match}")
        
        # Write to CSV
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([idx+1, target, pytorch_pred, esp32_pred, esp32_time, match])
            
        time.sleep(0.2) # Small delay between samples
        
    print("-" * 80)
    print(f"Evaluation complete. Match Rate (PyTorch vs ESP32): {100.0 * matches / args.samples:.2f}%")
    print(f"Results written to {csv_file}")
    ser.close()

if __name__ == "__main__":
    main()

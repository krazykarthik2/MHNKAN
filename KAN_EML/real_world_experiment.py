import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from eml_network import EMLKAN

def main():
    print("=" * 80)
    print("EML-KAN Real-World Dataset (Wine Classification) Experiment")
    print("=" * 80)
    
    # 1. Load and prepare Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # 2. Build model
    # Wine features: 13, Hidden: 8, Classes: 3
    model = EMLKAN(layers_hidden=[13, 8, 3], num_eml_components=3)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # 3. Training Loop
    epochs = 150
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 30 == 0 or epoch == 1:
            # Calculate train accuracy
            _, train_preds = torch.max(outputs, 1)
            train_acc = (train_preds == y_train_t).float().mean().item() * 100
            
            # Calculate test metrics
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_loss = criterion(test_outputs, y_test_t).item()
                _, test_preds = torch.max(test_outputs, 1)
                test_acc = (test_preds == y_test_t).float().mean().item() * 100
                
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            
    # 4. Final Evaluation
    model.eval()
    with torch.no_grad():
        final_test_outputs = model(X_test_t)
        _, final_test_preds = torch.max(final_test_outputs, 1)
        final_acc = (final_test_preds == y_test_t).float().mean().item() * 100
        
    print("=" * 80)
    print(f"Final EML-KAN Test Accuracy on Wine Dataset: {final_acc:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()

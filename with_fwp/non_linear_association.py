import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from fwp_eml_kan import EMLKANFWPModel, StandardLinearFWPModel

class NonLinearAssociativeDataset:
    """
    Generates synthetic sequences of non-linear Key-Value associations.
    The true value is a non-linear EML activation function of the key:
        V = exp(K) - ln(|K| + 1)
    """
    def __init__(self, num_samples, seq_len, d_model):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.d_model = d_model
        
    def generate_data(self):
        total_len = 2 * self.seq_len + 2
        
        X = torch.zeros(self.num_samples, total_len, self.d_model)
        Y = torch.zeros(self.num_samples, total_len, self.d_model)
        
        for i in range(self.num_samples):
            # Keys are random values in [-1, 1]
            keys = torch.rand(self.seq_len, self.d_model) * 2.0 - 1.0
            
            # Non-linear target values: V = exp(K) - ln(|K| + 1)
            values = torch.exp(keys) - torch.log(torch.abs(keys) + 1.0)
            
            # Choose one key to query
            query_idx = random.randint(0, self.seq_len - 1)
            query_key = keys[query_idx]
            target_value = values[query_idx]
            
            for j in range(self.seq_len):
                X[i, 2 * j] = keys[j]
                X[i, 2 * j + 1] = values[j]
                
            X[i, 2 * self.seq_len] = query_key
            Y[i, -1] = target_value
            
        return X, Y

def train_and_eval(model_class, name, X_train, Y_train, X_test, Y_test, epochs=50, d_model=8):
    if model_class == EMLKANFWPModel:
        model = EMLKANFWPModel(d_model=d_model, num_eml_components=2)
    else:
        model = StandardLinearFWPModel(d_model=d_model)
        
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    num_train = X_train.shape[0]
    batch_size = 32
    
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(num_train)
        
        for i in range(0, num_train, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs[:, -1, :], batch_y[:, -1, :])
            loss.backward()
            optimizer.step()
            
    # Final Eval
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        final_test_loss = criterion(test_outputs[:, -1, :], Y_test[:, -1, :]).item()
        
    print(f"{name} Final Test MSE: {final_test_loss:.6f}")
    return final_test_loss

def main():
    print("=" * 80)
    print("Evaluating EML-KAN FWP vs Standard Linear FWP on Non-Linear Association Task")
    print("=" * 80)
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    d_model = 8
    seq_len = 3
    
    dataset = NonLinearAssociativeDataset(1500, seq_len, d_model)
    X_train, Y_train = dataset.generate_data()
    
    test_dataset = NonLinearAssociativeDataset(300, seq_len, d_model)
    X_test, Y_test = test_dataset.generate_data()
    
    # Train both models
    eml_loss = train_and_eval(EMLKANFWPModel, "EML-KAN FWP (Ours)", X_train, Y_train, X_test, Y_test, epochs=60, d_model=d_model)
    linear_loss = train_and_eval(StandardLinearFWPModel, "Standard Linear FWP (Baseline)", X_train, Y_train, X_test, Y_test, epochs=60, d_model=d_model)
    
    improvement = ((linear_loss - eml_loss) / linear_loss) * 100
    print("-" * 80)
    print(f"EML-KAN FWP MSE Reduction: {improvement:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()

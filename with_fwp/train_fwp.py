import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

from fwp_eml_kan import EMLKANFWPModel

class AssociativeRetrievalDataset:
    """
    Generates synthetic sequences of Key-Value associations followed by a Query.
    Example sequence:
        x_seq = [K_0, V_0, K_1, V_1, ..., Q, zeros]
        y_target = [zeros, zeros, ..., target_V]
    Here, Q is equal to one of the previously presented keys, and the model
    must recall the corresponding value at the final step.
    """
    def __init__(self, num_samples, seq_len, d_model):
        self.num_samples = num_samples
        self.seq_len = seq_len # number of key-value pairs
        self.d_model = d_model
        
    def generate_data(self):
        # We construct input sequences of length: 2 * seq_len + 2
        # (K0, V0, K1, V1, ..., Kn, Vn, Q, dummy)
        total_len = 2 * self.seq_len + 2
        
        X = torch.zeros(self.num_samples, total_len, self.d_model)
        Y = torch.zeros(self.num_samples, total_len, self.d_model)
        
        for i in range(self.num_samples):
            # Generate random keys and values (normalized)
            keys = torch.randn(self.seq_len, self.d_model)
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)
            
            values = torch.randn(self.seq_len, self.d_model)
            values = values / torch.norm(values, dim=-1, keepdim=True)
            
            # Choose one key to query
            query_idx = random.randint(0, self.seq_len - 1)
            query_key = keys[query_idx]
            target_value = values[query_idx]
            
            # Fill inputs
            for j in range(self.seq_len):
                X[i, 2 * j] = keys[j]
                X[i, 2 * j + 1] = values[j]
                
            X[i, 2 * self.seq_len] = query_key
            # The last element X[i, 2*seq_len + 1] remains zero (dummy)
            
            # The target output is only evaluated at the final step
            Y[i, -1] = target_value
            
        return X, Y

def main():
    print("=" * 80)
    print("EML-KAN Fast Weight Programmer (FWP) Associative Retrieval Experiment")
    print("=" * 80)
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Parameters
    d_model = 16
    seq_len = 4 # 4 key-value pairs
    num_train = 1000
    num_test = 200
    epochs = 40
    batch_size = 32
    
    # Generate data
    dataset = AssociativeRetrievalDataset(num_train, seq_len, d_model)
    X_train, Y_train = dataset.generate_data()
    
    test_dataset = AssociativeRetrievalDataset(num_test, seq_len, d_model)
    X_test, Y_test = test_dataset.generate_data()
    
    # Create model
    model = EMLKANFWPModel(d_model=d_model, num_eml_components=2)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    print(f"Dataset summary:")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Sequence length: {X_train.shape[1]}")
    print(f"  Model size (d_model): {d_model}")
    print(f"Starting training...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(num_train)
        epoch_loss = 0.0
        
        for i in range(0, num_train, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # We only calculate loss on the last step (retrieval step)
            loss = criterion(outputs[:, -1, :], batch_y[:, -1, :])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(indices)
            
        epoch_loss /= num_train
        
        if epoch % 5 == 0 or epoch == 1:
            # Eval on test set
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs[:, -1, :], Y_test[:, -1, :]).item()
            print(f"Epoch {epoch:2d} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f}")
            
    print("\nTraining completed.")
    
    # Print a retrieval demo
    model.eval()
    with torch.no_grad():
        sample_x = X_test[0:1]
        sample_y = Y_test[0:1]
        pred_y = model(sample_x)
        
        cos_sim = torch.cosine_similarity(pred_y[0, -1, :], sample_y[0, -1, :], dim=0).item()
        print("Demo retrieval:")
        print(f"  Expected target vector (first 5 elements): {sample_y[0, -1, :5].tolist()}")
        print(f"  Predicted target vector (first 5 elements): {pred_y[0, -1, :5].tolist()}")
        print(f"  Cosine Similarity: {cos_sim:.4f}")
        
if __name__ == "__main__":
    main()

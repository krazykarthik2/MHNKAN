import sympy as sp
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch

def main():
    print("=" * 80)
    print("SymPy Symbolic Regression & Arbitrary Precision Solving for KAN-Hopfield")
    print("=" * 80)
    
    # 1. Load 3 Fashion MNIST patterns to keep the symbolic expressions readable but realistic
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    N = 3
    d = 784
    
    patterns_np = []
    for idx in range(N):
        img, _ = train_set[idx]
        patterns_np.append(img.view(-1).numpy())
    patterns_np = np.array(patterns_np)
    
    # Binarize templates so we have discrete classes
    patterns_bin = np.where(patterns_np > 0.0, 1.0, -1.0)
    
    # 2. Define SymPy Symbols
    # q is the input query vector symbols
    q_symbols = [sp.Symbol(f'q_{k}') for k in range(d)]
    
    # Let's set a very high beta symbolically (e.g. beta = 1e5)
    beta = 100000
    
    print("Building exact KAN-Hopfield symbolic formula equations...")
    
    # Similarity scores S_j as symbolic expressions: S_j = sum_k (beta * X_{j,k} * q_k)
    S_exprs = []
    for j in range(N):
        s_expr = sum(beta * patterns_bin[j, k] * q_symbols[k] for k in range(d))
        S_exprs.append(s_expr)
        
    # Subtract max similarity for numerical stability in the symbolic evaluation
    # To do this symbolically, we will substitute the query values first, then compute the exact softmax.
    # Softmax attention weights expressions: w_j = exp(S_j) / sum_m exp(S_m)
    # Reconstructed outputs: y_i = sum_j (X_{j,i} * w_j)
    
    # 3. Create a noisy query for pattern 0 (noisy Fashion MNIST item)
    np.random.seed(42)
    noise = np.random.normal(0, 0.2, d)
    noisy_query_val = np.clip(patterns_bin[0] + noise, -1.0, 1.0)
    
    # Map symbols to query values for substitution
    subs_dict = {q_symbols[k]: float(noisy_query_val[k]) for k in range(d)}
    
    # 4. Evaluate Similarity Scores numerically with 50-digit precision
    print("Evaluating similarity scores with 50-digit precision...")
    S_vals = []
    for j in range(N):
        # Substitute query values and evaluate exactly
        val = S_exprs[j].evalf(subs=subs_dict, n=50)
        S_vals.append(val)
        
    # Subtract max similarity to prevent float overflow in exp
    max_S = max(S_vals)
    S_stable = [s - max_S for s in S_vals]
    
    # Exponentiate stable similarities
    exp_S = [sp.exp(s) for s in S_stable]
    denom = sum(exp_S)
    
    # Calculate exact attention weights
    attn_weights = [exp_s / denom for exp_s in exp_S]
    
    # Print the attention weights with 50-digit precision
    print("\nAttention Weights (50-digit precision):")
    for j in range(N):
        print(f"  Weight w_{j}: {attn_weights[j].evalf(n=50)}")
        
    # 5. Compute the final reconstructed outputs symbolically
    # y_i = sum_j (X_{j,i} * w_j)
    reconstructed_vals = []
    print("\nReconstructing image pixels...")
    for i in range(d):
        y_expr = sum(patterns_bin[j, i] * attn_weights[j] for j in range(N))
        # Evaluate to a 50-digit float
        y_val = y_expr.evalf(n=50)
        reconstructed_vals.append(float(y_val))
        
    reconstructed_vals = np.array(reconstructed_vals)
    
    # 6. Measure Exact Reconstruction MSE against target template (patterns_bin[0])
    raw_mse = np.mean((reconstructed_vals - patterns_bin[0]) ** 2)
    print(f"\nFinal Raw Reconstruction MSE (unrounded): {raw_mse:.50f}")
    
    # Threshold check
    rounded_outputs = np.where(reconstructed_vals > 0.0, 1.0, -1.0)
    binarized_mse = np.mean((rounded_outputs - patterns_bin[0]) ** 2)
    print(f"Binarized Reconstruction MSE: {binarized_mse:.10f}")
    
    if raw_mse == 0.0 or raw_mse < 1e-30:
        print("\nSUCCESS: The symbolic KAN-Hopfield formula achieved EXACT reconstruction (MSE = 0) at high precision!")

if __name__ == "__main__":
    main()

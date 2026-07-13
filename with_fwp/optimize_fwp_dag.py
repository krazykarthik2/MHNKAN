import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from fwp_eml_kan import EMLKANFWPLayer

class FWPEMLDAGCompiler:
    """
    Compiles the dynamic retrieval step of EMLKANFWPLayer into flat, optimized
    Python code that eliminates standard PyTorch tensor layer overheads.
    """
    def __init__(self, layer, eps=1e-6):
        self.layer = layer
        self.eps = eps
        
    def generate_dag_code(self):
        in_features = self.layer.in_features
        out_features = self.layer.out_features
        num_components = self.layer.num_eml_components
        
        # Pull parameters to cpu
        a = self.layer.a.detach().cpu().numpy()
        b = self.layer.b.detach().cpu().numpy()
        c = self.layer.c.detach().cpu().numpy()
        d = self.layer.d.detach().cpu().numpy()
        
        code_lines = [
            "def eval_fwp_dag(q, W_base, W_eml):",
            "    # q: shape [in_features]",
            "    # W_base: shape [out_features, in_features]",
            "    # W_eml: shape [out_features, in_features, num_components]",
            "    import numpy as np",
            "    ",
            "    # 1. Base path: y_base = W_base @ q",
            "    y = np.dot(W_base, q)",
            "    ",
            "    # 2. EML KAN path",
        ]
        
        # Precompute the EML terms dynamically based on q
        for i in range(out_features):
            for j in range(in_features):
                for k in range(num_components):
                    # We compute: eml(a*q[j] + b, softplus(c*q[j] + d) + eps)
                    a_val = a[i, j, k]
                    b_val = b[i, j, k]
                    c_val = c[i, j, k]
                    d_val = d[i, j, k]
                    
                    code_lines.append(
                        f"    # Edge {j} -> {i}, component {k}"
                    )
                    # Division-free and using stable softplus
                    code_lines.append(
                        f"    arg_x = {a_val:.6f} * q[{j}] + {b_val:.6f}"
                    )
                    code_lines.append(
                        f"    arg_y = np.log(1.0 + np.exp({c_val:.6f} * q[{j}] + {d_val:.6f})) + {self.eps}"
                    )
                    code_lines.append(
                        f"    eml_val = np.exp(np.clip(arg_x, -10.0, 10.0)) - np.log(arg_y)"
                    )
                    code_lines.append(
                        f"    y[{i}] += W_eml[{i}, {j}, {k}] * eml_val"
                    )
                    
        code_lines.append("    return y")
        return "\n".join(code_lines)

def main():
    print("=" * 80)
    print("EML-KAN FWP Compilation & DAG Optimization Test")
    print("=" * 80)
    
    in_features = 4
    out_features = 4
    num_components = 2
    
    # Initialize layer
    layer = EMLKANFWPLayer(in_features, out_features, num_eml_components=num_components)
    compiler = FWPEMLDAGCompiler(layer)
    
    # Generate code
    print("Compiling retrieval step...")
    dag_code = compiler.generate_dag_code()
    
    # Let's inspect the compiled code (first 30 lines)
    print("\n--- Compiled DAG Code (Snippet) ---")
    lines = dag_code.split('\n')
    for line in lines[:30]:
        print(line)
    print("... [remaining lines omitted] ...")
    print("-----------------------------------\n")
    
    # Load function
    local_vars = {}
    exec(dag_code, globals(), local_vars)
    eval_fwp_dag = local_vars['eval_fwp_dag']
    
    # Test validation
    # Single sample
    q = torch.randn(in_features)
    W_base = torch.randn(out_features, in_features)
    W_eml = torch.randn(out_features, in_features, num_components)
    
    # PyTorch evaluation
    # Add batch dimensions
    q_batch = q.unsqueeze(0)
    W_base_batch = W_base.unsqueeze(0)
    W_eml_batch = W_eml.unsqueeze(0)
    
    layer.eval()
    with torch.no_grad():
        py_out = layer.retrieve(q_batch, W_base_batch, W_eml_batch).squeeze(0).numpy()
        
    # DAG evaluation
    q_np = q.numpy()
    W_base_np = W_base.numpy()
    W_eml_np = W_eml.numpy()
    dag_out = eval_fwp_dag(q_np, W_base_np, W_eml_np)
    
    # Compare
    mse = np.mean((py_out - dag_out) ** 2)
    print(f"Validation:")
    print(f"  PyTorch retrieved output: {py_out}")
    print(f"  DAG retrieved output:     {dag_out}")
    print(f"  MSE:                      {mse:.12f}")
    
    if mse < 1e-9:
        print("SUCCESS: Compiled DAG output matches PyTorch execution exactly!")
    else:
        print("WARNING: Output mismatch.")
        
    # Profile speeds
    print("\nProfiling performance (1000 runs)...")
    
    t0 = time.time()
    for _ in range(1000):
        with torch.no_grad():
            _ = layer.retrieve(q_batch, W_base_batch, W_eml_batch)
    pytorch_time = time.time() - t0
    
    t0 = time.time()
    for _ in range(1000):
        _ = eval_fwp_dag(q_np, W_base_np, W_eml_np)
    dag_time = time.time() - t0
    
    print(f"PyTorch Time: {pytorch_time:.6f} seconds")
    print(f"DAG Time:     {dag_time:.6f} seconds")
    print(f"Speedup:      {pytorch_time / dag_time:.2f}x")
    print("=" * 80)

if __name__ == "__main__":
    main()

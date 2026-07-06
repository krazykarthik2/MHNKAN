import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
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
        out = self.weight_base * x
        for k in range(self.num_components):
            arg_x = torch.clamp(self.a[:, k] * x + self.b[:, k], min=-10.0, max=10.0)
            arg_y = F.softplus(self.c[:, k] * x + self.d[:, k]) + 1e-6
            out = out + self.weight_eml[:, k] * (torch.exp(arg_x) - torch.log(arg_y))
        return out

class EMLKANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_components=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = EMLKANActivation(out_features, num_components)
        
    def forward(self, x):
        return self.act(self.linear(x))

class EMLKANFFNReplica(nn.Module):
    def __init__(self, d_model, num_components=2):
        super().__init__()
        self.layer1 = EMLKANLinear(d_model, d_model, num_components=num_components)
        self.ln = nn.LayerNorm(d_model)
        self.layer2 = EMLKANLinear(d_model, d_model, num_components=num_components)
        
    def forward(self, x):
        return self.layer2(self.ln(self.layer1(x)))

# 3. Dynamic Task Queue Worker

def distill_queue_worker(worker_id, task_queue, model_name, args_model, d_model, d_ffn, orig_params, device_str, return_dict):
    device = torch.device(device_str)
    print(f"[Worker {worker_id}] Started on device: {device}")
    
    # Load libraries locally
    from transformers import AutoModel, GPT2Model
    
    while not task_queue.empty():
        try:
            # Non-blocking get; if empty, throw empty exception
            layer_idx = task_queue.get_nowait()
        except Exception:
            break
            
        print(f"[Worker {worker_id} on {device_str}] Pulling Layer {layer_idx} from queue...")
        
        # Load FFN block dynamically
        if args_model == "gpt2":
            hf_model = GPT2Model.from_pretrained(model_name)
            mlp_block = hf_model.h[layer_idx].mlp
            def original_ffn(x):
                with torch.no_grad():
                    y = mlp_block(x)
                return y
        else:
            hf_model = AutoModel.from_pretrained(model_name)
            ffn_intermediate = hf_model.encoder.layer[layer_idx].intermediate
            ffn_output = hf_model.encoder.layer[layer_idx].output.dense
            def original_ffn(x):
                with torch.no_grad():
                    h = ffn_intermediate(x)
                    y = ffn_output(h)
                return y
                
        hf_model.to(device)
        hf_model.eval()
        
        # Setup KAN replica
        kan_replica = EMLKANFFNReplica(d_model, num_components=2).to(device)
        
        # Generate Synthetic Calibration Data (Zero-Data Distillation)
        X_train = torch.randn(50000, d_model).to(device)
        Y_train = original_ffn(X_train)
        
        X_test = torch.randn(2000, d_model).to(device)
        Y_test = original_ffn(X_test)
        
        # Free host memory of target model once targets are computed
        del hf_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # Group parameters
        params_2d = []
        params_1d = []
        for name, p in kan_replica.named_parameters():
            if p.requires_grad:
                if p.ndim >= 2:
                    params_2d.append(p)
                else:
                    params_1d.append(p)
                    
        opt_muon = Muon(params_2d, lr=0.02)
        opt_adam = optim.AdamW(params_1d, lr=0.002, weight_decay=1e-4)
        
        scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(opt_muon, T_max=100)
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=100)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        
        for epoch in range(100):
            kan_replica.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                opt_muon.zero_grad()
                opt_adam.zero_grad()
                outputs = kan_replica(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                opt_muon.step()
                opt_adam.step()
                epoch_loss += loss.item()
                
            scheduler_muon.step()
            scheduler_adam.step()
            
            if (epoch + 1) % 25 == 0 or epoch == 0:
                kan_replica.eval()
                with torch.no_grad():
                    test_outputs = kan_replica(X_test)
                    test_loss = criterion(test_outputs, Y_test).item()
                    cos_sim = F.cosine_similarity(test_outputs, Y_test).mean().item()
                print(f"[Layer {layer_idx} | Device {device_str}] Epoch {epoch+1:02d}/100 | Train Loss: {epoch_loss/len(loader):.6f} | Test MSE: {test_loss:.6f} | Cosine Sim: {cos_sim*100.0:.2f}%")
                
        # Final evaluation
        kan_replica.eval()
        with torch.no_grad():
            test_outputs = kan_replica(X_test)
            final_test_loss = criterion(test_outputs, Y_test).item()
            final_cos_sim = F.cosine_similarity(test_outputs, Y_test).mean().item()
            
        return_dict[layer_idx] = {
            "final_mse": final_test_loss,
            "final_cos_sim": final_cos_sim,
            "replica_params": sum(p.numel() for p in kan_replica.parameters() if p.requires_grad)
        }
        print(f"[Layer {layer_idx} | Device {device_str}] Done! Final Cosine Similarity: {final_cos_sim*100.0:.2f}%")
        
        # Cleanup KAN variables
        del kan_replica, X_train, Y_train, X_test, Y_test
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    print(f"[Worker {worker_id} on {device_str}] Completed all assigned tasks.")

# 4. Main Multi-processing Controller

def main():
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Parallelized Zero-Data Transformer FFN Layer Distillation")
    parser.add_argument("--model", type=str, default="bert-small", 
                        choices=["bert-tiny", "bert-small", "gpt2"],
                        help="Hugging Face model to distill: bert-tiny (4 layers), bert-small (4 layers), or gpt2 (12 layers)")
    args = parser.parse_args()
    
    print("Zero-Data FFN Layer Dynamic Queue Parallelized Distillation")
    print("=" * 60)
    
    if args.model == "bert-tiny":
        model_name = "prajjwal1/bert-tiny"
        d_model = 128
        d_ffn = 512
        num_layers = 4
    elif args.model == "bert-small":
        model_name = "prajjwal1/bert-small"
        d_model = 512
        d_ffn = 2048
        num_layers = 4
    elif args.model == "gpt2":
        model_name = "gpt2"
        d_model = 768
        d_ffn = 3072
        num_layers = 12
        
    if args.model == "gpt2":
        orig_params = d_model * d_ffn * 2 + d_ffn + d_model
    else:
        orig_params = d_model * d_ffn + d_ffn * d_model + d_ffn + d_model
        
    # Discover available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Discovered {num_gpus} CUDA devices for distillation.")
    
    # Initialize shared manager queue and output dict
    manager = mp.Manager()
    task_queue = manager.Queue()
    return_dict = manager.dict()
    
    # Add all layers as tasks into the queue
    for idx in range(num_layers):
        task_queue.put(idx)
        
    # Spawn exactly one persistent worker per available GPU (or 1 on CPU if none)
    num_workers = max(1, num_gpus)
    processes = []
    
    print(f"Spawning {num_workers} persistent worker processes...")
    for worker_id in range(num_workers):
        if num_gpus > 0:
            device_str = f"cuda:{worker_id}"
        else:
            device_str = "cpu"
            
        p = mp.Process(target=distill_queue_worker, args=(
            worker_id, task_queue, model_name, args.model, d_model, d_ffn, orig_params, device_str, return_dict
        ))
        p.start()
        processes.append(p)
        
    # Wait for all workers to finish their dynamic loops
    for p in processes:
        p.join()
        
    # Print Final Summary Report
    print("\n" + "=" * 80)
    print(f"DYNAMIC QUEUE DISTILLATION REPORT FOR: {model_name}")
    print("=" * 80)
    print(f"{'Layer Index':<12} | {'Original Params':<18} | {'EML-KAN Params':<18} | {'Test MSE':<12} | {'Cosine Sim'}")
    print("-" * 80)
    
    total_cos_sim = 0.0
    for idx in sorted(return_dict.keys()):
        stats = return_dict[idx]
        total_cos_sim += stats['final_cos_sim']
        print(f"{idx:<12} | {orig_params:<18,} | {stats['replica_params']:<18,} | {stats['final_mse']:<12.6f} | {stats['final_cos_sim']*100.0:.2f}%")
        
    print("-" * 80)
    print(f"Average Model Alignment (Cosine Similarity): {100.0 * total_cos_sim / num_layers:.2f}%")
    print("=" * 80)
    print("All layers distilled successfully!")

if __name__ == "__main__":
    main()

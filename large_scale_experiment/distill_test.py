import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np

# A comprehensive list of diverse, grammatically correct sentences to calibrate self-attention layers
CALIBRATION_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the landscape of modern technology and edge computing.",
    "Kolmogorov-Arnold Networks offer a promising alternative to multi-layer perceptrons.",
    "Deep learning models require careful quantization to run on low-power microcontrollers.",
    "We are evaluating the distillation performance of feed-forward networks in transformer models.",
    "Python is a versatile programming language widely used in data science and web development.",
    "The Hubble Space Telescope has captured stunning images of distant galaxies and nebulae.",
    "Quantum computing utilizes qubits and superposition to solve complex computational problems.",
    "Climate change poses a significant threat to global biodiversity and environmental stability.",
    "A healthy diet and regular exercise are essential for maintaining physical and mental well-being.",
    "The history of civilization is marked by technological innovations and cultural exchanges.",
    "Natural language processing allows computers to understand and generate human language.",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
    "The Great Wall of China is one of the most remarkable engineering feats in human history.",
    "Blockchain technology provides a decentralized and secure ledger for digital transactions.",
    "Learning a new language opens up new perspectives and opportunities for communication.",
    "The human brain is a complex network of billions of neurons communicating via synapses.",
    "Renewable energy sources such as solar and wind power are crucial for a sustainable future.",
    "Good communication skills are vital for personal relationships and professional success.",
    "The study of economics helps us understand how resources are allocated in society.",
    "Music has the power to evoke strong emotions and connect people across different cultures.",
    "The theory of relativity proposed by Albert Einstein revolutionized our understanding of space and time.",
    "Proper sleep hygiene is important for cognitive function and overall health preservation.",
    "The Amazon rainforest is home to a vast array of unique plant and animal species.",
    "Cryptography ensures secure communication in the presence of adversarial third parties.",
    "The scientific method involves observation, hypothesis formulation, and rigorous testing.",
    "Urban planning plays a key role in creating livable and sustainable cities.",
    "The discovery of penicillin by Alexander Fleming marked a turning point in medicine.",
    "Virtual reality creates immersive digital environments for gaming, education, and training.",
    "A positive attitude and perseverance can help individuals overcome major challenges in life.",
    "The ocean covers more than seventy percent of the Earth's surface and remains largely unexplored.",
    "Machine learning algorithms can identify patterns in large datasets to make predictions.",
    "The printing press invented by Johannes Gutenberg democratized access to information.",
    "Self-driving cars use sensors and artificial intelligence to navigate roads safely.",
    "The study of philosophy encourages critical thinking and questioning of fundamental assumptions.",
    "Biodegradable materials can help reduce plastic pollution and protect marine life.",
    "The internet has revolutionized the way we access information and communicate globally.",
    "A balanced ecosystem requires a delicate harmony between predators and prey.",
    "Microprocessors are the brain of modern electronic devices, from smartphones to supercomputers.",
    "The renaissance was a period of intense artistic and intellectual revival in Europe.",
    "Genomics is the study of the complete set of DNA within an organism.",
    "Sustainable agriculture practices help conserve soil fertility and water resources.",
    "The concept of democracy originated in ancient Greece and has evolved over centuries.",
    "Enzyme catalysts speed up chemical reactions in biological systems without being consumed.",
    "E-commerce has transformed the retail industry and changed consumer shopping habits.",
    "The standard model of particle physics describes the fundamental forces of nature.",
    "Public transport systems reduce traffic congestion and carbon emissions in urban areas.",
    "The library is a repository of human knowledge and a hub for community learning.",
    "Active listening is a critical component of effective leadership and teamwork.",
    "The industrial revolution shifted economies from agrarian to industrial and manufacturing-based."
]

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
        self.layer2 = EMLKANLinear(d_model, d_model, num_components=num_components)
        
    def forward(self, x):
        return self.layer2(self.layer1(x))

# 3. Dynamic Task Queue Worker

def distill_queue_worker(worker_id, task_queue, model_name, args_model, d_model, d_ffn, orig_params, device_str, return_dict):
    device = torch.device(device_str)
    print(f"[Worker {worker_id}] Started on device: {device}")
    
    # Load libraries locally
    from transformers import AutoModel, GPT2Model, AutoTokenizer
    
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
            # Extract statistics from LayerNorm preceding the MLP block
            ln_weight = hf_model.h[layer_idx].ln_2.weight.data.detach().clone().to(device)
            ln_bias = hf_model.h[layer_idx].ln_2.bias.data.detach().clone().to(device)
            
            def original_ffn(x):
                with torch.no_grad():
                    y = mlp_block(x)
                return y
        else:
            hf_model = AutoModel.from_pretrained(model_name)
            ffn_intermediate = hf_model.encoder.layer[layer_idx].intermediate
            ffn_output = hf_model.encoder.layer[layer_idx].output.dense
            # Extract statistics from LayerNorm preceding intermediate block (attention layer norm output)
            ln_weight = hf_model.encoder.layer[layer_idx].attention.output.LayerNorm.weight.data.detach().clone().to(device)
            ln_bias = hf_model.encoder.layer[layer_idx].attention.output.LayerNorm.bias.data.detach().clone().to(device)
            
            def original_ffn(x):
                with torch.no_grad():
                    h = ffn_intermediate(x)
                    y = ffn_output(h)
                return y
                
        hf_model.to(device)
        hf_model.eval()
        
        # Setup KAN replica (Increase components from 2 to 4 for higher representational capacity)
        kan_replica = EMLKANFFNReplica(d_model, num_components=4).to(device)
        
        # Setup dynamic tokenizer/forward hook tracking to capture real activations
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Tokenize calibration corpus once per worker
        calib_enc = tokenizer(CALIBRATION_CORPUS, padding=True, truncation=True, return_tensors="pt")
        calib_ids_base = calib_enc["input_ids"].to(device)
        calib_mask_base = calib_enc["attention_mask"].to(device)
        
        vocab_size = hf_model.config.vocab_size if hasattr(hf_model.config, 'vocab_size') else 30522
        
        # Hook target intermediate activations
        activation_buffer = []
        def target_hook_fn(module, input_tensor, output_tensor):
            # Input to intermediate layer (which is LN output)
            activation_buffer.append(input_tensor[0].detach())
            
        if args_model == "gpt2":
            hook_handle = hf_model.h[layer_idx].mlp.register_forward_hook(target_hook_fn)
        else:
            hook_handle = hf_model.encoder.layer[layer_idx].intermediate.register_forward_hook(target_hook_fn)
            
        def capture_real_features(size):
            activation_buffer.clear()
            
            # Subsample and inject noise into token IDs to generate infinite variations
            num_sentences = calib_ids_base.shape[0]
            indices = torch.randperm(num_sentences)[:max(1, size // 32)]
            
            input_ids = calib_ids_base[indices].clone()
            attention_mask = calib_mask_base[indices].clone()
            
            # Inject noise: randomly replace 15% of non-special tokens with random vocab words
            prob = torch.rand(input_ids.shape).to(device)
            mask = (prob < 0.15) & (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else True) & (input_ids != tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else True)
            random_tokens = torch.randint(0, vocab_size, input_ids.shape).to(device)
            input_ids[mask] = random_tokens[mask]
            
            with torch.no_grad():
                if args_model == "gpt2":
                    _ = hf_model(input_ids)
                else:
                    _ = hf_model(input_ids, attention_mask=attention_mask)
                    
            # Concat and return activations matching size
            feats = torch.cat(activation_buffer, dim=0).view(-1, d_model)
            if feats.shape[0] > size:
                feats = feats[:size]
            elif feats.shape[0] < size:
                # Pad to target size if sequence length was too short
                repeats = (size + feats.shape[0] - 1) // feats.shape[0]
                feats = feats.repeat(repeats, 1)[:size]
            return feats
            
        # Test baseline static set aligned with real dataset distribution
        X_test = capture_real_features(2000)
        Y_test = original_ffn(X_test)
        
        # Group parameters
        params_2d = []
        params_1d = []
        for name, p in kan_replica.named_parameters():
            if p.requires_grad:
                if p.ndim >= 2:
                    params_2d.append(p)
                else:
                    params_1d.append(p)
                    
        # Optimize learning rates for deeper KAN fitting
        opt_muon = Muon(params_2d, lr=0.05)
        opt_adam = optim.AdamW(params_1d, lr=0.005, weight_decay=1e-4)
        
        # Train for 300 epochs to squeeze out last 1% error
        epochs = 300
        scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(opt_muon, T_max=epochs)
        scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(opt_adam, T_max=epochs)
        criterion = nn.MSELoss()
        
        batches_per_epoch = 200
        batch_size = 256
        
        for epoch in range(epochs):
            kan_replica.train()
            epoch_loss = 0.0
            for _ in range(batches_per_epoch):
                opt_muon.zero_grad()
                opt_adam.zero_grad()
                
                # Dynamic calibration matching real attention manifolds
                batch_x = capture_real_features(batch_size)
                batch_y = original_ffn(batch_x)
                
                outputs = kan_replica(batch_x)
                # Joint Loss Objective: MSE + (1.0 - Cosine Proximity)
                mse_loss = criterion(outputs, batch_y)
                cos_loss = 1.0 - F.cosine_similarity(outputs, batch_y, dim=-1).mean()
                loss = mse_loss + cos_loss
                
                loss.backward()
                opt_muon.step()
                opt_adam.step()
                epoch_loss += mse_loss.item()
                
            scheduler_muon.step()
            scheduler_adam.step()
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                kan_replica.eval()
                with torch.no_grad():
                    test_outputs = kan_replica(X_test)
                    test_loss = criterion(test_outputs, Y_test).item()
                    cos_sim = F.cosine_similarity(test_outputs, Y_test).mean().item()
                print(f"[Layer {layer_idx} | Device {device_str}] Epoch {epoch+1:03d}/{epochs} | Train Loss: {epoch_loss/batches_per_epoch:.6f} | Test MSE: {test_loss:.6f} | Cosine Sim: {cos_sim*100.0:.2f}%")
                
        # Final evaluation
        kan_replica.eval()
        with torch.no_grad():
            test_outputs = kan_replica(X_test)
            final_test_loss = criterion(test_outputs, Y_test).item()
            final_cos_sim = F.cosine_similarity(test_outputs, Y_test).mean().item()
            
        # Save distilled layer model weights
        os.makedirs("distilled_weights", exist_ok=True)
        save_path = f"distilled_weights/layer_{layer_idx}.pth"
        torch.save(kan_replica.state_dict(), save_path)
        print(f"[Layer {layer_idx} | Device {device_str}] Saved weights to {save_path}")
        
        return_dict[layer_idx] = {
            "final_mse": final_test_loss,
            "final_cos_sim": final_cos_sim,
            "replica_params": sum(p.numel() for p in kan_replica.parameters() if p.requires_grad)
        }
        print(f"[Layer {layer_idx} | Device {device_str}] Done! Final Cosine Similarity: {final_cos_sim*100.0:.2f}%")
        
        # Remove the forward hook to prevent memory leaks and handle cleanup
        hook_handle.remove()
        
        # Cleanup KAN variables
        del kan_replica, X_test, Y_test
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
    parser.add_argument("--workers-per-gpu", type=int, default=4,
                        help="Number of parallel worker processes to run on each GPU (default: 4)")
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
        
    # Spawn multiple persistent workers per available GPU to fully utilize compute/memory
    if num_gpus > 0:
        num_workers = num_gpus * args.workers_per_gpu
    else:
        num_workers = 1
        
    processes = []
    
    print(f"Spawning {num_workers} persistent worker processes ({args.workers_per_gpu} per GPU)...")
    for worker_id in range(num_workers):
        if num_gpus > 0:
            device_id = worker_id % num_gpus
            device_str = f"cuda:{device_id}"
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

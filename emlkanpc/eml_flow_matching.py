import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from eml_pc import EMLActivation

class EMLFlowMatchingModel(nn.Module):
    def __init__(self, num_components=3):
        super().__init__()
        # Input: 784 (noisy image) + 10 (one-hot class) = 794
        self.input_dim = 784 + 10
        self.hidden_dim = 128
        self.output_dim = 784
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.act = EMLActivation(self.hidden_dim, num_components)
        
        # FiLM layers for time embedding modulation
        self.time_scale = nn.Linear(16, self.hidden_dim)
        self.time_shift = nn.Linear(16, self.hidden_dim)
        
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Xavier Initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        nn.init.zeros_(self.time_scale.weight)
        nn.init.ones_(self.time_scale.bias)
        nn.init.zeros_(self.time_shift.weight)
        nn.init.zeros_(self.time_shift.bias)

    def sinusoidal_time_embedding(self, t, dim=16):
        """
        Embeds time t (shape [B, 1] or [B]) to sinusoidal features shape [B, dim].
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half_dim = dim // 2
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device) * 
            -(math.log(10000.0) / (half_dim - 1))
        )
        args = t * frequencies.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

    def forward(self, x, t, y_one_hot):
        """
        Predicts the vector field v(x, t) conditional on class label and modulated by time t.
        """
        inp = torch.cat([x, y_one_hot], dim=-1)
        h = self.fc1(inp)
        
        t_emb = self.sinusoidal_time_embedding(t)
        scale = self.time_scale(t_emb)
        shift = self.time_shift(t_emb)
        
        h_modulated = h * scale + shift
        h_act = self.act(h_modulated)
        return self.fc2(h_act)

    @torch.no_grad()
    def sample(self, class_label, steps=15, noise_std=0.05, device='cpu'):
        """
        Generates a digit conditional on a class label using Euler integration starting from scaled noise.
        """
        self.eval()
        x = torch.randn(1, 784, device=device) * noise_std
        y_one_hot = torch.zeros(1, 10, device=device)
        y_one_hot[0, class_label] = 1.0
        
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t = torch.tensor([t_val], device=device, dtype=torch.float32)
            v = self.forward(x, t, y_one_hot)
            x = x + v * dt
            
        return torch.clamp(x.squeeze(0), 0.0, 1.0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class EMLKANFWPLayer(nn.Module):
    """
    EML-KAN Fast Weight Programmer Layer.
    Slow parameters (a, b, c, d) are trained via gradient descent.
    Fast weights (W_base, W_eml) are updated dynamically per sequence step.
    """
    def __init__(self, in_features, out_features, num_eml_components=2, eps=1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_eml_components = num_eml_components
        self.eps = eps
        
        # Slow parameters that govern the shape of the EML basis functions
        self.a = nn.Parameter(torch.randn(out_features, in_features, num_eml_components) * 0.5)
        self.b = nn.Parameter(torch.randn(out_features, in_features, num_eml_components) * 0.1)
        self.c = nn.Parameter(torch.randn(out_features, in_features, num_eml_components) * 0.5)
        self.d = nn.Parameter(torch.randn(out_features, in_features, num_eml_components) * 0.1)

    def eml(self, x, y):
        # eml(x, y) = exp(x) - ln(y)
        return torch.exp(torch.clamp(x, -10, 10)) - torch.log(torch.clamp(y, min=self.eps))

    def init_fast_weights(self, batch_size, device, dtype):
        # W_base shape: [batch, out_features, in_features]
        # W_eml shape: [batch, out_features, in_features, num_eml_components]
        W_base = torch.zeros(batch_size, self.out_features, self.in_features, device=device, dtype=dtype)
        W_eml = torch.zeros(batch_size, self.out_features, self.in_features, self.num_eml_components, device=device, dtype=dtype)
        return W_base, W_eml

    def update_weights(self, W_base, W_eml, k_base, v_base, k_eml, v_eml):
        """
        Updates the fast weights using outer products.
        k_base: [batch, in_features]
        v_base: [batch, out_features]
        k_eml: [batch, in_features]
        v_eml: [batch, out_features, num_eml_components]
        """
        # Update W_base: W_base_t = W_base_t-1 + v_base x k_base
        W_base = W_base + torch.bmm(v_base.unsqueeze(-1), k_base.unsqueeze(1))
        
        # Update W_eml: W_eml_t = W_eml_t-1 + v_eml x k_eml
        # v_eml is [batch, out_features, num_components]
        # k_eml is [batch, in_features]
        W_eml = W_eml + torch.einsum('bik,bj->bijk', v_eml, k_eml)
        
        return W_base, W_eml

    def retrieve(self, q, W_base, W_eml):
        """
        Retrieves the output using the query and current fast weights.
        q: [batch, in_features]
        """
        batch_size = q.shape[0]
        
        # 1. Base retrieval path
        y_base = torch.bmm(W_base, q.unsqueeze(-1)).squeeze(-1) # [batch, out_features]
        
        # 2. EML KAN retrieval path
        # Expand query q to [batch, out_features, in_features, num_eml_components]
        q_expanded = q.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, self.out_features, self.in_features, self.num_eml_components
        )
        
        eml_arg_x = self.a * q_expanded + self.b
        eml_arg_y = F.softplus(self.c * q_expanded + self.d) + self.eps
        
        eml_out = self.eml(eml_arg_x, eml_arg_y) # [batch, out_features, in_features, num_eml_components]
        
        # Multiply element-wise by fast weights and sum over input dims & components
        y_eml = torch.sum(W_eml * eml_out, dim=(2, 3)) # [batch, out_features]
        
        return y_base + y_eml


class EMLKANFWPModel(nn.Module):
    """
    A sequence processor utilizing EMLKANFWP layers.
    """
    def __init__(self, d_model, num_eml_components=2, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        
        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_base_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_eml_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.v_base_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_eml_proj = nn.Linear(d_model, d_model * num_eml_components, bias=False)
        
        # EML KAN FWP Layer
        self.fwp_layer = EMLKANFWPLayer(d_model, d_model, num_eml_components=num_eml_components, eps=eps)
        
    def forward(self, x):
        """
        x shape: [batch_size, seq_len, d_model]
        Returns: outputs [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        dtype = x.dtype
        
        # Initialize fast weights
        W_base, W_eml = self.fwp_layer.init_fast_weights(batch_size, device, dtype)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :] # [batch, d_model]
            
            # 1. Project to Q, K, V
            q_t = self.q_proj(x_t)
            k_base_t = self.k_base_proj(x_t)
            k_eml_t = self.k_eml_proj(x_t)
            v_base_t = self.v_base_proj(x_t)
            v_eml_t = self.v_eml_proj(x_t).view(batch_size, d_model, self.fwp_layer.num_eml_components)
            
            # 2. Retrieve output using current fast weights
            y_t = self.fwp_layer.retrieve(q_t, W_base, W_eml)
            outputs.append(y_t.unsqueeze(1))
            
            # 3. Update fast weights for the next step
            W_base, W_eml = self.fwp_layer.update_weights(
                W_base, W_eml, k_base_t, v_base_t, k_eml_t, v_eml_t
            )
            
        return torch.cat(outputs, dim=1)


class StandardLinearFWPModel(nn.Module):
    """
    A baseline Standard Linear Fast Weight Programmer model.
    It performs standard linear retrieval: y_t = W_t q_t.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        device = x.device
        dtype = x.dtype
        
        # Initialize fast weight matrix: [batch, d_model, d_model]
        W = torch.zeros(batch_size, d_model, d_model, device=device, dtype=dtype)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            q_t = self.q_proj(x_t)
            k_t = self.k_proj(x_t)
            v_t = self.v_proj(x_t)
            
            # Retrieve output: y_t = W_t @ q_t
            y_t = torch.bmm(W, q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(y_t.unsqueeze(1))
            
            # Update fast weight matrix: W_t = W_t-1 + v_t x k_t
            W = W + torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))
            
        return torch.cat(outputs, dim=1)


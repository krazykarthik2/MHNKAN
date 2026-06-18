import torch
import torch.nn as nn
import torch.nn.functional as F

class EMLKANLayer(nn.Module):
    """
    A KAN layer enhanced with the Exp-Minus-Log (EML) operator.
    The univariate function on each edge is parameterized as:
        phi(x) = w_base * x + sum_k w_eml_k * eml(a_k * x + b_k, softplus(c_k * x + d_k) + eps)
    where eml(x, y) = exp(x) - ln(y).
    """
    def __init__(self, in_features, out_features, num_eml_components=4, eps=1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_eml_components = num_eml_components
        self.eps = eps
        
        # Base linear path weights
        self.weight_base = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # EML mixture weights
        self.weight_eml = nn.Parameter(torch.randn(out_features, in_features, num_eml_components) * 0.1)
        
        # Parameter matrices for the EML inputs
        # eml(a * x + b, softplus(c * x + d) + eps)
        self.a = nn.Parameter(torch.randn(out_features, in_features, num_eml_components) * 0.5)
        self.b = nn.Parameter(torch.randn(out_features, in_features, num_eml_components) * 0.1)
        self.c = nn.Parameter(torch.randn(out_features, in_features, num_eml_components) * 0.5)
        self.d = nn.Parameter(torch.randn(out_features, in_features, num_eml_components) * 0.1)
        
    def eml(self, x, y):
        # eml(x, y) = exp(x) - ln(y)
        # y is constrained to be strictly positive (y >= eps)
        return torch.exp(torch.clamp(x, -10, 10)) - torch.log(torch.clamp(y, min=self.eps))

    def forward(self, x):
        # x shape: [batch, in_features]
        batch_size = x.shape[0]
        
        # 1. Base linear transformation path
        y_base = F.linear(x, self.weight_base) # [batch, out_features]
        
        # 2. EML expansion path
        # Expand input x to [batch, out_features, in_features, K]
        x_expanded = x.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.out_features, self.in_features, self.num_eml_components)
        
        # Compute x and y arguments for the EML operator
        eml_arg_x = self.a * x_expanded + self.b
        # Ensure the log argument is positive using softplus + eps
        eml_arg_y = F.softplus(self.c * x_expanded + self.d) + self.eps
        
        # Evaluate EML operator
        eml_out = self.eml(eml_arg_x, eml_arg_y) # [batch, out_features, in_features, K]
        
        # Multiply by EML mixture weights and sum over components K and input features j
        # y_eml_j = sum_k weight_eml_{i,j,k} * eml_out_{batch, i, j, k}
        weighted_eml = eml_out * self.weight_eml.unsqueeze(0) # [batch, out_features, in_features, K]
        y_eml = torch.sum(weighted_eml, dim=(2, 3)) # [batch, out_features]
        
        return y_base + y_eml

class EMLKAN(nn.Module):
    """
    A multi-layer KAN constructed using EMLKANLayers.
    """
    def __init__(self, layers_hidden, num_eml_components=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for idx in range(len(layers_hidden) - 1):
            self.layers.append(
                EMLKANLayer(
                    in_features=layers_hidden[idx],
                    out_features=layers_hidden[idx+1],
                    num_eml_components=num_eml_components
                )
            )
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EMLActivation(nn.Module):
    def __init__(self, channels, num_components=4):
        super().__init__()
        self.channels = channels
        self.num_components = num_components
        
        self.a = nn.Parameter(torch.randn(channels, num_components) * 0.02)
        self.b = nn.Parameter(torch.zeros(channels, num_components))
        self.c = nn.Parameter(torch.randn(channels, num_components) * 0.02)
        self.d = nn.Parameter(torch.zeros(channels, num_components))
        
        self.weight_base = nn.Parameter(torch.ones(channels) * 1.0)
        self.weight_eml = nn.Parameter(torch.randn(channels, num_components) * 0.05)
        self.epsilon = 1e-6

    def forward(self, x):
        out, _, _ = self.fused_all(x)
        return out

    def fused_all(self, x):
        """
        Computes forward pass f(x), first derivative f'(x), and second derivative f''(x)
        sharing intermediate computations. Uses a smooth soft-clamping function to map
        exponent terms to a safe range [-40, 40] smoothly, ensuring no torch.inf or flat gradients.
        """
        out = self.weight_base * x
        deriv = self.weight_base.unsqueeze(0).expand(x.shape[0], -1).clone()
        second_deriv = torch.zeros_like(x)
        
        x_unsqueezed = x.unsqueeze(-1)
        
        a = self.a.unsqueeze(0)
        b = self.b.unsqueeze(0)
        c = self.c.unsqueeze(0)
        d = self.d.unsqueeze(0)
        w_eml = self.weight_eml.unsqueeze(0)
        
        u1 = a * x_unsqueezed + b
        
        # Stable soft-clamping of u1 to prevent overflows to inf:
        # For |u1| > 30, we smoothly scale it using tanh so it never exceeds 40.
        u1_gt30 = u1 > 30.0
        u1_ltm30 = u1 < -30.0
        
        tanh_pos = torch.tanh((u1 - 30.0) / 10.0)
        tanh_neg = torch.tanh((u1 + 30.0) / 10.0)
        
        s_u1 = torch.where(u1_gt30, 30.0 + 10.0 * tanh_pos,
                           torch.where(u1_ltm30, -30.0 + 10.0 * tanh_neg, u1))
        
        # s'(u1)
        s_prime = torch.where(u1_gt30, 1.0 - tanh_pos ** 2,
                              torch.where(u1_ltm30, 1.0 - tanh_neg ** 2, torch.ones_like(u1)))
        
        # s''(u1)
        s_double_prime = torch.where(u1_gt30, -0.2 * tanh_pos * (1.0 - tanh_pos ** 2),
                                     torch.where(u1_ltm30, -0.2 * tanh_neg * (1.0 - tanh_neg ** 2), torch.zeros_like(u1)))
        
        exp_u1 = torch.exp(s_u1)
        
        u2 = c * x_unsqueezed + d
        sig_u2 = torch.sigmoid(u2)
        sp_u2 = F.softplus(u2) + self.epsilon
        
        # 1. Function output
        eml_terms = exp_u1 - torch.log(sp_u2)
        out = out + torch.sum(w_eml * eml_terms, dim=-1)
        
        # 2. First derivative (chain rule with s_prime)
        term2_1 = c * sig_u2 / sp_u2
        deriv_terms = w_eml * (a * exp_u1 * s_prime - term2_1)
        deriv = deriv + torch.sum(deriv_terms, dim=-1)
        
        # 3. Second derivative (product rule with s_prime and s_double_prime)
        sig_deriv = sig_u2 * (1.0 - sig_u2)
        h_prime = (sig_deriv * sp_u2 - sig_u2 ** 2) / (sp_u2 ** 2)
        
        exp_double_deriv = (a ** 2) * exp_u1 * (s_prime ** 2) + a * exp_u1 * s_double_prime
        second_terms = w_eml * (exp_double_deriv - (c ** 2) * h_prime)
        second_deriv = second_deriv + torch.sum(second_terms, dim=-1)
        
        return out, deriv, second_deriv

class EMLPredictiveCodingNetwork(nn.Module):
    def __init__(self, layer_sizes, num_components=4):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        
        # Xavier Initialization: std = sqrt(2 / input_dim)
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(layer_sizes[l], layer_sizes[l-1]) * math.sqrt(2.0 / layer_sizes[l]))
            for l in range(1, self.L + 1)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(layer_sizes[l-1]))
            for l in range(1, self.L + 1)
        ])
        
        self.acts = nn.ModuleList([
            EMLActivation(layer_sizes[l], num_components)
            for l in range(self.L)
        ])

    def forward_predict(self, x_l, l):
        u = torch.matmul(x_l, self.W[l-1]) + self.b[l-1]
        return self.acts[l-1](u)

    def infer(self, x_0, x_L=None, steps=20, lr=0.5, damping=1e-3, use_newton=True):
        """
        Runs state inference (settling phase).
        If use_newton is True, uses analytical diagonal Second-Order Gauss-Newton updates.
        Damping prevents zero-division and controls step size in steep regions.
        """
        batch_size = x_0.shape[0]
        
        xs = [None] * (self.L + 1)
        xs[0] = x_0.clone()
        
        with torch.no_grad():
            for l in range(1, self.L + 1):
                proj = torch.matmul(xs[l-1], self.W[l-1].t())
                xs[l] = proj.clone()
        
        if x_L is not None:
            xs[self.L] = x_L.clone()
            
        update_range = range(1, self.L) if x_L is not None else range(1, self.L + 1)
        
        for l in update_range:
            xs[l].requires_grad = True
            
        energy_history = []
        
        for step in range(steps):
            errors = []
            energy = 0.0
            
            # Predict and evaluate activation, first, and second derivatives
            acts_out = []
            acts_deriv = []
            acts_second_deriv = []
            
            for l in range(self.L):
                u = torch.matmul(xs[l+1], self.W[l]) + self.b[l]
                p_l, d_l, sd_l = self.acts[l].fused_all(u)
                acts_out.append(p_l)
                acts_deriv.append(d_l)
                acts_second_deriv.append(sd_l)
                
                e_l = xs[l] - p_l
                errors.append(e_l)
                energy += 0.5 * torch.sum(e_l ** 2)
            
            energy_history.append(energy.item() / batch_size)
            
            with torch.no_grad():
                new_xs = [x.clone() if x is not None else None for x in xs]
                for l in update_range:
                    e_lm1 = errors[l-1]
                    d_l = acts_deriv[l-1]
                    sd_l = acts_second_deriv[l-1]
                    
                    # Local Gradient
                    grad_back = torch.matmul(e_lm1 * d_l, self.W[l-1].t())
                    if l < self.L:
                        grad = errors[l] - grad_back
                    else:
                        grad = -grad_back
                    
                    if use_newton:
                        # Gauss-Newton approximation (always positive-definite):
                        # H = I + d_l^2 @ W_l^2.t()
                        W_sq = self.W[l-1] ** 2
                        curvature = d_l ** 2
                        hessian_back = torch.matmul(curvature, W_sq.t())
                        
                        if l < self.L:
                            hessian = 1.0 + hessian_back
                        else:
                            hessian = hessian_back
                            
                        step_update = grad / (hessian + damping)
                        new_xs[l] = xs[l] - lr * step_update
                    else:
                        new_xs[l] = xs[l] - lr * grad
                
                for l in update_range:
                    xs[l] = new_xs[l].clone()
                    xs[l].requires_grad = True
                    
        return energy_history, xs

    def update_weights(self, xs, lr=0.01, weight_decay=1e-4, grad_clip=0.5):
        """Local update rule for weights and biases minimizing local prediction error energy"""
        with torch.no_grad():
            for l in range(1, self.L + 1):
                p_lm1 = self.forward_predict(xs[l], l)
                e_lm1 = xs[l-1] - p_lm1
                u = torch.matmul(xs[l], self.W[l-1]) + self.b[l-1]
                _, act_deriv, _ = self.acts[l-1].fused_all(u)
                
                delta = e_lm1 * act_deriv
                
                # Compute gradients and normalize by batch size
                batch_sz = xs[l].shape[0]
                dW = torch.matmul(xs[l].t(), delta) / batch_sz
                db = torch.sum(delta, dim=0) / batch_sz
                
                # Clip gradients to prevent explosions
                dW = torch.clamp(dW, min=-grad_clip, max=grad_clip)
                db = torch.clamp(db, min=-grad_clip, max=grad_clip)
                
                # Update weights with L2 regularization
                self.W[l-1].mul_(1.0 - weight_decay)
                self.W[l-1].add_(dW * lr)
                self.b[l-1].add_(db * lr)

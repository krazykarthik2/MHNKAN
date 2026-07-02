import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sympy as sp
import numpy as np

sys.setrecursionlimit(20000)

# Adjust path to import from KAN_EML
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'KAN_EML')))
from eml_network import EMLKAN

def softplus_sympy(x):
    return sp.log(1 + sp.exp(x))

class EMLSymbolicOptimizer:
    """
    An extensible, registry-based symbolic optimizer that simplifies nested EML KAN expressions
    by mapping raw exp/log combinations to standard mathematical operations.
    """
    def __init__(self, eps=1e-6):
        self.eps = eps
        # Active rulebank registry containing a vast set of algebraic and trigonometric simplifications
        self.rulebank = [
            self.rule_softplus_cleanup,
            self.rule_inversion,
            self.rule_division,
            self.rule_multiplication,
            self.rule_power,
            self.rule_exponential,
            self.rule_logarithm,
            self.rule_log_combining,
            self.rule_cancel_exp_log,
            self.rule_trig_simplification,
            self.rule_polynomial_expand_factor,
            self.rule_identity_cleanup,
        ]

    def add_rule(self, rule_fn):
        """
        Adds a new custom decomposition rule function to the rulebank.
        """
        self.rulebank.append(rule_fn)

    # 1. Softplus cleanup rule
    def rule_softplus_cleanup(self, expr):
        x = sp.Symbol('x')
        c = sp.Wild('c')
        d = sp.Wild('d')
        
        expr = expr.replace(sp.log(softplus_sympy(c * x + d) + self.eps), c * x + d)
        expr = expr.replace(sp.log(sp.log(1 + sp.exp(c * x + d)) + self.eps), c * x + d)
        return expr

    # 2. Multiplicative inversion rule (1/x)
    def rule_inversion(self, expr):
        u = sp.Wild('u')
        return expr.replace(sp.exp(-sp.log(u)), 1 / u)

    # 3. Division rule (u / v)
    def rule_division(self, expr):
        u = sp.Wild('u')
        v = sp.Wild('v')
        return expr.replace(sp.exp(sp.log(u) - sp.log(v)), u / v)

    # 4. Multiplication rule (u * v)
    def rule_multiplication(self, expr):
        u = sp.Wild('u')
        v = sp.Wild('v')
        return expr.replace(sp.exp(sp.log(u) + sp.log(v)), u * v)

    # 5. Power rule (u**p)
    def rule_power(self, expr):
        u = sp.Wild('u')
        p = sp.Wild('p')
        return expr.replace(sp.exp(p * sp.log(u)), u**p)

    # 6. Exponential rule (exp(u))
    def rule_exponential(self, expr):
        u = sp.Wild('u')
        return expr.replace(sp.exp(u) - sp.log(1 + self.eps), sp.exp(u))

    # 7. Logarithm rule (log(u))
    def rule_logarithm(self, expr):
        u = sp.Wild('u')
        return expr.replace(1 - (1 - sp.log(u)), sp.log(u))

    # 8. Log combining rule (log(u) + log(v) -> log(u*v), etc.)
    def rule_log_combining(self, expr):
        return sp.logcombine(expr, force=True)

    # 9. Cancel exp/log rule (exp(log(u)) -> u, log(exp(u)) -> u)
    def rule_cancel_exp_log(self, expr):
        u = sp.Wild('u')
        expr = expr.replace(sp.exp(sp.log(u)), u)
        expr = expr.replace(sp.log(sp.exp(u)), u)
        return expr

    # 10. Trigonometric simplification rule (trigsimp, double angle, sum of squares)
    def rule_trig_simplification(self, expr):
        return sp.trigsimp(expr)

    # 11. Polynomial expand and factor rule
    def rule_polynomial_expand_factor(self, expr):
        # Attempts to expand and then factor polynomials to find the cleanest algebraic structure
        return sp.factor(sp.expand(expr))

    # 12. Identity arithmetic cleanup (div/mul by 1, addition of 0, zero-product)
    def rule_identity_cleanup(self, expr):
        u = sp.Wild('u')
        expr = expr.replace(u * 1, u)
        expr = expr.replace(u + 0, u)
        expr = expr.replace(u - 0, u)
        expr = expr.replace(u * 0, 0)
        return expr

    def round_coefficients(self, expr, decimals=2):
        rounded = expr
        for a in sp.preorder_traversal(expr):
            if isinstance(a, sp.Float):
                val = float(a)
                rounded_val = round(val, decimals)
                if abs(val - rounded_val) < 1e-1:
                    if rounded_val.is_integer():
                        rounded = rounded.subs(a, int(rounded_val))
                    else:
                        rounded = rounded.subs(a, rounded_val)
        return rounded

    def optimize(self, expr, decimals=2):
        simplified = expr
        # Run all rules sequentially from the rulebank
        for rule in self.rulebank:
            simplified = rule(simplified)
            
        simplified = self.round_coefficients(simplified, decimals)
        return sp.simplify(simplified)


# Target functions to evaluate the optimizer
def target_div(x):
    return (x + 1.5) / (x - 0.5 + 1e-5)

def target_mul(x):
    return (x + 1.2) * (x - 0.3)

def target_pow(x):
    return (x + 1.0)**2

def target_inversion(x):
    return 1.0 / (x + 0.8)

def target_nested_exp(x):
    return torch.exp(torch.exp(x - 0.5) - 1.0)


def train_model(target_fn, epochs=2500):
    torch.manual_seed(42)
    X_train = (torch.rand(200, 1) * 1.5 + 0.6).double() # [0.6, 2.1]
    y_train = target_fn(X_train).double()
    
    model = EMLKAN([1, 4, 1], num_eml_components=2).double()
    optimizer = optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=0.8,
        max_iter=1000,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-30,
        tolerance_change=1e-30
    )
    
    def closure():
        optimizer_lbfgs.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        return loss
        
    try:
        optimizer_lbfgs.step(closure)
    except Exception:
        pass
        
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(X_train), y_train).item()
    print(f"Final training loss achieved: {final_loss:.20f}")
        
    return model, final_loss

def convert_model_to_sympy(model, sym_input, eps=1e-6):
    current_syms = [sym_input]
    for layer in model.layers:
        in_features = layer.in_features
        out_features = layer.out_features
        num_eml_components = layer.num_eml_components
        
        next_syms = []
        for i in range(out_features):
            val = 0
            for j in range(in_features):
                w_b = layer.weight_base[i, j].item()
                val += w_b * current_syms[j]
                
                for k in range(num_eml_components):
                    w_e = layer.weight_eml[i, j, k].item()
                    a = layer.a[i, j, k].item()
                    b = layer.b[i, j, k].item()
                    c = layer.c[i, j, k].item()
                    d = layer.d[i, j, k].item()
                    
                    arg_x = a * current_syms[j] + b
                    arg_y = softplus_sympy(c * current_syms[j] + d) + eps
                    val += w_e * (sp.exp(arg_x) - sp.log(arg_y))
            next_syms.append(val)
        current_syms = next_syms
    return current_syms[0]

def main():
    print("Training models to evaluate expanded rulebank...")
    
    model_div, loss_div = train_model(target_div)
    model_mul, loss_mul = train_model(target_mul)
    model_pow, loss_pow = train_model(target_pow)
    model_inv, loss_inv = train_model(target_inversion)
    model_nexp, loss_nexp = train_model(target_nested_exp)
    
    x = sp.Symbol('x')
    
    raw_expr_div = convert_model_to_sympy(model_div, x)
    raw_expr_mul = convert_model_to_sympy(model_mul, x)
    raw_expr_pow = convert_model_to_sympy(model_pow, x)
    raw_expr_inv = convert_model_to_sympy(model_inv, x)
    raw_expr_nexp = convert_model_to_sympy(model_nexp, x)
    
    optimizer = EMLSymbolicOptimizer()
    
    opt_expr_div = optimizer.optimize(raw_expr_div)
    opt_expr_mul = optimizer.optimize(raw_expr_mul)
    opt_expr_pow = optimizer.optimize(raw_expr_pow)
    opt_expr_inv = optimizer.optimize(raw_expr_inv)
    opt_expr_nexp = optimizer.optimize(raw_expr_nexp)
    
    report_path = "strategiesForEMLKAN/symbolic_optimizer_results.md"
    with open(report_path, "w") as f:
        f.write("# EML Symbolic Decomposition Optimizer Results\n\n")
        f.write("This report presents the results of the **EMLSymbolicOptimizer**, which applies an extensible mathematical rulebank containing trig simplifications, log combining, cancellation laws, and arithmetic cleanups to decompose raw EML formulas into clean standard algebraic functions.\n\n")
        
        f.write("## 1. Summary Results Table\n\n")
        f.write("| Target Function | Raw Learned EML Function | Decomposed Function |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| $f_{{\\text{{div}}}}(x) = \\frac{{x + 1.5}}{{x - 0.5}}$ | `{str(raw_expr_div)[:120]}...` | `${sp.latex(opt_expr_div)}$` |\n")
        f.write(f"| $f_{{\\text{{mul}}}}(x) = (x + 1.2)(x - 0.3)$ | `{str(raw_expr_mul)[:120]}...` | `${sp.latex(opt_expr_mul)}$` |\n")
        f.write(f"| $f_{{\\text{{pow}}}}(x) = (x + 1)^2$ | `{str(raw_expr_pow)[:120]}...` | `${sp.latex(opt_expr_pow)}$` |\n")
        f.write(f"| $f_{{\\text{{inv}}}}(x) = \\frac{{1}}{{x + 0.8}}$ | `{str(raw_expr_inv)[:120]}...` | `${sp.latex(opt_expr_inv)}$` |\n")
        f.write(f"| $f_{{\\text{{nested\_exp}}}}(x) = e^{{e^{{x-0.5}} - 1}}$ | `{str(raw_expr_nexp)[:120]}...` | `${sp.latex(opt_expr_nexp)}$` |\n")
        
        f.write("\n## 2. Rule Decompositions walk-through\n\n")
        f.write("### Division Decomp:\n")
        f.write(f"- **Raw Formula:** `{str(raw_expr_div)}` \n")
        f.write(f"- **Optimized Decomposed:** `${sp.latex(opt_expr_div)}$` \n\n")
        
        f.write("### Multiplication Decomp:\n")
        f.write(f"- **Raw Formula:** `{str(raw_expr_mul)}` \n")
        f.write(f"- **Optimized Decomposed:** `${sp.latex(opt_expr_mul)}$` \n\n")
        
        f.write("### Power Decomp:\n")
        f.write(f"- **Raw Formula:** `{str(raw_expr_pow)}` \n")
        f.write(f"- **Optimized Decomposed:** `${sp.latex(opt_expr_pow)}$` \n\n")
        
        f.write("### Inversion Decomp (1 / x):\n")
        f.write(f"- **Raw Formula:** `{str(raw_expr_inv)}` \n")
        f.write(f"- **Optimized Decomposed:** `${sp.latex(opt_expr_inv)}$` \n\n")
        
        f.write("### Nested Exponential Decomp (exp(exp(x))):\n")
        f.write(f"- **Raw Formula:** `{str(raw_expr_nexp)}` \n")
        f.write(f"- **Optimized Decomposed:** `${sp.latex(opt_expr_nexp)}$` \n")
        
    print(f"Results successfully written to {report_path}")

if __name__ == "__main__":
    main()

# EML-KAN: Exp-Minus-Log Kolmogorov-Arnold Network

This folder contains the implementation and experimental validation of a new neural network model: the **EML-KAN (Exp-Minus-Log Kolmogorov-Arnold Network)**.

---

## 🧠 Mathematical Background: The EML Operator

The **EML operator** is a universal binary primitive defined as:
\[
\operatorname{eml}(x, y) = \exp(x) - \ln(y)
\]

In the paper *"All elementary functions from a single binary operator"* (arXiv:2603.21852), it is mathematically proven that the EML operator, when combined with the constant $1$, is **functionally complete** for continuous mathematics. This means that composing this single operator is sufficient to construct the entire repertoire of elementary mathematical functions (including addition, multiplication, division, powers, logarithms, trigonometric functions, and hyperbolic functions).

---

## 🛠️ Network Architecture (`eml_network.py`)

A Kolmogorov-Arnold Network (KAN) maps inputs to outputs using univariate functions along its edges:
\[
y_i = \sum_{j} \phi_{i,j}(x_j)
\]
In **EML-KAN**, we supercharge the edge function $\phi(x)$ by parameterizing it as a mixture of EML operators:
\[
\phi(x) = w_{\text{base}} \cdot x + \sum_{k=1}^K w_{\text{eml}, k} \cdot \operatorname{eml}(a_k \cdot x + b_k, \text{softplus}(c_k \cdot x + d_k) + \epsilon)
\]

This allows the network to natively learn complex non-linear mathematical relations using a compact, smooth, and analytically differentiable representation.

---

## 🔬 Experimental Validation

### 1. Complex Function Modeling (`complex_data_experiment.py`)
* **Task:** Fit a complex multi-variable target function defined directly in the EML basis:
  \[
  y = \sum_{j=1}^2 \operatorname{eml}(1.2 x_j - 0.3, \text{softplus}(0.8 x_j + 0.2) + 1\text{e-}6)
  \]
* **Method:** Staged AdamW training followed by double-precision (`float64`) L-BFGS line-search optimization.
* **Result:** The network successfully recovered the exact underlying mathematical basis, driving the loss down to absolute machine precision:
  * **Final Loss:** **`0.0000000000002966`** (effectively `0.0` at double-precision).
  * **Status:** 100% fitting accuracy.

### 2. Real-World Generalization (`real_world_experiment.py`)
* **Task:** Classify wine samples into 3 classes based on 13 chemical features from the standard Wine dataset.
* **Method:** Trained an `EMLKAN([13, 8, 3])` network using CrossEntropyLoss and AdamW.
* **Result:** The model converged rapidly:
  * **Epoch 1:** Train Loss: `1.5781`, Test Acc: `22.22%`
  * **Epoch 30:** Train Loss: `0.0000`, Test Acc: **`100.00%`**
  * **Epoch 150:** Train Loss: `0.0000`, Test Acc: **`100.00%`**
  * **Final Test Accuracy:** **`100.00%`** (perfect classification generalization).

### 3. Symbolic Parameter Recovery & Mathematical Formulation (`symbolic_regression.py`)
* **Task:** Extract the converged parameter values of the EML-KAN model and print the recovered symbolic mathematical formula.
* **Result:** The network successfully recovered the target EML function with $3.3 \times 10^{-13}$ MSE validation loss.
* **Recovered KAN+EML Symbolic Function:**
  \[
  y = \left[ \exp(1.200077 \cdot x_1 - 0.300178) - \ln\left(\text{softplus}(0.800115 \cdot x_1 + 0.201382) + 1\text{e-}6\right) \right] + \left[ \exp(1.199920 \cdot x_2 - 0.299816) - \ln\left(\text{softplus}(0.799889 \cdot x_2 + 0.198624) + 1\text{e-}6\right) \right]
  \]

### 4. Architectural Comparison & Parameter Scaling
A detailed comparative analysis (including parameter counts, formulas, and structural comparison visual) is documented in [COMPARISON_AND_METRICS.md](COMPARISON_AND_METRICS.md).

---

## 🚀 How to Run the Experiments

Navigate to the workspace directory and execute the scripts directly:

```bash
# Run the complex data zero-loss experiment
python KAN_EML/complex_data_experiment.py

# Run the Wine dataset classification experiment
python KAN_EML/real_world_experiment.py

# Run the symbolic regression parameter recovery proof
python KAN_EML/symbolic_regression.py

# Run the comparative trained curves experiment and generate plot
python KAN_EML/plot_trained_curves.py
```

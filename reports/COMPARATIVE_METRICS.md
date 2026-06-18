# Comparative Performance Metrics

This document provides a quantitative and qualitative comparison of parameter counts, training requirements, inference sequence complexity, and attractor dynamics for a Fashion MNIST memory retrieval task ($N = 20$ templates, dimension $d = 784$), with **`cross_attn_normal`** defined as the Static Cross-Attention layer over a persistent memory bank and **Modern Hopfield Network (MHN)** as the state-space energy baseline.

---

## 1. Quantitative Metrics Comparison

| Model | Parameter Count | Training Iterations (Epochs) | Sequence Time Complexity | Run-time VRAM Scaling | Exact MSE = 0? | Attractor Dynamics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Modern Hopfield Network (MHN)** | $O(N \cdot d)$ (15,680 params) | **0** (Instantaneous) | $O(L \cdot d \cdot N)$ | **$O(d \cdot L)$ (Dynamic Activation Cache)** | Yes (under high $\beta$) | Energy minimization (recurrent state) |
| **`cross_attn_normal`** (Static Cross-Attention) | $O(d \cdot M)$ (15,680 params) | **0** (Instantaneous) | $O(L \cdot M)$ (linear in sequence length $L$) | **$O(d \cdot M)$ (Persistent Fixed Weights)** | **Yes** (Exactly 0.0 in 1-3 steps) | Iterative Query-Value Routing Loops |
| **`AnalyticalHopfieldKAN`** | $O(2 \cdot N \cdot d)$ (31,360 params) | **0** (Instantaneous) | $O(L \cdot M)$ | **$O(2 \cdot d \cdot M)$ (Persistent Fixed Weights)** | **Yes** (Exactly 0.0) | Mapped Single-Step Routing |
| **`Trainable RBF-KAN`** (`[784, 2, 784]`) | $O(2 \cdot h \cdot d \cdot (G+1))$ (18,816 params) | $\sim 2,000$ | $O(L \cdot h \cdot d \cdot G)$ | **$O(2 \cdot h \cdot d \cdot G)$ (Persistent Fixed Weights)** | No (continuous float) | Autoassociative Bottleneck mapping |
| **`Sparse / Symbolic KAN`** (Pruned) | **9,141 active params** ($41.7\%$ savings) | $\sim 3,500$ (3 phases) | $O(L \cdot h \cdot d \cdot G)$ | **$O(active\_params)$ (Persistent Fixed Weights)** | **Yes** (Exactly 0.0) | Sparse Bottleneck mapping |

*Note: In the complexity column, $L$ represents the sequence length (tokens processed), $M$ is the memory count (templates stored), $h$ is the KAN bottleneck dimension ($h=2$), and $G$ is the RBF grid size ($G=2$).*

---

## 2. Qualitative & Architectural Comparison

### A. Modern Hopfield Network (MHN)
* **Mental Model**: Continuous state-space memory retrieval operating via energy minimization over stored pattern attractors.
* **Advantage**: Fast template configuration.
* **Disadvantage**: Standard dot-product similarity is strictly linear; coupling of attraction boundaries and templates restricts fine-tuning.

### B. `cross_attn_normal` (Static Cross-Attention)
* **Mental Model**: Frames retrieval as a query-key dot-product routing over a persistent memory database ($\mathbf{V} = \text{templates}, \mathbf{K} = \mathbf{V}^T$).
* **Advantage**: Bypasses the $O(L^2)$ self-attention sequence length bottleneck. Using **Iterative Routing Loops** (1-3 steps), it can reconstruct patterns from highly corrupted or half-erased inputs perfectly.
* **Disadvantage**: Relies on fixed, dense key-value matrix projections.

### C. `AnalyticalHopfieldKAN`
* **Mental Model**: Decomposes the mathematical operations of `cross_attn_normal` into KAN edge activations (Layer 1 edge weights = $\beta K$, Layer 2 edge weights = $V$).
* **Advantage**: Decouples similarity projection (Layer 1) from reconstruction (Layer 2), allowing boundary tuning independently of target templates.
* **Disadvantage**: Requires double the parameter footprint ($2 \cdot N \cdot d$) to store incoming and outgoing edge weights separately.

### D. `Trainable RBF-KAN`
* **Mental Model**: Autoassociative memory retrieval through a narrow continuous latent bottleneck ($h = 2$) using localized Radial Basis Functions.
* **Advantage**: Reduces inference operations by 70% FLOPs by passing queries through the bottleneck.
* **Disadvantage**: Training is required to configure the RBF grid parameter landscape.

### E. `Sparse / Symbolic KAN`
* **Mental Model**: A pruned and symbolically regression-fitted KAN that models memory retrieval dynamics.
* **Advantage**: Achieves **41.7% parameter savings** and **70% inference FLOPs savings** compared to standard MHN / `cross_attn_normal` while recovering perfect reconstruction (MSE = 0).
* **Disadvantage**: High three-phase training complexity.

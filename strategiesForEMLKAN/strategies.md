# EML-KAN Structure & Parameter Selection Strategies

This document provides a comprehensive framework and set of strategies for constructing and configuring **Exp-Minus-Log Kolmogorov-Arnold Networks (EML-KAN)** for any data modality.

---

## 1. Core Mathematical Anatomy of EML-KAN

The univariate function $\phi(x)$ on each edge connecting node $j$ to node $i$ is parameterized as:
\[
\phi(x) = w_{\text{base}} \cdot x + \sum_{k=1}^K w_{\text{eml}, k} \cdot \left[ \exp(a_k \cdot x + b_k) - \ln\left(\text{softplus}(c_k \cdot x + d_k) + \epsilon\right) \right]
\]

Here, the parameters to tune and optimize are:
1. **Network Structure**: Depth (layers) and Width (neurons per layer).
2. **Number of EML Components ($K$)**: Primitives per edge.
3. **Initialization Scales** for EML inner parameters ($a, b, c, d$).
4. **Regularization and Training Procedure** (AdamW + L-BFGS, Weight Decay).

---

## 2. Structural Design Guidelines (Layers & Neurons)

Since EML-KAN operates as a Kolmogorov-Arnold representation, the universal approximation property states that any multivariate continuous function can be represented by a 2-layer KAN. However, in practice, different modalities benefit from highly distinct topologies:

### A. Tabular Datasets
- **Rule**: Keep it shallow. Tabular features generally exhibit simpler, monotonic, or additive relationships.
- **Topologies**: Direct mapping `[D_in, D_out]` or a single hidden layer `[D_in, D_hidden, D_out]` where $D_{\text{hidden}} \approx 0.5 \times D_{\text{in}}$.
- **EML Components ($K$)**: $K=1$ is optimal. Larger $K$ increases parameter counts and causes optimization overfitting.
  - *Observation:* On Wine tabular classification, a simple `[13, 3]` network with $K=1$ achieves **100.00%** accuracy, whereas going deeper (`[13, 15, 15, 3]`) degrades accuracy to **94.44%**.

### B. Image Datasets
- **Rule**: Single-layer KANs (`[D_in, D_out]`) are extremely parameter-efficient feature classifiers. If deep architectures are used, **residual connections** (e.g., `ResidualEMLKAN`) are mandatory.
- **Topologies**: `[D_in, D_out]` for direct classification.
- **EML Components ($K$)**: $K=1$ is optimal for classification.
  - *Observation:* On the Digits dataset, `[64, 10]` with $K=1$ achieved **96.67%** classification accuracy using only 3,840 parameters. Deep feedforward networks without skip connections suffered from optimization bottlenecks, yielding poor accuracy (e.g., `[64, 16, 16, 10]` with $K=1$ dropped to **36.11%** due to gradient vanishing/instabilities).

### C. Audio/Waveform Datasets
- **Rule**: Standard EML primitives ($\exp$, $\ln$) are non-periodic. Consequently, fitting high-frequency cyclic signals is difficult for pure KANs without periodic basis components (e.g., sine/cosine activations).
- **Topologies**: If using EML-KAN for audio, wider hidden layers and larger $K$ are required to compose localized transitions.
  - *Observation:* On multi-frequency waveform regression, shallow models scored high MSE values (~0.56-0.62). If modeling wave signals, combine EML layers with periodic activations.

### D. Clean vs. Noisy Function Approximation
- **Clean Functions**:
  - **Rule**: Increase depth and EML components $K$ to capture sharp local transitions.
  - *Observation:* `[3, 3, 1]` with $K=2$ drove MSE down to **`0.1339`** compared to **`0.5055`** on a shallow $K=1$ network.
- **Noisy Functions**:
  - **Rule**: Keep EML components low ($K=1$) and use a deeper network as a low-pass filter to reject noise.
  - *Observation:* On noisy function data (noise $\sigma=0.2$), `[3, 3, 1]` with $K=2$ overfit to the noise (MSE **`0.8857`**), whereas a deeper `[3, 5, 5, 1]` network with $K=1$ filtered the noise, yielding a test MSE of **`0.1210`**.

---

## 3. Parameter Tuning Strategy Summary

| Modality | Depth | Width Ratio (Hidden/Input) | EML components ($K$) | Init Scale ($a, c$) |
| :--- | :--- | :--- | :--- | :--- |
| **Tabular** | 1 (no hidden layer) | N/A | 1 | 0.5 |
| **Image** | 1 (or Residual if deeper) | N/A | 1 | 0.1 (smoother boundaries) |
| **Audio/Waveform**| 2-3 | $2.0 \times$ to $4.0 \times$ | $\ge 2$ | 1.0 (steeper edges) |
| **Clean Function**| 2 | $1.0 \times$ | $\ge 2$ | 1.0 (fast fitting) |
| **Noisy Function**| 3 | $1.5 \times$ | 1 (regularization) | 0.5 (noise rejection) |

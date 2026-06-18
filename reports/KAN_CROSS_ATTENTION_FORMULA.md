# KAN-Wise Summation Formulation of Cross-Attention

This document defines how standard continuous **Cross-Attention** (and the Modern Hopfield Network retrieval equation) is mathematically structured as a two-layer **Kolmogorov-Arnold Network (KAN)** using univariate edge activations and sum-reduction nodes.

---

## 1. Classical Cross-Attention Retrieval Formula
In a Modern Hopfield Network (MHN) or cross-attention memory bank, a query vector $Q \in \mathbb{R}^d$ retrieves a reconstructed output $y \in \mathbb{R}^d$ from a memory database of $M$ stored key-value templates $K \in \mathbb{R}^{M \times d}$ and $V \in \mathbb{R}^{d \times M}$ (where typically $V = K^T$).

The classical retrieval equation is:
\[
y_i = \sum_{j=1}^{M} V_{ij} \cdot \text{Softmax}_j \left( \beta \sum_{k=1}^d Q_k K_{jk} \right)
\]
where $\beta$ is the inverse temperature scaling parameter, and:
\[
\text{Softmax}_j (z) = \frac{\exp(z_j)}{\sum_{l=1}^M \exp(z_l)}
\]

---

## 2. KAN-Wise Summation Mapping

A Kolmogorov-Arnold Network layer maps inputs $x$ to outputs $y$ by applying univariate activation functions $\phi$ along its edges and summing them at the output nodes:
\[
y_i = \sum_{j=1}^{n_{\text{in}}} \phi_{i,j}(x_j)
\]
For a two-layer KAN `[d, M, d]` with input features $Q \in \mathbb{R}^d$, hidden memory nodes $h \in \mathbb{R}^M$, and output features $y \in \mathbb{R}^d$, the network computes:
\[
y_i = \sum_{j=1}^M \Phi^{(2)}_{i,j} \left( \text{Act}_j \left( \sum_{k=1}^d \Phi^{(1)}_{j,k}(Q_k) \right) \right)
\]

By mapping the cross-attention equations to this structure, we define the edge functions $\Phi$ and node activations $\text{Act}$ as follows:

### Step 1: Input-to-Hidden Layer (Edge Summation)
The first layer projects the query $Q$ onto the key templates $K$.
* **KAN Edge Activation:** We define the univariate function on the edge from input $k$ to hidden node $j$ as a linear template projection:
  \[
  \Phi^{(1)}_{j,k}(x) = K_{jk} \cdot x
  \]
* **Node Summation:** Summing these edge activations yields the projection score $h_j$:
  \[
  h_j = \sum_{k=1}^d \Phi^{(1)}_{j,k}(Q_k) = \sum_{k=1}^d K_{jk} Q_k
  \]

### Step 2: Hidden Node Activation (Softmax Routing)
To represent the global Softmax normalizing factor within a KAN activation framework, we use the Log-Sum-Exp (LSE) shift:
\[
\text{Softmax}_j(\beta h) = \exp \left( \beta h_j - \text{LSE}(\beta h) \right)
\]
where $\text{LSE}(\beta h) = \ln \sum_{l=1}^M \exp(\beta h_l)$.
* **KAN Hidden Activation:** We define the hidden node activation as:
  \[
  \text{Act}_j(u) = \exp(u)
  \]
  applied to the LSE-normalized score $u_j = \beta h_j - \text{LSE}(\beta h)$.

### Step 3: Hidden-to-Output Layer (Value Reconstruction)
The second KAN layer aggregates the value templates $V$ weighted by the attention routing coefficients.
* **KAN Edge Activation:** We define the univariate function on the edge from hidden node $j$ to output feature $i$ as a linear value projection:
  \[
  \Phi^{(2)}_{i,j}(z) = V_{ij} \cdot z
  \]
* **Node Summation:** Summing these edge activations produces the final retrieved feature $y_i$:
  \[
  y_i = \sum_{j=1}^M \Phi^{(2)}_{i,j} \left( \text{Act}_j(u_j) \right) = \sum_{j=1}^M V_{ij} \cdot \exp\left( \beta h_j - \text{LSE}(\beta h) \right)
  \]

---

## 3. Combined KAN Summation Formula

Substituting all components together, the entire cross-attention mechanism is represented as a pure KAN summation of univariate transformations:

\[
y_i = \sum_{j=1}^{M} \phi^{\text{value}}_{i,j} \left( \psi^{\text{softmax}}_j \left( \sum_{k=1}^d \phi^{\text{key}}_{j,k}(Q_k) \right) \right)
\]

Where:
* **Key-matching univariate edge:** $\phi^{\text{key}}_{j,k}(x) = K_{jk} \cdot x$
* **Softmax normalizer shift:** $\psi^{\text{softmax}}_j(h_j) = \exp\left(\beta h_j - \ln \sum_{l=1}^M \exp(\beta h_l)\right)$
* **Value-retrieval univariate edge:** $\phi^{\text{value}}_{i,j}(z) = V_{ij} \cdot z$

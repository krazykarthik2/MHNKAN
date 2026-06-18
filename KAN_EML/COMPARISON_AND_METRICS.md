# Standard KAN vs. EML-KAN Architectural Comparison

This report evaluates and compares **Standard grid-based KAN** (utilizing Radial Basis Functions / B-splines) and **EML-KAN** (utilizing the Exp-Minus-Log analytic operator).

---

## 📊 Comparison Summary Table

| Metric / Dimension | Standard KAN (B-Splines / RBFs) | EML-KAN |
| :--- | :--- | :--- |
| **Edge Activation Mechanism** | Linear combination of SiLU residual + $G$ grid RBF splines | Mixture of SiLU base + $K$ universal EML primitives |
| **Grid Dependability** | Highly dependent. Requires pre-defining grid limits $[-L, L]$. | **Grid-free**. Uses smooth continuous functions on $(-\infty, \infty)$. |
| **Inference Extrapolation** | Drops to zero or diverges outside the training grid. | Fits natural physical boundaries via $\exp$ and $\ln$ asymptotics. |
| **Parameter Complexity** | Scales linearly with grid size $G$: **$\mathcal{O}(G)$** | Fixed size per EML primitive: **$\mathcal{O}(K)$** (usually $K \le 2$) |
| **Active Parameter Count (Per Edge)** | For $G = 12$ RBF grid: **13 parameters** | For $K = 1$ EML component: **5 parameters** |
| **Symbolic Regressibility** | Requires post-training genetic/curve-fit algorithms. | **Native**. Weight variables are the direct function coefficients. |

---

## 🧮 Parameter Scaling Analysis (Per Layer)

For an input dimension $D_{\text{in}}$ and output dimension $D_{\text{out}}$:

### 1. Standard RBF-KAN Layer Parameter Count
Each edge contains $1$ base linear weight and $G$ Radial Basis Function weights.
\[
P_{\text{Standard}} = D_{\text{out}} \times D_{\text{in}} \times (G + 1)
\]
* **Example ($13 \rightarrow 8$ layer, $G=12$):**
  \[
  P = 8 \times 13 \times 13 = \mathbf{1,352 \text{ parameters}}
  \]

### 2. EML-KAN Layer Parameter Count
Each edge contains $1$ base linear weight, and for each of the $K$ EML components, it has $1$ mixture weight and $4$ function coefficients ($a, b, c, d$).
\[
P_{\text{EML-KAN}} = D_{\text{out}} \times D_{\text{in}} \times (5K + 1)
\]
* **Example ($13 \rightarrow 8$ layer, $K=1$):**
  \[
  P = 8 \times 13 \times 6 = \mathbf{624 \text{ parameters}} \quad \text{(\textbf{53.8\% Parameter Reduction})}
  \]

---

## 📐 Recovered Symbolic Mathematical Functions

In our symbolic regression tests ($X \in \mathbb{R}^2 \rightarrow y \in \mathbb{R}^1$), we set the target mathematical function to a sum of EML operators:
\[
y = \sum_{j=1}^2 \left[ \exp(1.2 \cdot x_j - 0.3) - \ln\left(\text{softplus}(0.8 \cdot x_j + 0.2) + 1\text{e-}6\right) \right]
\]

The trained EML-KAN model reconstructed the following function with **validation MSE loss = `3.3e-13` (mathematically zero)**:
\[
y_{\text{learned}} = \left[ \exp(1.200077 \cdot x_1 - 0.300178) - \ln\left(\text{softplus}(0.800115 \cdot x_1 + 0.201382) + 1\text{e-}6\right) \right] + \left[ \exp(1.199920 \cdot x_2 - 0.299816) - \ln\left(\text{softplus}(0.799889 \cdot x_2 + 0.198624) + 1\text{e-}6\right) \right]
\]

---

## 🖼️ Architectural Schematic Visuals

### 1. Connection Edge Detail Comparison
The graphic below highlights the structural difference on each connection edge between a Standard KAN (grid splines) and the EML-KAN (analytic primitive):

![Architecture Comparison](kan_comparison.png)

### 2. Full Network Node & Edge Curve Graph Layout
Below is the complete network representation ($[2, 3, 1]$ structure) illustrating how the univariate function curves reside directly on the connecting edges:

![Network Nodes and Curves Comparison](kan_network_edges_comparison.png)

### 3. Empirical Comparison of Actually Trained Edge Curves
Below is the plot showing the **exact real values** of the univariate curves learned on the edges of the networks after training on the target function:

![Trained Edge Curves Comparison](trained_curves_comparison.png)

---

## 🔬 Complex Function Approximation Comparison (18-Edge `[2, 6, 1]` Network)

* **Target Function:** 
  \[
  y = \exp(\sin(x_1) - \cos(x_2)) - \ln\left(\text{softplus}(x_1 \cdot x_2) + 0.1\right)
  \]
* **Standard RBF-KAN (`[2, 6, 1]`, $G=10$) Final Loss:** **`0.000004601337`**
* **EML-KAN (`[2, 6, 1]`, $K=4$) Final Loss:** **`0.016432942823`**

### Learned EML-KAN Hidden-to-Output Parameter Grid ($h_j \rightarrow y_1$)
The final output is computed by summing the $K=4$ EML components over all 6 hidden nodes:
\[
y_1 = \sum_{j=1}^6 \sum_{k=1}^4 \left[ \exp\left(a_{j,k} h_j + b_{j,k}\right) - \ln\left(\text{softplus}\left(c_{j,k} h_j + d_{j,k}\right) + 1\text{e-}6\right) \right]
\]

The exact learned parameters ($a, b, c, d$) for each connection edge are:

* **Hidden $h_1 \rightarrow y_1$:**
  * Component 1: $a=-0.4765, b=-0.3181, c=-0.1863, d=0.8034$
  * Component 2: $a=-1.2522, b=-0.3267, c=-0.0395, d=0.6094$
  * Component 3: $a=0.0187, b=-0.2391, c=0.3049, d=0.5911$
  * Component 4: $a=-0.0232, b=-0.1628, c=1.0121, d=0.7377$
* **Hidden $h_2 \rightarrow y_1$:**
  * Component 1: $a=-0.1709, b=-0.1528, c=0.1819, d=0.6530$
  * Component 2: $a=-0.4083, b=-0.4516, c=0.7302, d=0.6918$
  * Component 3: $a=0.0065, b=-0.1368, c=0.1581, d=0.4073$
  * Component 4: $a=-0.7711, b=-0.3793, c=0.4471, d=0.5268$
* **Hidden $h_3 \rightarrow y_1$:**
  * Component 1: $a=-0.1646, b=-0.3487, c=0.9612, d=0.6131$
  * Component 2: $a=-0.5681, b=-0.4009, c=0.4225, d=0.4127$
  * Component 3: $a=-0.6975, b=-0.5267, c=0.0530, d=0.7272$
  * Component 4: $a=-1.0396, b=-0.3975, c=0.1606, d=0.3935$
* **Hidden $h_4 \rightarrow y_1$:**
  * Component 1: $a=-0.2509, b=-0.2651, c=0.7548, d=0.6747$
  * Component 2: $a=-0.3927, b=-0.4204, c=0.0245, d=0.5729$
  * Component 3: $a=-0.3653, b=-0.4338, c=0.2191, d=0.3286$
  * Component 4: $a=-0.0217, b=-0.2802, c=0.5887, d=0.4908$
* **Hidden $h_5 \rightarrow y_1$:**
  * Component 1: $a=0.1790, b=-0.0725, c=0.2416, d=0.6094$
  * Component 2: $a=0.0484, b=-0.2329, c=0.3383, d=0.4909$
  * Component 3: $a=-0.1189, b=-0.2849, c=0.2132, d=0.6421$
  * Component 4: $a=0.1534, b=-0.1981, c=0.1201, d=0.6492$
* **Hidden $h_6 \rightarrow y_1$:**
  * Component 1: $a=0.3517, b=0.1285, c=0.8153, d=0.8815$
  * Component 2: $a=0.0738, b=-0.2183, c=1.0096, d=0.8416$
  * Component 3: $a=0.0904, b=-0.3187, c=0.4461, d=0.5334$
  * Component 4: $a=-0.2816, b=-0.2887, c=1.4364, d=0.9713$

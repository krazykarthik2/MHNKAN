# Scaling Laws and Complexity Analysis

This document outlines the parameter scalability, space complexity, time complexity, and scalability limits of the **Radial Basis Function Kolmogorov-Arnold Network (RBF-KAN)**, the standard **Modern Hopfield Network (MHN)**, and the **Analytical Hopfield-KAN**.

---

## 1. Parameter Complexity Equations

Let:
* $D_{in}, D_{out}$: Input and output dimension of a layer.
* $G$: Grid size (number of RBF basis centers).
* $d$: Image / feature dimension.
* $N$: Number of stored prototype patterns (memories).

| Model | Parameter Count Equation | Parameter Count for Fashion MNIST ($d=784, N=5, G=15$) |
| :--- | :--- | :--- |
| **Standard MHN** | $N \times d$ (stored templates matrix) | $5 \times 784 = \mathbf{3,920}$ |
| **RBF-KAN Layer** | $D_{out} \times D_{in} \times (G + 1)$ (Base + RBF weights) | $784 \times 784 \times 16 \approx \mathbf{9.85\times 10^6}$ |
| **Analytical Hopfield-KAN** | $2 \times N \times d$ (Linear edge scales for input/output layers) | $2 \times 5 \times 784 = \mathbf{7,840}$ |

---

## 2. Computational and Memory Scaling Laws

### Time Complexity (Forward Pass)
* **Standard MHN**: $O(N \cdot d)$ (matrix multiplications for similarity projection and output reconstruction).
* **RBF-KAN Layer**: $O(D_{in} \cdot D_{out} \cdot G)$ (evaluating $G$ RBF kernels per input-output edge pair).
* **Analytical Hopfield-KAN**: $O(N \cdot d)$ (identical to standard MHN, as it maps directly onto the same steps).

### Space Complexity (RAM/VRAM Footprint)
* **Standard MHN**: $O(N \cdot d)$ to store patterns.
* **RBF-KAN Layer**: $O(D_{in} \cdot D_{out} \cdot G)$ to store the RBF weight tensor.

---

## 3. What Scales

1. **Memory Capacity of Hopfield-KAN**:
   The memory capacity of the Hopfield-KAN scales exponentially with the dimension $d$ (specifically, $C \cong 2^{d/2}$ for random patterns), inheriting the dense associative memory characteristics of Modern Hopfield Networks.
2. **Analytical Equivalence Scaling**:
   The `AnalyticalHopfieldKAN` structure preserves $O(N \cdot d)$ parameters and computes reconstructions with exactly **MSE = 0** unrounded error across arbitrary dimensions (from toy $d=8$ to Fashion MNIST $d=784$).
3. **Inference Latency**:
   Since Hopfield retrieval takes place in a fixed forward-pass sequence (1 layer of similarity KAN + softmax normalization + 1 layer of projection KAN), the latency scales as a constant $O(1)$ depth regardless of query complexity.

---

## 4. What Doesn't Scale

1. **Grid Parameter Explosion in Trainable KANs**:
   As the input dimension ($D_{in}$) and output dimension ($D_{out}$) grow, a fully-connected trainable RBF-KAN suffers from quadratic parameter growth scaled by grid size: $O(D^2 \cdot G)$. For a standard $1024 \times 1024$ resolution image, a single KAN layer with $G=15$ would require $\sim 15.7$ million parameters, making it computationally heavy.
2. **Empirical Training Sample Complexity**:
   To train a raw `RBF-KAN` to map noisy queries to clean memories *empirically*, it requires a large dataset of noise variations to chart the attraction basins. In contrast, the `AnalyticalHopfieldKAN` maps the basins of attraction instantly without training.
3. **Softmax Numerical Stability at Scale**:
   For very large dimensions ($d \gg 10^4$) and large temperature parameters ($\beta \gg 10^3$), the intermediate similarity values can overflow float32 limits if not stabilized (fixed in our implementation using stabilized subtraction: $S - S_{max}$).

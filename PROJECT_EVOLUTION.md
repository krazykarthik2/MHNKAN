# Chronological Evolution of the MHNKAN Project

This document provides a comprehensive, file-by-file, and phase-by-phase account of the evolution of the **Modern Hopfield Network & Kolmogorov-Arnold Network (MHNKAN)** codebase. It details the terminologies, architectural iterations, improvements, experiment results, and comparative metrics of the project.

---

## 1. Core Terminology & Architectural Concepts

To understand the evolution of this project, it is essential to establish its foundational terms:

*   **Modern Hopfield Network (MHN)**: A dense continuous associative memory model. Retrieval is governed by continuous state-space updates using softmax-based energy minimization:
    \[
    \mathbf{y} = \mathbf{V} \operatorname{softmax}(\beta \mathbf{V}^T \mathbf{x})
    \]
    where \(\mathbf{V}\) represents stored template patterns, \(\mathbf{x}\) is the corrupted query, and \(\beta\) is the inverse temperature (governing attraction boundaries).
*   **Kolmogorov-Arnold Network (KAN)**: A neural network paradigm based on the Kolmogorov-Arnold representation theorem. Unlike standard Multi-Layer Perceptrons (MLPs) which apply activation functions at nodes, KANs place learnable 1D activation functions on the connection edges:
    \[
    y_i = \sum_{j} \phi_{i,j}(x_j)
    \]
*   **Radial Basis Functions (RBFs)**: Gaussian basis functions of the form \(\exp\left(-\frac{(x - c)^2}{2\sigma^2}\right)\) distributed over a grid, used to parameterize the learnable edge activations \(\phi(x)\) in standard KAN implementations.
*   **Analytical Hopfield-KAN**: A custom KAN architecture that mathematically maps standard MHN dynamics directly onto univariate edge activations and sum nodes, achieving exact equivalence (**MSE = 0.0**).
*   **Exp-Minus-Log (EML) Operator**: A universal binary mathematical primitive:
    \[
    \operatorname{eml}(x, y) = \exp(x) - \ln(y)
    \]
    Proven to be functionally complete for continuous mathematics when combined with the constant \(1\), meaning any elementary function (trig, log, division, powers) can be composed solely of EML primitives.
*   **EML-KAN**: An advanced KAN variant where edge activation functions \(\phi(x)\) are parameterized as mixtures of EML operators:
    \[
    \phi(x) = w_{\text{base}} \cdot x + \sum_{k=1}^K w_{\text{eml}, k} \cdot \left[ \exp(a_k \cdot x + b_k) - \ln\left(\text{softplus}(c_k \cdot x + d_k) + \epsilon\right) \right]
    \]
*   **Directed Acyclic Graph (DAG) Compilation**: A code-generation pipeline that exports trained KAN or EML-KAN layers directly into flat, static C++ arrays or PyTorch register operations, eliminating dynamic memory allocations, tensor library dependencies, and division/power calculations.
*   **Genetic Algorithm (GA) Pruning**: A post-training search strategy that zero-out inactive EML-KAN components to optimize sparsity and execution speeds.

---

## 2. Phase-by-Phase Chronological Project Evolution

Based on the chronological file modification order, the codebase evolved through five distinct architectural phases.

### Phase 1: Foundations of MHN & RBF-KAN Integration (Mid-June 2026)
*   **Core Files**: 
    *   [kan_hopfield.py](core/kan_hopfield.py)
    *   [memorize_proof.py](experiments/01_binary_memorization/memorize_proof.py)
    *   [plot_hopfield_kan.py](experiments/01_binary_memorization/plot_hopfield_kan.py)
    *   [research_analysis.md](reports/research_analysis.md)
    *   [SCALING_LAWS.md](reports/SCALING_LAWS.md)
*   **The Problem**: Standard RBF-KANs suffer from grid parameter explosion: a single layer mapping \(D_{in} \to D_{out}\) with grid size \(G\) scales as \(O(D_{in} \cdot D_{out} \cdot G)\), requiring massive memory and dense training datasets to chart attraction basins.
*   **The Innovation**: Creation of `AnalyticalHopfieldKAN` which mathematically decomposes the continuous softmax lookup of MHNs into KAN structures. By setting edge parameters analytically without training, it scales as \(O(2 \cdot N \cdot d)\) where \(N\) is the number of templates and \(d\) is the pattern dimension.
*   **Experiment Results**:
    *   **Equivalence MSE**: Perfect equivalence (**MSE = 0.0000000000000000**) between standard MHN and Analytical KAN.
    *   **Trained RBF-KAN**: A multi-layer KAN `[8, 16, 8]` trained via AdamW on 4 binary patterns of length 8 successfully converged to **MSE = 1.27e-05 (unrounded)** and **0.0 (rounded)** under noisy inputs.

---

### Phase 2: Domain Expansions, Sparsity, & Symbolic Fitting (Mid-June 2026)
*   **Core Files**:
    *   [symbolic_sparse_kan.py](core/symbolic_sparse_kan.py)
    *   [sparse_cross_attn_kan.py](core/sparse_cross_attn_kan.py)
    *   [memorize_proof_sparse.py](experiments/01_binary_memorization/memorize_proof_sparse.py)
    *   [fashion_mnist_experiment.py](experiments/02_fashion_mnist/fashion_mnist_experiment.py)
    *   [basin_proof.py](experiments/03_basin_of_attraction/basin_proof.py)
    *   [continuous_real_exact.py](experiments/05_continuous_real_valued/continuous_real_exact.py)
    *   [genomic_memory_proof.py](experiments/06_genomic_gue/genomic_memory_proof.py)
*   **The Innovations & Improvements**:
    1.  **L1 Sparsity Pruning & Symbolic Regression**: Designed a three-phase pipeline using L1 regularization to prune inactive edges and scipy curve-fitting to map remaining edges to symbolic formulas (e.g. \(x, x^2, \exp(x)\)), freezing them to save parameters.
    2.  **Hybrid Sparse Cross-Attention KAN**: Combined the parameter efficiency of KAN edge-expansion with the sequence length scaling of cross-attention (`cross_attn_normal`), bypassing the \(O(L^2)\) attention bottleneck and scaling as \(O(L \cdot M)\).
    3.  **Real-Valued Exact Memorization**: Proved that by scaling the inverse temperature parameter \(\beta \to 10^5\) (winner-take-all limit), the KAN-Hopfield network achieves **MSE = 0.0** on continuous, non-binarized floating-point templates under 40% random erasure and \(\sigma = 0.6\) Gaussian noise.
    4.  **Genomic Restoration (GUE)**: Validated the framework on HF `leannmlindsey/GUE` promoter sequences. The Analytical Hopfield-KAN achieved a **100.00% nucleotide base recovery rate** (1400/1400 nucleotides) under 25% random base mutations and 30% contiguous deletions.
*   **Results & Comparison**:
    *   **Fashion MNIST (N=20, d=784)**: Standard MHN requires 15,680 parameters. The **Sparse KAN** achieved perfect reconstruction using only **9,141 non-zero active parameters (41.70% parameter savings)** and a **70% reduction in inference FLOPs**.
    *   **Basin of Attraction isolation**: Interpolating between template \(A\) and \(B\) mapped a perfect phase transition boundary. At \(\alpha = 0.49\), retrieval locked onto Pattern B (distance = 0.0); at \(\alpha = 0.51\), retrieval locked onto Pattern A (distance = 0.0).

---

### Phase 3: The EML-KAN Breakthrough (Late-June 2026)
*   **Core Files**:
    *   [eml_network.py](KAN_EML/eml_network.py)
    *   [complex_data_experiment.py](KAN_EML/complex_data_experiment.py)
    *   [real_world_experiment.py](KAN_EML/real_world_experiment.py)
    *   [symbolic_regression.py](KAN_EML/symbolic_regression.py)
    *   [fit_image_experiment.py](KAN_EML/fit_image_experiment.py)
*   **The Innovation**: Replaced Splines and RBFs with the **Universal Exp-Minus-Log (EML) operator**. The edge functions are parameterized as mixtures of EML transformations, allowing the KAN to organically learn trigonometric, logarithmic, and polynomial functions.
*   **Results**:
    *   **Complex Target Function Recovery**: Trained on \(y = \sum_{j=1}^2 \operatorname{eml}(1.2 x_j - 0.3, \text{softplus}(0.8 x_j + 0.2) + 1\text{e-}6)\) using staged AdamW + double-precision L-BFGS line search, EML-KAN drove loss to absolute machine precision (**Final Loss = 2.966e-13**).
    *   **Tabular wine Classification**: Trained an `EMLKAN([13, 8, 3])` network. Achieved **100.00% Test Accuracy** within 30 epochs.
    *   **Portrait coordinate regression**: Parameterized coordinate-to-color mapping \(f(x, y) \to (R, G, B)\) to reconstruct a stylized Rick Astley portrait (`rick_roll_reconstructed.png`), reaching a final reconstruction MSE of **0.03598**.
*   **Design & Strategy Rules**:
    *   *Tabular features*: Deep networks overfit. Optimal configuration is shallow (`[D_in, D_out]`) with \(K=1\) component.
    *   *High-frequency waveforms*: EML functions are non-periodic. Wider architectures combined with external periodic basis layers are required.
    *   *Noisy data*: Higher \(K\) overfits. Keeping \(K=1\) and using deeper hidden networks acting as low-pass filters isolates underlying signals (MSE 0.12 vs 0.88).

---

### Phase 4: Symbolic DAG Compilation & Genetic Optimization (Early July 2026)
*   **Core Files**:
    *   [eml_symbolic_optimizer.py](strategiesForEMLKAN/eml_symbolic_optimizer.py)
    *   [eml_dag_optimizer.py](strategiesForEMLKAN/eml_dag_optimizer.py)
*   **The Innovation**: Developed a rule-based `EMLSymbolicOptimizer` to simplify nested expressions (exponential combinations, softplus cleanups, inversions, identity cleanup) and a division-free `EMLDAGOptimizer` that compiles network paths directly to CPU register operations.
*   **Results**:
    *   **Optimized Dense DAGs**: Replaced costly exponentiation and division routines with additive log-domain projections, resulting in significant execution speedups.
    *   **Genetically Optimized DAGs**: A GA searching for optimal sparse pathways achieved up to **3.39x CPU speedups** and up to **75.0% sparsity rates** across various target functions without degrading the MSE fitting accuracy (MSE remains between \(10^{-4}\) and \(10^{-6}\)).

---

### Phase 5: Hybrid ESP32 Edge Deployment (July 2026)
*   **Core Files**:
    *   [train_cifar100_esp32.py](large_scale_experiment/train_cifar100_esp32.py)
    *   [esp32_cifar100_inference.h](large_scale_experiment/esp32_project/esp32_cifar100_inference.h)
*   **The Innovations & Edge Constraints**:
    *   **Hybrid Feature-Classifier Pipeline**: ESP32 microcontrollers are constrained to 520 KB RAM and 4MB-8MB Flash, making standard 5,000,000 parameter baseline CNN classifiers (MobileNet / ResNet) impossible to execute standalone on the chip. To solve this, a hybrid setup was deployed:
        *   **Host PC (Laptop)**: Runs the heavy pre-trained MobileNetV3 backbone to extract a 576-dimensional feature vector from input CIFAR-100 images.
        *   **ESP32 Edge Chip**: The 576 extracted features are transmitted to the ESP32 via USB serial, which executes the compiled EML-KAN classification head (`576 -> 100` classes) using register-level division-free static calculations.
    *   **Memory footprints**: The EML-KAN classification head fits inside **70.3 KB to 234.4 KB Flash** storage and requires **< 10 KB runtime execution RAM** (executing directly on CPU registers).
    *   **Accuracy Results**: The hybrid network consistently converged to **97.00% Training Accuracy** and **79.09% Test Accuracy** on CIFAR-100.
    *   **Remarkable Edge Speedup**: The classification head inference on the ESP32 CPU showed remarkable execution efficiency, completing in an average of **9.91 milliseconds** (9,915 microseconds) per pass.

---

## 3. Under Development (LLaMA & POC Scaling)
*   **Core Files**:
    *   [most_optimized_llm.py](large_scale_experiment/most_optimized_llm.py)
    *   [most_optimized_llm_dag.py](large_scale_experiment/most_optimized_llm_dag.py)
    *   [train.py](train.py)
*   **Current Status**: Under Development. The integration of EML-KAN projections within Grouped-Query Attention (GQA) blocks (Query, Key, Value, Output projections) and Feed-Forward Networks (FFN) to replace traditional dense linear weights is established architecturally. However, because these large-scale systems (like the `EMLKANLLaMA` POC) require extensive resources, they cannot be compared or evaluated quantitatively without undergoing full pre-training. Vectorized compiler outputs (`most_optimized_llm_dag.py`) are structured for 50% magnitude-pruned FFN acceleration once training concludes.

---

## 4. Comparative Performance & Validation Matrix

The quantitative performance, parameter sizes, and computational speedups across the project's completed milestones are summarized below:

| Model Architecture | Task / Domain | Parameter Count | Sparsity / Compression | Accuracies (Train / Test) | Retrieval MSE | Latency Speedup / Execution Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard MHN** | Fashion MNIST (N=20, d=784) | 15,680 | 0.00% (Baseline) | N/A | 0.000000 (Binarized) | 1.00x |
| **Analytical Hopfield-KAN** | Fashion MNIST (N=20, d=784) | 31,360 | 0.00% (Dense) | N/A | **0.000000** (Unrounded float32) | 1.00x |
| **Sparse / Symbolic KAN** | Fashion MNIST (N=20, d=784) | **9,141** | **41.70% parameters saved** | N/A | 0.000000 (Binarized) | **3.3x FLOPs reduction** |
| **PyTorch EML-KAN** | Wine Classification (13 features) | 139 | 0.00% | 100.00% / 100.00% | 0.000000 | 1.00x |
| **Optimized EML-KAN DAG** | Algebraic regression sweeps | 24 | 0.00% (Dense) | N/A | 2.74e-04 | 1.21x speedup |
| **Genetically Optimized DAG** | Algebraic regression sweeps | **17** | **29.17% to 75.00% sparse** | N/A | 2.74e-04 | **1.38x to 3.39x speedup** |
| **EML-KAN head (ESP32)** | CIFAR-100 features classification | **17,580** | **39.4x compression on head** | **97.00% / 79.09%** | N/A | **9.91 ms (ESP32 execution time)** |

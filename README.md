# MHNKAN: Modern Hopfield Network & Kolmogorov-Arnold Network Integration

This repository implements, evaluates, and proves the integration of **Modern Hopfield Networks (MHNs)** with **Kolmogorov-Arnold Networks (KANs)**. We demonstrate that KAN structures utilizing Radial Basis Functions (RBFs) can perform associative memory storage and retrieval, achieving exactly **unrounded MSE = 0.0** pattern reconstruction under severe noise and corruption conditions.

---

## 📂 Repository Structure

The project is structured experiment-wise to keep the code, generated visual proofs, and reports grouped together:

```text
/
├── core/                                     # Core Model architectures
│   ├── kan_hopfield.py                       # SiLU+RBF layers, MHN, and Analytical KAN mapping
│   ├── symbolic_sparse_kan.py                # L1 Sparsity pruning & symbolic regression fitting
│   └── sparse_cross_attn_kan.py              # Pruned KAN hybrid with cross-attention
│
├── experiments/
│   ├── 01_binary_memorization/               # Binary template memorization tests
│   │   ├── memorize_proof.py                 # RBF-KAN binarized memorization proof
│   │   ├── memorize_proof_sparse.py          # L1 pruned model parameter savings proof
│   │   ├── symbolic_exact_solve.py           # SymPy 50-digit high-precision symbolic validation
│   │   ├── plot_hopfield_kan.py              # Generates Analytical KAN computational graph
│   │   ├── plot_kan.py                       # Generates general KAN structure plot
│   │   ├── MEMORIZATION_PROOF.md             # Sparse memorization results
│   │   ├── SYMBOLIC_EXACT_SOLVE.md           # Symbolic solving details
│   │   ├── kan_network_visualization.png     # General KAN visual layout
│   │   └── hopfield_kan_exact_graph.png      # Exact Analytical Hopfield KAN computational graph
│   │
│   ├── 02_fashion_mnist/                     # Fashion MNIST retrieval experiments
│   │   ├── fashion_mnist_experiment.py       # RBF KAN training on Fashion MNIST patterns
│   │   ├── FASHION_MNIST_RECONSTRUCTION.md   # Retrieval MSE and parameter reports
│   │   └── fashion_mnist_reconstruction.png  # Image reconstruction visualization
│   │
│   ├── 03_basin_of_attraction/               # Isolation & energy boundary proofs
│   │   ├── basin_proof.py                    # Interpolates queries between template basins
│   │   ├── BASIN_OF_ATTRACTION.md            # Step phase transition mathematical analysis
│   │   └── basin_attraction.png              # Basin boundary phase transition plot
│   │
│   ├── 04_hybrid_sparse_cross_attn/          # Sparse KAN and cross-attention hybrid
│   │   ├── PROOF_and_ALL_comparison.md       # Sparsity vs. normal cross-attention comparison
│   │   └── sparse_cross_attn_reconstruction.png # Hybrid network inpainting results
│   │
│   ├── 05_continuous_real_valued/            # Exact continuous template retrieval
│   │   ├── continuous_real_exact.py          # Validation script for float32 exact matching
│   │   ├── plot_continuous_real.py           # Real-valued image corruption reconstruction
│   │   ├── REAL_VALUED_PROOF.md              # Mathematical proof of MSE = 0.0 on float32
│   │   └── continuous_real_reconstruction.png # Visual proof under noise std=0.6, 40% erasure
│   │
│   └── 06_genomic_gue/                       # Genome sequence recovery (GUE Dataset)
│       ├── genomic_memory_proof.py           # Genomic error correction and segment inpainting script
│       ├── GENOMIC_RECONSTRUCTION_PROOF.md   # Base recovery rate & alignment metrics
│       └── genomic_reconstruction.png        # Nucleotide sequence reconstruction alignment plot
│
├── reports/                                  # Project reports & analyses
│   ├── COMPARATIVE_METRICS.md                # FLOPs, parameters, and time complexity table
│   ├── SCALING_LAWS.md                       # Scaling behavior with respect to dimension and memory count
│   ├── cross_attention_normal.md             # Sequence length scaling logic of MHNs
│   ├── research_analysis.md                  # Pre-implementation design analysis
│   └── walkthrough.md                        # Master walkthrough of all experiments
│
└── README.md                                 # Project homepage and documentation index
```

---

## 🧠 Model Architectures (`core/`)

1. **`RBFKANLayer`** ([kan_hopfield.py](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/core/kan_hopfield.py)):
   * Combined parallel SiLU residual path and trainable Radial Basis Function (RBF) grid mappings on the network edges.
2. **`AnalyticalHopfieldKAN`** ([kan_hopfield.py](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/core/kan_hopfield.py)):
   * A KAN mapping standard Modern Hopfield continuous attention querying onto logarithmic exp-sum-exp univariate activations, achieving exact memory retrieval (unrounded MSE = 0.0) at the $\beta \to \infty$ limit.
3. **`SymbolicEdge` & Sparse fitting** ([symbolic_sparse_kan.py](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/core/symbolic_sparse_kan.py)):
   * A workflow to prune KAN edges using L1 regularization, select active paths, fit symbolic formulas (linear, quadratic, exponential) via Scipy curve-fitting, and freeze them to eliminate learning overhead.
4. **`SparseCrossAttentionKAN`** ([sparse_cross_attn_kan.py](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/core/sparse_cross_attn_kan.py)):
   * A hybrid combining the parameters of KAN edge expansions with the sequence length efficiency of cross-attention (`cross_attn_normal`), reducing active parameters and scaling linearly as $O(L \cdot M)$.

---

## 🔬 Key Experiments & Proofs

### 1. Genomic Understanding Evaluation (GUE) Proof
* **Script:** [genomic_memory_proof.py](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/experiments/06_genomic_gue/genomic_memory_proof.py)
* **Dataset:** HF `leannmlindsey/GUE` (`prom_core_all` train split)
* **Reconstruction MSE:** `0.0000000000000000` (perfect unrounded float32 retrieval)
* **Nucleotide Recovery Rate:** `100.00%` (1400/1400 bases recovered perfectly) under 25% random base mutations and 30% segment deletions.
* **Details:** [GENOMIC_RECONSTRUCTION_PROOF.md](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/experiments/06_genomic_gue/GENOMIC_RECONSTRUCTION_PROOF.md)

### 2. Continuous Real-Valued Template Reconstruction
* **Script:** [plot_continuous_real.py](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/experiments/05_continuous_real_valued/plot_continuous_real.py)
* **Reconstruction MSE:** `0.0000000000000000` (Perfect unrounded retrieval on continuous Fashion MNIST classes) under $\sigma=0.6$ noise and 40% pixel erasure.
* **Details:** [REAL_VALUED_PROOF.md](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/experiments/05_continuous_real_valued/REAL_VALUED_PROOF.md)

### 3. Memorization Sparsity Proof
* **Script:** [memorize_proof_sparse.py](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/experiments/01_binary_memorization/memorize_proof_sparse.py)
* **Parameter Savings:** **41.7% active parameter reduction** (9,141 vs 15,680) compared to standard MHN while maintaining exact recall.
* **Details:** [MEMORIZATION_PROOF.md](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/experiments/01_binary_memorization/MEMORIZATION_PROOF.md)

### 4. Basin of Attraction Interpolation Phase Transition
* **Script:** [basin_proof.py](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/experiments/03_basin_of_attraction/basin_proof.py)
* **Behavior:** Proves that KAN-Hopfield stores stable, isolated attractor basins. Query interpolation between two targets exhibits an exact step-function phase transition at $\alpha = 0.5$.
* **Details:** [BASIN_OF_ATTRACTION.md](file:///C:/Users/karthikkrazy/Documents/antigravity/busy-einstein/experiments/03_basin_of_attraction/BASIN_OF_ATTRACTION.md)

---

## 🚀 How to Run the Experiments

Set up your python environment (requires `torch`, `matplotlib`, `numpy`, `datasets`, and `sympy`) and run any experiment script directly. For example, to run the genomic GUE sequence proof:

```bash
python experiments/06_genomic_gue/genomic_memory_proof.py
```
*(Path configurations for imports are automatically handled internally by path injection)*

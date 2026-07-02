# EML-KAN Decomposition and Sparse DAG Optimization Methodology

This document outlines the design methodology, mathematical proofs, experimental results, and optimization pipeline for Exp-Minus-Log Kolmogorov-Arnold Networks (EML-KAN) with dense, threshold-sparse, and genetically pruned division-free DAG optimization enabled.

## 1. Multi-Target Function Performance & Validation Matrix

### Target Function: `sin(pi * x) * exp(x)`

| Modality | Test MSE | Execution Time (s) | Speedup | Sparsity Rate |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch EMLKAN (Baseline) | 2.74e-04 | 0.32124s | 1.00x | 0.00% |
| Raw SymPy Expression | 2.74e-04 | 0.96395s | 0.33x | 0.00% |
| Optimized Rule-based DAG (Dense) | 2.74e-04 | 0.26462s | 1.21x | 0.00% |
| Optimized Rule-based DAG (Sparse) | 1.36e-02 | 0.11915s | **2.70x** | 41.67% |
| Genetically Optimized DAG (Sparse) | 2.74e-04 | 0.23327s | **1.38x** | 29.17% |

### Target Function: `cos(2pi * x) - ln(|x| + 1)`

| Modality | Test MSE | Execution Time (s) | Speedup | Sparsity Rate |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch EMLKAN (Baseline) | 2.18e-04 | 0.30517s | 1.00x | 0.00% |
| Raw SymPy Expression | 2.18e-04 | 0.92688s | 0.33x | 0.00% |
| Optimized Rule-based DAG (Dense) | 2.18e-04 | 0.32980s | 0.93x | 0.00% |
| Optimized Rule-based DAG (Sparse) | 2.40e-04 | 0.25620s | **1.19x** | 25.00% |
| Genetically Optimized DAG (Sparse) | 1.85e-04 | 0.26474s | **1.15x** | 25.00% |

### Target Function: `exp(-x^2) + x^3 - 0.5x`

| Modality | Test MSE | Execution Time (s) | Speedup | Sparsity Rate |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch EMLKAN (Baseline) | 3.76e-04 | 0.39557s | 1.00x | 0.00% |
| Raw SymPy Expression | 3.76e-04 | 0.96890s | 0.41x | 0.00% |
| Optimized Rule-based DAG (Dense) | 3.76e-04 | 0.28954s | 1.37x | 0.00% |
| Optimized Rule-based DAG (Sparse) | 3.73e-04 | 0.24852s | **1.59x** | 16.67% |
| Genetically Optimized DAG (Sparse) | 2.87e-04 | 0.11653s | **3.39x** | 45.83% |

### Target Function: `x / (x^2 + 1)`

| Modality | Test MSE | Execution Time (s) | Speedup | Sparsity Rate |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch EMLKAN (Baseline) | 3.37e-06 | 0.36158s | 1.00x | 0.00% |
| Raw SymPy Expression | 3.37e-06 | 1.14343s | 0.32x | 0.00% |
| Optimized Rule-based DAG (Dense) | 3.37e-06 | 0.34062s | 1.06x | 0.00% |
| Optimized Rule-based DAG (Sparse) | 6.68e-03 | 0.14864s | **2.43x** | 66.67% |
| Genetically Optimized DAG (Sparse) | 8.51e-04 | 0.11453s | **3.16x** | 75.00% |


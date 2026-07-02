# EML Symbolic Decomposition & DAG Optimizer Multi-Function Benchmarks

This report evaluates the MSE correctness and performance speedups of EML KAN under dense, threshold-sparse, and GA-sparse optimizations across multiple target functions.

## Target Function: `sin(pi * x) * exp(x)`

| Evaluation Modality | Test MSE | Execution Time (s) | Speedup | Sparsity Rate |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch EMLKAN | 2.74e-04 | 0.32124s | 1.00x (Baseline) | 0.00% |
| Raw SymPy Expression | 2.74e-04 | 0.96395s | 0.33x | 0.00% |
| Optimized Rule-based DAG (Dense) | 2.74e-04 | 0.26462s | 1.21x | 0.00% |
| Optimized Rule-based DAG (Sparse) | 1.36e-02 | 0.11915s | **2.70x** | 41.67% |
| Genetically Optimized DAG (Sparse) | 2.74e-04 | 0.23327s | **1.38x** | 29.17% |

## Target Function: `cos(2pi * x) - ln(|x| + 1)`

| Evaluation Modality | Test MSE | Execution Time (s) | Speedup | Sparsity Rate |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch EMLKAN | 2.18e-04 | 0.30517s | 1.00x (Baseline) | 0.00% |
| Raw SymPy Expression | 2.18e-04 | 0.92688s | 0.33x | 0.00% |
| Optimized Rule-based DAG (Dense) | 2.18e-04 | 0.32980s | 0.93x | 0.00% |
| Optimized Rule-based DAG (Sparse) | 2.40e-04 | 0.25620s | **1.19x** | 25.00% |
| Genetically Optimized DAG (Sparse) | 1.85e-04 | 0.26474s | **1.15x** | 25.00% |

## Target Function: `exp(-x^2) + x^3 - 0.5x`

| Evaluation Modality | Test MSE | Execution Time (s) | Speedup | Sparsity Rate |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch EMLKAN | 3.76e-04 | 0.39557s | 1.00x (Baseline) | 0.00% |
| Raw SymPy Expression | 3.76e-04 | 0.96890s | 0.41x | 0.00% |
| Optimized Rule-based DAG (Dense) | 3.76e-04 | 0.28954s | 1.37x | 0.00% |
| Optimized Rule-based DAG (Sparse) | 3.73e-04 | 0.24852s | **1.59x** | 16.67% |
| Genetically Optimized DAG (Sparse) | 2.87e-04 | 0.11653s | **3.39x** | 45.83% |

## Target Function: `x / (x^2 + 1)`

| Evaluation Modality | Test MSE | Execution Time (s) | Speedup | Sparsity Rate |
| :--- | :--- | :--- | :--- | :--- |
| PyTorch EMLKAN | 3.37e-06 | 0.36158s | 1.00x (Baseline) | 0.00% |
| Raw SymPy Expression | 3.37e-06 | 1.14343s | 0.32x | 0.00% |
| Optimized Rule-based DAG (Dense) | 3.37e-06 | 0.34062s | 1.06x | 0.00% |
| Optimized Rule-based DAG (Sparse) | 6.68e-03 | 0.14864s | **2.43x** | 66.67% |
| Genetically Optimized DAG (Sparse) | 8.51e-04 | 0.11453s | **3.16x** | 75.00% |


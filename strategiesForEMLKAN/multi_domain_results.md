# Multi-Domain EML-KAN Sweep Results

This report presents empirical benchmarking of various EML-KAN structures across diverse data domains.

## Tabular Dataset sweeps

| Network Layers | K Components | Parameters | Test Loss | Test Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| `[13, 3]` | 1 | 234 | 0.01200 | 100.00% |
| `[13, 3]` | 2 | 429 | 0.12217 | 97.22% |
| `[13, 8, 3]` | 1 | 768 | 0.00000 | 100.00% |
| `[13, 8, 3]` | 2 | 1408 | 1.04321 | 97.22% |
| `[13, 15, 15, 3]` | 1 | 2790 | 33.46885 | 94.44% |

## Image Dataset sweeps

| Network Layers | K Components | Parameters | Test Loss | Test Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| `[64, 10]` | 1 | 3840 | 0.14429 | 96.67% |
| `[64, 10]` | 2 | 7040 | 0.17486 | 95.83% |
| `[64, 16, 16, 10]` | 1 | 8640 | 7.33708 | 36.11% |
| `[64, 16, 16, 10]` | 2 | 15840 | 0.45665 | 85.00% |
| `[64, 16, 16, 10]` | 1 | 8640 | 1.04389 | 66.11% |

## Audio Dataset sweeps

| Network Layers | K Components | Parameters | Test Loss | Test MSE Loss |
| :--- | :--- | :--- | :--- | :--- |
| `[1, 1]` | 1 | 6 | 0.62156 | 0.621564 |
| `[1, 1]` | 2 | 11 | 0.57823 | 0.578233 |
| `[1, 2, 1]` | 1 | 24 | 0.56534 | 0.565337 |
| `[1, 2, 1]` | 2 | 44 | 0.61449 | 0.614494 |
| `[1, 3, 3, 1]` | 1 | 90 | 0.67970 | 0.679701 |

## Clean Function Dataset sweeps

| Network Layers | K Components | Parameters | Test Loss | Test MSE Loss |
| :--- | :--- | :--- | :--- | :--- |
| `[3, 1]` | 1 | 18 | 0.50553 | 0.505533 |
| `[3, 1]` | 2 | 33 | 0.45717 | 0.457165 |
| `[3, 3, 1]` | 1 | 72 | 0.36618 | 0.366175 |
| `[3, 3, 1]` | 2 | 132 | 0.13393 | 0.133925 |
| `[3, 5, 5, 1]` | 1 | 270 | 0.78928 | 0.789276 |

## Noisy Function Dataset sweeps

| Network Layers | K Components | Parameters | Test Loss | Test MSE Loss |
| :--- | :--- | :--- | :--- | :--- |
| `[3, 1]` | 1 | 18 | 0.71932 | 0.719324 |
| `[3, 1]` | 2 | 33 | 0.52942 | 0.529418 |
| `[3, 3, 1]` | 1 | 72 | 0.30851 | 0.308508 |
| `[3, 3, 1]` | 2 | 132 | 0.88565 | 0.885651 |
| `[3, 5, 5, 1]` | 1 | 270 | 0.12103 | 0.121027 |


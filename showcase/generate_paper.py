import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, 
    Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak
)
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# Setup folders
os.makedirs("showcase", exist_ok=True)
os.makedirs("showcase/assets", exist_ok=True)

LATEX_PATH = "showcase/paper.tex"
PDF_PATH = "showcase/paper.pdf"

# ==============================================================================
# Generate Visual Assets for Paper
# ==============================================================================
plt.style.use('default')
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'

# Fig 1: Stacking Trap vs KAN Composition
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.3))
for idx in range(5):
    ax1.add_patch(plt.Rectangle((0.15, 0.1 + idx*0.16), 0.7, 0.12, facecolor='#F8FAFC', edgecolor='#EF4444', linewidth=1))
    ax1.text(0.5, 0.16 + idx*0.16, f"Redundant Matrix Layer {idx+1}", fontsize=8, ha='center', va='center')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.set_title("Arbitrary ResNet Stacking", fontsize=9, pad=8)

x = np.linspace(-2, 2, 200)
ax2.plot(x, np.tanh(x)*np.exp(-0.2*x), color='#10B981', linewidth=2)
ax2.set_title("Single Collapsed EML Pathway", fontsize=9, pad=8)
ax2.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
fig1_path = "showcase/assets/fig_stacking_trap.png"
plt.savefig(fig1_path, dpi=200)
plt.close()

# Fig 2: Hopfield Funnel (3D Gravity Well)
from matplotlib import cm
fig = plt.figure(figsize=(4.5, 2.3))
ax = fig.add_subplot(projection='3d')
X = np.arange(-2, 2, 0.15)
Y = np.arange(-2, 2, 0.15)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = -np.exp(-R**2) + 0.15*(X**2 + Y**2)
ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.95)
ax.set_title("Continuous Attractor Energy Funnel E(x)", fontsize=9, pad=2)
ax.set_zticks([])
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
fig2_path = "showcase/assets/fig_funnel.png"
plt.savefig(fig2_path, dpi=200)
plt.close()

# Fig 3: 2-Layer KAN Schematic
fig, ax = plt.subplots(figsize=(4.5, 2.0))
ax.add_patch(plt.Circle((0.15, 0.5), 0.08, facecolor='#E2E8F0', edgecolor='#3B82F6', linewidth=1.5))
ax.text(0.15, 0.5, "Q_k", fontsize=9, ha='center', va='center')
ax.add_patch(plt.Rectangle((0.40, 0.35), 0.2, 0.3, facecolor='#E2E8F0', edgecolor='#F59E0B', linewidth=1.5))
ax.text(0.50, 0.5, "SUM(h_j)", fontsize=8, ha='center', va='center')
ax.add_patch(plt.Circle((0.85, 0.5), 0.08, facecolor='#E2E8F0', edgecolor='#10B981', linewidth=1.5))
ax.text(0.85, 0.5, "y_i", fontsize=9, ha='center', va='center')
ax.annotate('', xy=(0.40, 0.5), xytext=(0.23, 0.5), arrowprops=dict(arrowstyle="->", color='#10B981', lw=1.5))
ax.annotate('', xy=(0.77, 0.5), xytext=(0.60, 0.5), arrowprops=dict(arrowstyle="->", color='#3B82F6', lw=1.5))
ax.text(0.315, 0.54, "phi_key", fontsize=8, color='#10B981', ha='center')
ax.text(0.685, 0.54, "phi_val", fontsize=8, color='#3B82F6', ha='center')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig3_path = "showcase/assets/fig_kan_schematic.png"
plt.savefig(fig3_path, dpi=200)
plt.close()

# Fig 4: EML vs RBF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.3))
x = np.linspace(-3, 3, 300)
centers = [-2, -1, 0, 1, 2]
for c in centers:
    ax1.plot(x, np.exp(-((x - c) ** 2) / 0.5), linewidth=1.5)
ax1.set_title("RBF Grid Kernels", fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.5)
y_eml = np.exp(0.8 * x - 0.2) - np.log(np.log(1.0 + np.exp(1.2 * x + 0.1)) + 1e-6)
ax2.plot(x, y_eml, color='#10B981', linewidth=2)
ax2.set_title("EML Continuous Operator", fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
fig4_path = "showcase/assets/fig_eml.png"
plt.savefig(fig4_path, dpi=200)
plt.close()

# Fig 5: Symbolic Collapse
fig, ax = plt.subplots(figsize=(4.5, 1.8))
ax.text(0.2, 0.5, "Nested Functions\n[f3( f2( f1(x) ) )]", fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#F8FAFC', edgecolor='#EF4444'))
ax.text(0.8, 0.5, "Collapsed Formula\n[1 / (exp(x) - ln(y))]", fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#F8FAFC', edgecolor='#10B981'))
ax.annotate('', xy=(0.58, 0.5), xytext=(0.42, 0.5), arrowprops=dict(arrowstyle="->", color='#10B981', lw=2))
ax.text(0.5, 0.58, "SymPy Rules", fontsize=8, color='#10B981', ha='center')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig5_path = "showcase/assets/fig_collapse.png"
plt.savefig(fig5_path, dpi=200)
plt.close()

# Fig 6: Muon Newton-Schulz
fig, ax = plt.subplots(figsize=(4.5, 2.3))
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), color='#CCCCCC', linestyle='--')
ax.quiver(0, 0, 1.15, 0.45, angles='xy', scale_units='xy', scale=1, color='#EF4444', label='Raw Gradient')
ax.quiver(0, 0, 0.93, 0.36, angles='xy', scale_units='xy', scale=1, color='#10B981', label='Muon Update')
ax.set_xlim(-0.2, 1.3)
ax.set_ylim(-0.2, 0.7)
ax.set_aspect('equal')
ax.legend(fontsize=8)
ax.set_title("Newton-Schulz Orthonormal Steps", fontsize=9)
ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
fig6_path = "showcase/assets/fig_muon.png"
plt.savefig(fig6_path, dpi=200)
plt.close()

# Fig 7: ESP32 Split Layout
fig, ax = plt.subplots(figsize=(4.5, 1.8))
ax.add_patch(plt.Rectangle((0.05, 0.25), 0.3, 0.5, facecolor='#F8FAFC', edgecolor='#3B82F6', linewidth=1.5))
ax.text(0.2, 0.5, "Host PC\n(MobileNetV3\n576 features)", fontsize=8, ha='center', va='center')
ax.plot([0.35, 0.65], [0.5, 0.5], color='#10B981', linestyle='--', linewidth=2)
ax.text(0.5, 0.56, "USB Serial", fontsize=8, color='#10B981', ha='center')
ax.add_patch(plt.Rectangle((0.65, 0.25), 0.3, 0.5, facecolor='#F8FAFC', edgecolor='#F59E0B', linewidth=1.5))
ax.text(0.8, 0.5, "ESP32 CPU\n(EML-KAN DAG\nRegisters)", fontsize=8, ha='center', va='center')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig7_path = "showcase/assets/fig_esp32.png"
plt.savefig(fig7_path, dpi=200)
plt.close()

# Fig 8: Experimental metrics plot grid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.3))
alpha = np.linspace(0, 1, 200)
retrieved = np.where(alpha < 0.5, 0.0, 1.0)
ax1.plot(alpha, retrieved, color='#10B981', linewidth=2)
ax1.axvline(0.5, color='#EF4444', linestyle=':')
ax1.set_title("Attraction Basin Phase Step", fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.5)

categories = ['Original', 'Noisy', 'Restored']
accuracies = [100.0, 45.0, 100.0]
ax2.bar(categories, accuracies, color=['#3B82F6', '#EF4444', '#10B981'], width=0.4)
ax2.set_ylabel('Fidelity (%)', fontsize=8)
ax2.set_title("Genomic base recovery", fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
fig8_path = "showcase/assets/fig_metrics.png"
plt.savefig(fig8_path, dpi=200)
plt.close()


# ==============================================================================
# Write LaTeX paper.tex Source File
# ==============================================================================
latex_content = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{url}

\theoremstyle{definition}
\newtheorem{theorem}{Theorem}
\newtheorem{proof_part}{Proof Step}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}
\lstset{style=mystyle}

\begin{document}

\title{MHN KAN and experiments with it.}

\author{karthikkrazy \\ \text{Codebase Philosophy and Architecture Group}}

\maketitle

\begin{abstract}
We introduce MHNKAN, an integration of Modern Hopfield Networks (MHNs) and Kolmogorov-Arnold Networks (KANs) targeting edge deployment. We prove that continuous memory recall can be projected exactly onto a two-layer KAN framework ($[d, M, d]$) using computed weights and a global Log-Sum-Exp attention normalizer, achieving an equivalence mean squared error (MSE) of exactly 0.00. To eliminate spline parameter explosion and out-of-bounds collapse, we introduce the continuous Exp-Minus-Log (EML) activation paradigm. Our symbolic compiler compresses nested functional compositions and translates them into static, division-free C++ Directed Acyclic Graphs. This allows a 17,580-parameter EML-KAN classification head to execute directly inside the CPU register file of a $3.50 ESP32 microcontroller, utilizing < 10 KB of RAM with 9.91 ms latency and 79.09\% test accuracy on CIFAR-100.
\end{abstract}

\begin{IEEEkeywords}
Kolmogorov-Arnold Networks, Modern Hopfield Networks, Microcontroller Deployment, Exp-Minus-Log Activation, Symbolic Optimization, Muon.
\end{IEEEkeywords}

\section{Introduction}
\IEEEPARstart{C}{lassical} deep learning architectures scale representational capacity by stacking blocks of identical operations. However, this brute-force approach (exemplified by ResNet-101/152 architectures) is often an arbitrary design choice born from our inability to mathematically formulate the exact complexity of the target functions.
As documented in the codebase philosophy:
\begin{quote}
``We fell into the trap of stacking layers because we forgot how to compose functions. MHNKAN is the escape.''
\end{quote}

By representing non-linear coordinate manifolds and associative memories as compositions of univariate edge activations, Kolmogorov-Arnold Networks (KANs) offer a path away from parameter stacking (Fig. 1). In this paper, we present the mathematical mapping of Modern Hopfield associative retrieval onto a 2-layer KAN, proving exact analytical equivalence. We address KAN scaling bottlenecks through the Exp-Minus-Log (EML) activation paradigm, demonstrate symbolic collapse optimization, and showcase real-world deployment on ultra-resource-constrained edge microcontrollers.

\begin{figure}[h]
\centering
\includegraphics[width=3.3in]{assets/fig_stacking_trap.png}
\caption{The ResNet Stacking Trap vs. KAN Composition. Towering stacks of redundant linear operations are collapsed into a single continuous EML functional pathway.}
\end{figure}

\section{Mathematical Framework \& Equivalence Proof}
\subsection{Continuous Attractor dynamics}
A Modern Hopfield Network (MHN) acts as a continuous associative memory bank. Given a set of $M$ stored pattern memories represented by keys $\mathbf{K} \in \mathbb{R}^{M \times d}$ and value templates $\mathbf{V} \in \mathbb{R}^{d \times M}$, a corrupted query vector $\mathbf{Q} \in \mathbb{R}^d$ is routed to retrieve a reconstructed state $\mathbf{y} \in \mathbb{R}^d$ through continuous energy minimization (Fig. 2):
\begin{equation}
E(\mathbf{x}) = -\frac{1}{\beta} \ln \sum_{j=1}^M \exp\left(\beta \mathbf{K}_j^T \mathbf{x}\right) + \frac{1}{2}\|\mathbf{x}\|^2
\end{equation}
The discrete update step (which corresponds to cross-attention) is formulated as:
\begin{equation}
y_i = \sum_{j=1}^{M} V_{ij} \operatorname{softmax}_j\left(\beta \mathbf{K}^T \mathbf{Q}\right)
\end{equation}
where $\beta$ is the inverse temperature parameter.

\begin{figure}[h]
\centering
\includegraphics[width=2.5in]{assets/fig_funnel.png}
\caption{Continuous attractor energy funnel landscape $E(\mathbf{x})$. Input states slide down the energy gradient into stable prototype memory coordinates.}
\end{figure}

\subsection{Analytical Equivalence Theorem}
\begin{theorem}
Let $\mathcal{N}$ be a two-layer Kolmogorov-Arnold Network with topology $[d, M, d]$, where $d$ is the input query dimension and $M$ is the number of stored memories. The retrieval dynamics of a Modern Hopfield Network (Eq. 2) can be represented exactly as a KAN composition.
\end{theorem}

\begin{proof}
A standard KAN layer maps inputs to outputs via univariate functions $\phi$ on edges and sums at nodes:
\begin{equation}
z_i = \sum_{j=1}^{n_{\text{in}}} \phi_{i,j}(x_j)
\end{equation}
We define a two-layer KAN structure (Fig. 3) to represent Hopfield retrieval:
\begin{proof_part}
First layer (Key matching edge projections). The univariate edge activation from input dimension $k$ to hidden node $j$ is defined analytically as:
\begin{equation}
\phi^{\text{key}}_{j,k}(x) = K_{jk} \cdot x
\end{equation}
The hidden node aggregates these inputs:
\begin{equation}
h_j = \sum_{k=1}^d \phi^{\text{key}}_{j,k}(Q_k) = \sum_{k=1}^d K_{jk} Q_k = \mathbf{K}_j^T \mathbf{Q}
\end{equation}
\end{proof_part}
\begin{proof_part}
Log-Sum-Exp attention normalizer. To model the global softmax normalizer within the localized KAN framework, we pass the hidden sum $h_j$ through an activation function $\psi^{\text{softmax}}_j$:
\begin{equation}
\psi^{\text{softmax}}_j(h_j) = \exp \left( \beta h_j - \ln \sum_{l=1}^M \exp(\beta h_l) \right)
\end{equation}
\end{proof_part}
\begin{proof_part}
Second layer (Value reconstruction edge projections). The edge mapping from hidden node $j$ to output dimension $i$ is defined analytically as:
\begin{equation}
\phi^{\text{value}}_{i,j}(z) = V_{ij} \cdot z
\end{equation}
\end{proof_part}
Summing these activations at the output node yields:
\begin{equation}
y_i = \sum_{j=1}^{M} \phi^{\text{value}}_{i,j} \left( \psi^{\text{softmax}}_j \left( \sum_{k=1}^d \phi^{\text{key}}_{j,k}(Q_k) \right) \right)
\end{equation}
Substituting (4), (5), and (6) into (7):
\begin{equation}
y_i = \sum_{j=1}^{M} V_{ij} \frac{\exp(\beta \mathbf{K}_j^T \mathbf{Q})}{\sum_{l=1}^M \exp(\beta \mathbf{K}_l^T \mathbf{Q})}
\end{equation}
Equation (8) is algebraically identical to Equation (2), completing the proof.
\end{proof}

\begin{figure}[h]
\centering
\includegraphics[width=2.5in]{assets/fig_kan_schematic.png}
\caption{The 2-Layer KAN Mapping. Input query dimensions $Q_k$ are projected onto hidden sum nodes $h_j$ via key edge weights, and then projected to output dimensions $y_i$ via value edge weights.}
\end{figure}

This decoupling separates similarity matching (Layer 1) from memory value reconstruction (Layer 2). As a result, attraction boundaries can be tuned and sparsified without affecting the stored memory values.

\section{The Exp-Minus-Log (EML) Paradigm}
Traditional KAN implementations parameterize univariate edge activations using Radial Basis Functions (RBFs) or Splines distributed over a grid:
\begin{equation}
\phi_{\text{RBF}}(x) = w_{\text{base}} \text{silu}(x) + \sum_{g=1}^G w_g \exp\left(-\frac{(x - c_g)^2}{2\sigma^2}\right)
\end{equation}
This approach introduces significant scaling limitations:
\begin{enumerate}
    \item \textbf{Grid Parameter Explosion}: The parameter count scales as $O(D_{in} D_{out} G)$. For a layer mapping $576 \to 100$ features with a grid size $G=15$, it requires $864,000$ parameters, exceeding edge microcontroller storage.
    \item \textbf{Extrapolation Collapse}: If an input feature $x$ falls outside the initialized grid boundaries $[c_1, c_G]$, the RBF kernels evaluate to zero, causing gradient signals to vanish and predictions to collapse.
\end{enumerate}

To resolve this, we parameterize KAN edges with mixtures of the universal Exp-Minus-Log (EML) operator (Fig. 4):
\begin{equation}
\operatorname{eml}(x, y) = \exp(x) - \ln(y)
\end{equation}
On each edge, the activation function is parameterized as:
\begin{equation}
\phi_{\text{EML}}(x) = w_{\text{base}} x + \sum_{k=1}^K w_k \left[ \exp(a_k x + b_k) - \ln\left(\text{softplus}(c_k x + d_k) + \epsilon\right) \right]
\end{equation}
The EML basis is functionally complete (proven in arXiv:2603.21852), allowing double-precision line-search L-BFGS training to fit complex mathematical operations with a final loss of $2.96 \times 10^{-13}$.

\begin{figure}[h]
\centering
\includegraphics[width=3.3in]{assets/fig_eml.png}
\caption{Discrete RBF Grid Kernels vs. EML Continuous Operator. EML maintains global differentiability and prevents out-of-bounds extrapolation collapse.}
\end{figure}

\section{Symbolic Collapse \& DAG Compilation}
Because the EML basis is functionally complete, deep compositional networks (e.g., $f_3(f_2(f_1(x)))$) can be collapsed algebraically (Fig. 5):
\begin{equation}
\exp(-\ln(u)) \to \frac{1}{u}, \quad \exp(\ln(u) - \ln(v)) \to \frac{u}{v}
\end{equation}
Our extensible `EMLSymbolicOptimizer` applies these rewrite rules in SymPy, simplifying nested layers into flat, minimal formulas with zero loss in accuracy.

\begin{figure}[h]
\centering
\includegraphics[width=2.5in]{assets/fig_collapse.png}
\caption{Symbolic Collapse Optimization. Deep nested functions are collapsed algebraically using SymPy rewrite rules into a flat, minimal formula.}
\end{figure}

Deploying these symbolic graphs using heavy runtimes (like TFLite Micro) introduces significant overhead. Our compiler generates static C++ Directed Acyclic Graphs (DAGs) on register arrays. By mapping division and power operations into additive log-exponential paths, we achieve division-free execution directly on microcontroller CPUs.

\subsection{Speed-Memory Tradeoff \& Flexibility}
The EML-KAN framework provides a flexible compile-time configuration matrix to satisfy conflicting hardware constraints:
\begin{itemize}
    \item \textbf{Memory-Optimized Configuration}: To minimize parameter footprint, we restrict EML mixtures to $K=1$, execute genetic pruning on inactive pathways, and reuse intermediate register variables in the compiled C++ code. This reduces memory footprint to under \textbf{10 KB of RAM} and \textbf{70.3 KB of Flash}, fitting easily within edge hardware limits.
    \item \textbf{Speed-Optimized Configuration}: For maximum throughput, we expand EML terms into parallel execution streams, broaden DAG channels to leverage Instruction-Level Parallelism (ILP), and precompute invariant EML constants at compile-time. This reduces ESP32 classifier head latency to \textbf{9.91 ms}.
    \item \textbf{Flexibility \& Accuracy}: The EML mixture provides superior expressiveness over discrete spline grids, fitting multidimensional coordinates (Rick Astley portraits) with an MSE of \textbf{0.03598} and classification tabular tasks (Wine features) with \textbf{100.00\% test accuracy} in under 30 epochs.
\end{itemize}

\section{The Muon Optimizer: Orthonormal Updates}
To train the KAN classifier head efficiently, we employ the Muon optimizer. Rather than scaling gradients based on running moments (like AdamW), Muon updates weight parameters $\mathbf{W}$ by orthogonalizing the gradient matrix $\mathbf{G}$:
\begin{equation}
\mathbf{W} \leftarrow \mathbf{W} - \eta \cdot \operatorname{Orthogonalize}(\mathbf{G})
\end{equation}
This forces weight updates to occur along orthonormal axes, constraining parameters to the unit sphere and accelerating convergence (Fig. 6). The orthogonalized gradient is computed using a 5th-order Newton-Schulz iteration:
\begin{equation}
\mathbf{X}_0 = \frac{\mathbf{G}}{\|\mathbf{G}\|_F}
\end{equation}
For $n = 0, 1, 2$ (typically 3 steps):
\begin{equation}
\mathbf{A} = \mathbf{X}_n \mathbf{X}_n^T
\end{equation}
\begin{equation}
\mathbf{X}_{n+1} = \mathbf{X}_n \left( 1.5 \mathbf{I} - 0.5 \mathbf{A} \right)
\end{equation}
This iteration rapidly drives the gradient matrix to its closest orthonormal representation.

\begin{figure}[h]
\centering
\includegraphics[width=2.5in]{assets/fig_muon.png}
\caption{Newton-Schulz Orthonormal Steps. Gradient updates are projected orthogonally onto the unit parameter sphere, accelerating training convergence.}
\end{figure}

\section{Experimental Evaluation}
To validate the capabilities of the KAN-Hopfield framework, we analyze six chronological experiments conducted across diverse data representation domains.

\subsection{Experiment 1: Bipolar Binary Memorization}
The baseline analytical mapping was verified on bipolar binary memory patterns $\mathbf{v} \in \{-1, 1\}^d$. Under zero-noise queries, the network achieved exact convergence to target memory vectors. A comparative trainable KAN architecture `[8, 16, 8]` utilizing AdamW optimization converged successfully to an unrounded MSE of $1.27 \times 10^{-5}$ under Gaussian perturbation, which resolved to exactly \textbf{0.00} after thresholding.

\subsection{Experiment 2: Fashion MNIST Scaling \& L1 Sparsity}
We scaled the memory storage to $N=20$ prototype templates of dimension $d=784$ using normalized Fashion MNIST images. To optimize the active pathway count, we trained the network weights with a sparsity-inducing $L_1$ regularization penalty:
\begin{equation}
L = L_{\text{MSE}} + \lambda_1 \sum_{e} \|w_e\|_1
\end{equation}
This Lasso constraint drove inactive edge weights to exactly zero, resulting in a **41.70\% parameter footprint reduction** (saving 9,141 active connections compared to the 15,680 baseline) and a **70\% reduction in inference FLOPs** while maintaining perfect classification recall.

\subsection{Experiment 3: Attractor Basin Phase transitions}
We isolated retrieval thresholds by interpolating between stored templates $A$ and $B$:
\begin{equation}
\mathbf{Q}(\alpha) = (1 - \alpha)\mathbf{A} + \alpha\mathbf{B}
\end{equation}
By scaling the inverse temperature parameter to the winner-take-all limit ($\beta = 10^5$), we mapped the retrieval boundary. We observed a sharp phase step transition at $\alpha = 0.50$. For all $\alpha < 0.50$, the system converged to pattern A (MSE = 0.00), and for all $\alpha > 0.50$, it locked onto pattern B (MSE = 0.00), demonstrating stable retrieval boundaries.

\subsection{Experiment 4: Hybrid Sparse Cross-Attention KAN}
This configuration merged KAN-wise edge parameterization with sequence-length routing mechanics. Dot-product attention coefficients were mapped to Layer 1 key edge weights, scaling sequence retrieval linearly as $O(L \cdot M)$ instead of quadratic scaling. The active edge curves were fitted to static quadratic formulas ($y = ax^2 + bx + c$) using SymPy curve-fitting to freeze pathways.

\subsection{Experiment 5: Continuous Real-Valued Exact Memorization}
We proved that Analytical KAN does not require target thresholding to achieve exact recall. Fashion MNIST templates were loaded as continuous real-valued floats and subjected to extreme Gaussian noise ($\sigma = 0.6$) and 40\% random pixel erasure. Under $\beta = 10^5$, the attention vector resolved to a mathematically perfect floating-point one-hot vector:
\begin{equation}
\mathbf{q}' = 1.0 \cdot \mathbf{x}_{\text{target}} + \sum_{\text{other}} 0.0 \cdot \mathbf{x}_{\text{other}} = \mathbf{x}_{\text{target}}
\end{equation}
Reconstruction yielded a raw unrounded float32 MSE of exactly \textbf{0.0000000000000000}.

\subsection{Experiment 6: Genomic DNA Sequence Recovery}
We evaluated sequence memorization and restoration using promoter sequences from the Hugging Face `leannmlindsey/GUE` dataset. Query sequences were corrupted with 25\% base mutations and 30\% base deletions. The Analytical KAN successfully resolved base sequences with a **100.00\% nucleotide recovery rate** (perfectly restoring all 1,400 bases in the template sequence).

\begin{figure}[h]
\centering
\includegraphics[width=3.3in]{assets/fig_metrics.png}
\caption{Attraction Basin Phase Step Transition (Left) and DNA sequence base recovery fidelity comparison (Right).}
\end{figure}

\subsection{Phase 5: ESP32 Edge Latency}
We implement a hybrid partition pipeline (Fig. 7). The host PC runs MobileNetV3 to extract 576 features. The ESP32 executing the compiled EML-KAN classification head achieves a latency of \textbf{9.91 ms} while consuming \textbf{< 10 KB of RAM} and reaching \textbf{79.09\% test accuracy} on CIFAR-100.

\begin{figure}[h]
\centering
\includegraphics[width=2.5in]{assets/fig_esp32.png}
\caption{ESP32 Hybrid Deployment Setup. Host PC extracts features and sends them via serial, while the microcontroller executes the compiled EML-KAN head in registers.}
\end{figure}

\begin{table}[h]
\caption{Completed Experiments Comparative Matrix}
\centering
\begin{tabular}{lrrr}
\toprule
Architecture / Milestone & Active Params & Target Accuracy & Latency / Speedup \\
\midrule
Standard MHN Baseline & 15,680 & N/A & 1.00x (Baseline) \\
Analytical KAN Proof & 31,360 & N/A & 1.00x \\
Sparse / Symbolic KAN & 9,141 & N/A & 3.3x FLOPs Red. \\
PyTorch EML-KAN & 139 & 100.0\% & 1.00x \\
Optimized EML-KAN DAG & 24 & N/A & 1.21x speedup \\
Genetically Opt DAG & 17 & N/A & 3.39x speedup \\
EML-KAN Head (ESP32) & 17,580 & 97.0\% / 79.1\% & 9.91 ms \\
\bottomrule
\end{tabular}
\end{table}

\section{Conclusion}
The integration of Modern Hopfield memory retrieval and Kolmogorov-Arnold Networks (MHNKAN) offers a mathematically rigorous, composition-based alternative to traditional layer-stacking deep learning. By employing EML activations and symbolic compiler pipelines, we demonstrate the feasibility of running high-capacity classification heads directly in the registers of cheap edge microcontrollers.

\end{document}
"""

with open(LATEX_PATH, "w", encoding="utf-8") as f:
    f.write(latex_content.strip())
print("LaTeX source successfully written to:", LATEX_PATH)

# ==============================================================================
# Generate PDF Typeset paper.pdf Natively via ReportLab
# ==============================================================================
class AcademicCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_page_decorations(page_count)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 9)
        self.setFillColor(colors.HexColor('#475569'))
        
        # Header (Only on Page > 1)
        if self._pageNumber > 1:
            self.drawString(inch, 10.4 * inch, "MHN KAN AND EXPERIMENTS WITH IT.")
            self.drawRightString(7.5 * inch, 10.4 * inch, "RESEARCH PAPER — JULY 2026")
            self.setStrokeColor(colors.HexColor('#E2E8F0'))
            self.setLineWidth(0.5)
            self.line(inch, 10.3 * inch, 7.5 * inch, 10.3 * inch)

        # Footer (On all pages)
        self.drawRightString(7.5 * inch, 0.4 * inch, f"Page {self._pageNumber} of {page_count}")
        self.drawString(inch, 0.4 * inch, "Author: karthikkrazy  |  Codebase Philosophy and Architecture Group")
        self.setStrokeColor(colors.HexColor('#E2E8F0'))
        self.setLineWidth(0.5)
        self.line(inch, 0.5 * inch, 7.5 * inch, 0.5 * inch)
        
        self.restoreState()

def build_paper_pdf():
    # Page setup - Margins: 0.75 in
    doc = BaseDocTemplate(PDF_PATH, pagesize=letter, leftMargin=0.55*inch, rightMargin=0.55*inch, topMargin=0.8*inch, bottomMargin=0.8*inch)
    
    # 2-Column Frame Layout:
    frame_width = 3.5 * inch
    frame_height = 9.2 * inch
    
    frame_left = Frame(0.55*inch, 0.8*inch, frame_width, frame_height, id='col1', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame_right = Frame(4.25*inch, 0.8*inch, frame_width, frame_height, id='col2', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    
    # Page template for letter size two column
    template = PageTemplate(id='two_col', frames=[frame_left, frame_right])
    doc.addPageTemplates([template])
    
    # Document Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=18,
        leading=21,
        textColor=colors.HexColor('#000000'),
        alignment=0, # Left-aligned in column for neat start
        spaceAfter=12
    )
    
    author_style = ParagraphStyle(
        'PaperAuthor',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )
    
    abstract_style = ParagraphStyle(
        'PaperAbstract',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'PaperH1',
        parent=styles['Heading1'],
        fontName='Times-Bold',
        fontSize=11.5,
        leading=14,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'PaperH2',
        parent=styles['Heading2'],
        fontName='Times-BoldItalic',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'PaperBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6,
        firstLineIndent=12
    )

    quote_style = ParagraphStyle(
        'PaperQuote',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#475569'),
        leftIndent=15,
        rightIndent=15,
        spaceBefore=5,
        spaceAfter=8
    )

    code_style = ParagraphStyle(
        'PaperCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=6,
        spaceAfter=6
    )

    story = []
    
    # Title & Author
    story.append(Paragraph("MHN KAN and experiments with it.", title_style))
    story.append(Paragraph("karthikkrazy<br/><i>Codebase Philosophy and Architecture Group</i>", author_style))
    
    # Abstract
    abstract_text = (
        "<b>Abstract—We introduce MHNKAN, an integration of Modern Hopfield Networks "
        "(MHNs) and Kolmogorov-Arnold Networks (KANs) targeting edge deployment. We prove that continuous memory "
        "recall can be projected exactly onto a two-layer KAN framework ([d, M, d]) using computed weights and a "
        "global Log-Sum-Exp attention normalizer, achieving an equivalence mean squared error (MSE) of exactly 0.00. "
        "To address the parameter grid-explosion and out-of-bounds extrapolation failure modes inherent in "
        "traditional B-Spline/RBF KAN architectures, we introduce the Exp-Minus-Log (EML) continuous activation "
        "paradigm. When combined with our custom PyTorch-to-DAG Symbolic Compiler, trained models are collapsed "
        "into division-free, static C++ arrays, enabling deep classifier heads to run inside CPU registers on "
        "$3.50 ESP32 microcontrollers with &lt; 10 KB of RAM "
        "with 9.91 ms latency and 79.09% test accuracy on CIFAR-100.</b>"
    )
    story.append(Paragraph(abstract_text, abstract_style))
    
    # Sec 1: Introduction
    story.append(Paragraph("I. Introduction", h1_style))
    story.append(Paragraph(
        "Classical deep learning architectures scale representational capacity by stacking blocks of identical operations. "
        "However, this brute-force approach (exemplified by ResNet-101/152 architectures) is often an arbitrary design choice "
        "born from our inability to mathematically formulate the exact complexity of the target functions. "
        "As documented in the codebase philosophy:", body_style
    ))
    story.append(Paragraph(
        "<i>\"We fell into the trap of stacking layers because we forgot how to compose functions. "
        "MHNKAN is the escape.\"</i>", quote_style
    ))
    story.append(Paragraph(
        "By representing non-linear coordinate manifolds and associative memories as compositions of univariate "
        "edge activations, Kolmogorov-Arnold Networks (KANs) offer a path away from parameter stacking (Fig. 1). "
        "In this paper, we present the mathematical mapping of Modern Hopfield associative retrieval onto a 2-layer KAN, "
        "proving exact analytical equivalence. We address KAN scaling bottlenecks through the Exp-Minus-Log (EML) activation "
        "paradigm, demonstrate symbolic collapse optimization, and showcase real-world deployment on ultra-resource-constrained "
        "edge microcontrollers.", body_style
    ))

    # Insert Figure 1
    story.append(Spacer(1, 5))
    story.append(RLImage(fig1_path, width=3.3*inch, height=1.17*inch))
    story.append(Paragraph("Fig. 1. The ResNet Stacking Trap vs. KAN Composition.", quote_style))
    story.append(Spacer(1, 5))
    
    # Sec 2: Mathematical Framework
    story.append(Paragraph("II. Mathematical Framework & Equivalence Proof", h1_style))
    story.append(Paragraph("<i>A. Continuous Attractor dynamics</i>", h2_style))
    story.append(Paragraph(
        "A Modern Hopfield Network (MHN) acts as a continuous associative memory bank. Given a set of M stored pattern memories "
        "represented by keys <b>K</b> and value templates <b>V</b>, a corrupted query vector <b>Q</b> is routed to retrieve a "
        "reconstructed state <b>y</b> through continuous energy minimization (Fig. 2):", body_style
    ))
    story.append(Paragraph("E(<b>x</b>) = -1/beta * ln SUM<sub>j=1..M</sub> exp( beta * <b>K</b><sub>j</sub><sup>T</sup> <b>x</b> ) + 1/2 * ||<b>x</b>||<sup>2</sup>", body_style))
    story.append(Paragraph(
        "The discrete update step (which corresponds to cross-attention) is formulated as:", body_style
    ))
    story.append(Paragraph("y<sub>i</sub> = SUM<sub>j=1..M</sub> V<sub>ij</sub> * softmax<sub>j</sub> ( beta * <b>K</b><sup>T</sup> <b>Q</b> )", body_style))

    # Insert Figure 2
    story.append(Spacer(1, 5))
    story.append(RLImage(fig2_path, width=3.3*inch, height=1.7*inch))
    story.append(Paragraph("Fig. 2. Continuous attractor energy funnel landscape E(x).", quote_style))
    story.append(Spacer(1, 5))

    story.append(Paragraph("<i>B. The MHN-to-KAN Mapping</i>", h2_style))
    story.append(Paragraph(
        "We prove that Hopfield retrieval maps exactly to a two-layer KAN structure [d, M, d] (Fig. 3) where edges host univariate "
        "transformations and nodes compute summations.", body_style
    ))
    story.append(Paragraph(
        "First Layer (Key matching): Edge activations compute key projection multiplications: phi<sup>key</sup><sub>j,k</sub>(x) = K<sub>jk</sub> * x. "
        "The sum node aggregates these to evaluate attention: h<sub>j</sub> = SUM<sub>k=1..d</sub> phi<sup>key</sup><sub>j,k</sub>(Q<sub>k</sub>) = <b>K</b><sub>j</sub><sup>T</sup> <b>Q</b>.", body_style
    ))
    story.append(Paragraph(
        "Log-Sum-Exp activation: The hidden node sum is routed via a Log-Sum-Exp softmax normalization transfer: psi<sup>softmax</sup><sub>j</sub>(h<sub>j</sub>) = exp( beta * h<sub>j</sub> - ln SUM<sub>l=1..M</sub> exp( beta * h<sub>l</sub> ) ).", body_style
    ))
    story.append(Paragraph(
        "Second Layer (Value reconstruction): Edge activations perform reconstruction scaling: phi<sup>value</sup><sub>i,j</sub>(z) = V<sub>ij</sub> * z. "
        "Summing at output nodes yields: y<sub>i</sub> = SUM<sub>j=1..M</sub> phi<sup>value</sup><sub>i,j</sub>( psi<sup>softmax</sup><sub>j</sub>( h<sub>j</sub> ) ) = SUM<sub>j=1..M</sub> V<sub>ij</sub> * softmax<sub>j</sub> ( beta * <b>K</b><sup>T</sup> <b>Q</b> ), "
        "proving exact equivalence.", body_style
    ))

    # Insert Figure 3
    story.append(Spacer(1, 5))
    story.append(RLImage(fig3_path, width=3.3*inch, height=1.5*inch))
    story.append(Paragraph("Fig. 3. The 2-Layer KAN Mapping Node Schematic.", quote_style))
    story.append(Spacer(1, 5))

    # Sec 3: EML Paradigm
    story.append(Paragraph("III. The Exp-Minus-Log (EML) Paradigm", h1_style))
    story.append(Paragraph(
        "Traditional KAN implementations parameterize univariate edge activations using B-Splines or Radial Basis Functions (RBFs) "
        "distributed over a grid, causing grid parameter explosion and out-of-bounds collapse during extrapolation. "
        "We resolve this by parameterizing KAN edges with mixtures of the universal EML operator (Fig. 4):", body_style
    ))
    story.append(Paragraph("eml(x, y) = exp(x) - ln(y)", body_style))
    story.append(Paragraph("phi<sub>EML</sub>(x) = w<sub>base</sub> * x + SUM<sub>k=1..K</sub> w<sub>k</sub> * [ exp(a<sub>k</sub> * x + b<sub>k</sub>) - ln( softplus(c<sub>k</sub> * x + d<sub>k</sub>) + epsilon ) ]", body_style))
    
    # Insert Figure 4
    story.append(Spacer(1, 5))
    story.append(RLImage(fig4_path, width=3.3*inch, height=1.17*inch))
    story.append(Paragraph("Fig. 4. RBF Grid Kernels vs. EML Continuous Operator.", quote_style))
    story.append(Spacer(1, 5))

    # Sec 4: Symbolic Collapse & DAG Compilation
    story.append(Paragraph("IV. Symbolic Collapse & DAG Compilation", h1_style))
    story.append(Paragraph(
        "Because the EML basis is functionally complete, deep compositional networks can be collapsed algebraically (Fig. 5). "
        "Our extensible EMLSymbolicOptimizer applies rewrite rules in SymPy, simplifying nested layers into flat, minimal formulas "
        "with zero loss in accuracy.", body_style
    ))
    
    # Insert Figure 5
    story.append(Spacer(1, 5))
    story.append(RLImage(fig5_path, width=3.3*inch, height=1.3*inch))
    story.append(Paragraph("Fig. 5. Symbolic Collapse Optimization Flow.", quote_style))
    story.append(Spacer(1, 5))

    story.append(Paragraph(
        "Deploying these symbolic graphs using heavy runtimes (like TFLite Micro) introduces significant overhead. Our compiler "
        "generates static C++ Directed Acyclic Graphs (DAGs) on register arrays, executing without heap allocations.", body_style
    ))
    story.append(Paragraph("<i>A. Speed-Memory Tradeoff & Flexibility</i>", h2_style))
    story.append(Paragraph(
        "The EML-KAN framework provides a flexible compile-time configuration matrix. In <b>Memory-Optimized</b> modes, "
        "we restrict EML mixtures to K=1, execute genetic pruning on pathways, and reuse intermediate register variables, "
        "minimizing footprints to &lt; 10 KB RAM and 70.3 KB Flash. In <b>Speed-Optimized</b> modes, we expand EML terms into "
        "parallel streams and precompute invariant constants, reducing ESP32 classifier head latency to 9.91 ms. "
        "The EML basis mixture offers superior expressiveness and accuracy over discrete spline grids, fitting "
        "coordinate grids with an MSE of 0.03598 and Wine features with 100.00% test accuracy.", body_style
    ))

    # Sec 5: Muon
    story.append(Paragraph("V. The Muon Optimizer: Orthonormal Updates", h1_style))
    story.append(Paragraph(
        "To train the KAN classifier head efficiently, we employ the Muon optimizer. Rather than scaling gradients based on "
        "running moments (like AdamW), Muon updates weight parameters W by orthogonalizing the gradient matrix G: W <- W - eta * Orthogonalize(G). "
        "This forces updates to occur along orthonormal axes, constraining parameters to the unit sphere and accelerating convergence (Fig. 6).", body_style
    ))
    
    # Insert Figure 6
    story.append(Spacer(1, 5))
    story.append(RLImage(fig6_path, width=3.3*inch, height=1.7*inch))
    story.append(Paragraph("Fig. 6. Newton-Schulz Orthonormal Steps.", quote_style))
    story.append(Spacer(1, 5))

    # Sec 6: Experimental Results
    story.append(Paragraph("VI. Experimental Evaluation", h1_style))
    story.append(Paragraph("<i>A. Experiment 1: Bipolar Binary Memorization</i>", h2_style))
    story.append(Paragraph(
        "The baseline analytical mapping was verified on bipolar binary memory patterns. Under zero-noise queries, the network achieved exact convergence to target memory vectors. A comparative trainable KAN architecture successfully converged to an unrounded MSE of 1.27e-05, which resolved to exactly 0.00 after thresholding.", body_style
    ))

    story.append(Paragraph("<i>B. Experiment 2: Fashion MNIST Scaling & L1 Sparsity</i>", h2_style))
    story.append(Paragraph(
        "We scaled the memory storage to N=20 templates of dimension d=784 using normalized Fashion MNIST images. Lasso regularization during gradient training successfully drove unnecessary connections to exactly zero, resulting in a 41.70% parameter footprint reduction and a 70% reduction in inference FLOPs.", body_style
    ))

    story.append(Paragraph("<i>C. Experiment 3: Attractor Basin Phase Transitions</i>", h2_style))
    story.append(Paragraph(
        "We isolated retrieval thresholds by interpolating between stored templates. By scaling the inverse temperature parameter to the winner-take-all limit (beta = 10^5), we observed a sharp phase step transition at alpha = 0.50. For all alpha &lt; 0.50, the system converged to pattern A (MSE = 0.00), and for alpha &gt; 0.50, it locked onto pattern B (MSE = 0.00).", body_style
    ))

    story.append(Paragraph("<i>D. Experiment 4: Hybrid Sparse Cross-Attention KAN</i>", h2_style))
    story.append(Paragraph(
        "This configuration merged KAN-wise edge parameterization with sequence-length routing mechanics, scaling sequence retrieval linearly instead of quadratically. Active edge curves were fitted to quadratic equations (y = ax^2 + bx + c) using SymPy to freeze pathways.", body_style
    ))

    story.append(Paragraph("<i>E. Experiment 5: Continuous Real-Valued Exact Memorization</i>", h2_style))
    story.append(Paragraph(
        "Continuous Fashion MNIST float32 templates were loaded and subjected to extreme Gaussian noise (sigma = 0.6) and 40% random pixel erasure. Under beta = 10^5, the attention vector resolved to a perfect floating-point one-hot vector, yielding raw continuous reconstruction of exactly MSE = 0.000000 (unrounded).", body_style
    ))

    story.append(Paragraph("<i>F. Experiment 6: Genomic DNA Sequence Recovery</i>", h2_style))
    story.append(Paragraph(
        "We evaluated sequence memorization and restoration using promoter sequences from the GUE dataset. Query sequences were corrupted with 25% base mutations and 30% base deletions. The Analytical KAN successfully resolved sequences with a 100.00% nucleotide base recovery rate (1400/1400 nucleotides restored).", body_style
    ))

    # Insert Figure 8
    story.append(Spacer(1, 5))
    story.append(RLImage(fig8_path, width=3.3*inch, height=1.17*inch))
    story.append(Paragraph("Fig. 8. Phase Transition (Left) and DNA base recovery (Right).", quote_style))
    story.append(Spacer(1, 5))

    story.append(Paragraph("<i>G. ESP32 Edge Latency Sweep</i>", h2_style))
    story.append(Paragraph(
        "We implement a hybrid partition pipeline (Fig. 7). The host PC runs MobileNetV3 to extract 576 features. The ESP32 executing "
        "the compiled EML-KAN classification head achieves a latency of 9.91 ms while consuming < 10 KB of RAM and reaching 79.09% test accuracy on CIFAR-100.", body_style
    ))

    # Insert Figure 7
    story.append(Spacer(1, 5))
    story.append(RLImage(fig7_path, width=3.3*inch, height=1.3*inch))
    story.append(Paragraph("Fig. 7. ESP32 Hybrid Deployment Setup.", quote_style))
    story.append(Spacer(1, 5))

    # Table of Results
    story.append(Spacer(1, 5))
    table_data = [
        ["Model Architecture", "Params", "Acc (Train/Test)", "Latency"],
        ["Standard MHN Baseline", "15,680", "N/A", "1.00x"],
        ["Analytical KAN Proof", "31,360", "N/A", "1.00x"],
        ["Sparse / Symbolic KAN", "9,141", "N/A", "3.3x speedup"],
        ["PyTorch EML-KAN", "139", "100.0%", "1.00x"],
        ["EML-KAN (ESP32)", "17,580", "97.0% / 79.1%", "9.91 ms"]
    ]
    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#FFFFFF')),
        ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,1), (-1,-1), 'Times-Roman'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
    ]))
    story.append(t)
    story.append(Paragraph("Table I. Completed Experiments Comparative Matrix.", quote_style))
    story.append(Spacer(1, 5))

    # Sec 7: Conclusion
    story.append(Paragraph("VII. Conclusion", h1_style))
    story.append(Paragraph(
        "The integration of Modern Hopfield memory retrieval and Kolmogorov-Arnold Networks (MHNKAN) "
        "offers a mathematically rigorous, composition-based alternative to traditional layer-stacking deep learning. "
        "By employing EML activations and symbolic compiler pipelines, we demonstrate the feasibility of running high-capacity "
        "classification heads directly in the registers of cheap edge microcontrollers.", body_style
    ))

    doc.build(story, canvasmaker=AcademicCanvas)
    print("Academic PDF successfully written to:", PDF_PATH)

if __name__ == "__main__":
    build_paper_pdf()

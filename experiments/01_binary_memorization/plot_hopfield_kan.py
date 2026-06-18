import os
import sys
# Injected path for root and core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core')))

import os
import matplotlib.pyplot as plt
import numpy as np

def draw_hopfield_kan():
    # Setup plotting area
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.axis('off')
    
    # Coordinates of layers
    # x values for: 
    # 1. Inputs (Query q)
    # 2. Similarity Sum (S_j)
    # 3. Softmax Attention Weights (w_j)
    # 4. Outputs (Reconstructed y)
    x_coords = [0.1, 0.4, 0.7, 0.95]
    
    # Sizes for visualization
    d = 4  # Simplify query dimensions to 4 for clean plotting
    N = 3  # Simplify stored patterns to 3
    
    node_positions = {
        'q': [(x_coords[0], y) for y in np.linspace(0.25, 0.75, d)],
        'S': [(x_coords[1], y) for y in np.linspace(0.3, 0.7, N)],
        'w': [(x_coords[2], y) for y in np.linspace(0.3, 0.7, N)],
        'y': [(x_coords[3], y) for y in np.linspace(0.25, 0.75, d)]
    }
    
    colors = ['#4F46E5', '#06B6D4', '#10B981']
    
    # 1. Draw inputs (q) to similarity sums (S)
    for j, (xs, ys) in enumerate(node_positions['S']):
        for k, (xq, yq) in enumerate(node_positions['q']):
            # Draw line
            ax.plot([xq, xs], [yq, ys], color='#D1D5DB', lw=1.2, zorder=1)
            # Annotate mid-edge activation function symbol
            # phi_{j,k}(q_k) = beta * X_{j,k} * q_k
            if k == 0:
                ax.text(0.7*xq + 0.3*xs, 0.7*yq + 0.3*ys, f"$\\phi_{{j,k}}$", 
                        fontsize=7, ha='center', color='#4B5563', 
                        bbox=dict(facecolor='white', edgecolor='none', pad=1))

    # 2. Draw similarity sums (S) to attention weights (w) (Softmax/Attn Step)
    for j in range(N):
        xs, ys = node_positions['S'][j]
        xw, yw = node_positions['w'][j]
        # Draw connection representing exp / normalization
        ax.plot([xs, xw], [ys, yw], color='#9CA3AF', lw=1.8, linestyle='--', zorder=1)
        # Add normalization/division block visual label
        ax.text(0.5*(xs + xw), ys + 0.02, "$\exp$ / $\\sum$", fontsize=8, ha='center', color='#111827')

    # 3. Draw attention weights (w) to output nodes (y)
    for i, (xy, yy) in enumerate(node_positions['y']):
        for j, (xw, yw) in enumerate(node_positions['w']):
            ax.plot([xw, xy], [yw, yy], color='#D1D5DB', lw=1.2, zorder=1)
            if j == 0:
                ax.text(0.4*xw + 0.6*xy, 0.4*yw + 0.6*yy, f"$\\psi_{{i,j}}$", 
                        fontsize=7, ha='center', color='#4B5563', 
                        bbox=dict(facecolor='white', edgecolor='none', pad=1))
                
    # Draw Nodes & Labels
    # Inputs (q)
    for k, (xq, yq) in enumerate(node_positions['q']):
        circle = plt.Circle((xq, yq), 0.025, color='#374151', zorder=3)
        ax.add_patch(circle)
        ax.text(xq - 0.04, yq, f"$q_{k+1}$", ha='right', va='center', fontsize=11, fontweight='bold', color='#111827')
        
    # Similarity Nodes (S_j)
    for j, (xs, ys) in enumerate(node_positions['S']):
        circle = plt.Circle((xs, ys), 0.03, color='#1E3A8A', zorder=3)
        ax.add_patch(circle)
        ax.text(xs, ys, f"$S_{j+1}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#FFFFFF')
        ax.text(xs, ys - 0.06, f"$\\sum_k \\beta X_{{{j+1},k}} q_k$", ha='center', va='top', fontsize=8, color='#1E3A8A')
        
    # Attention Weight Nodes (w_j)
    for j, (xw, yw) in enumerate(node_positions['w']):
        # Highlight winner-take-all node with a different color/size if beta is high
        color = '#10B981' if j == 0 else '#6B7280' # Pattern 1 is the retrieved memory
        circle = plt.Circle((xw, yw), 0.03, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(xw, yw, f"$w_{j+1}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#FFFFFF')
        
        # Label weight value
        val_str = "1.0 (Winner)" if j == 0 else "0.0"
        ax.text(xw, yw - 0.06, val_str, ha='center', va='top', fontsize=8, fontweight='bold', color=color)

    # Outputs (y)
    for i, (xy, yy) in enumerate(node_positions['y']):
        circle = plt.Circle((xy, yy), 0.025, color='#059669', zorder=3)
        ax.add_patch(circle)
        ax.text(xy + 0.04, yy, f"$y_{i+1}$", ha='left', va='center', fontsize=11, fontweight='bold', color='#111827')
        ax.text(xy, yy - 0.05, f"$X_{{1,{i+1}}}$", ha='center', va='top', fontsize=8, color='#059669')

    # Draw titles of stages
    ax.text(x_coords[0], 0.85, "Input Query\n(Noisy Pattern)", ha='center', va='center', fontsize=11, fontweight='bold', color='#374151')
    ax.text(x_coords[1], 0.85, "Similarity Layer\n(Edge $\\phi_{j,k}$)", ha='center', va='center', fontsize=11, fontweight='bold', color='#1E3A8A')
    ax.text(x_coords[2], 0.85, "Softmax Weights\n(WTA Limit $\\beta \\to \\infty$)", ha='center', va='center', fontsize=11, fontweight='bold', color='#10B981')
    ax.text(x_coords[3], 0.85, "Reconstruction\n(Edge $\\psi_{i,j}$)", ha='center', va='center', fontsize=11, fontweight='bold', color='#059669')

    plt.title("Exact Analytical Hopfield KAN Computation Graph (MSE = 0)\nDemonstrating reconstruction through a Winner-Take-All similarity layout", fontsize=13, fontweight='bold', pad=30)
    
    artifact_dir = r"C:\Users\karthikkrazy\.gemini\antigravity\brain\b61fde41-981b-4214-ae72-96441b49d932"
    os.makedirs(artifact_dir, exist_ok=True)
    save_path = os.path.join(artifact_dir, "hopfield_kan_exact_graph.png")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Hopfield KAN exact visual saved to {save_path}")

if __name__ == "__main__":
    draw_hopfield_kan()

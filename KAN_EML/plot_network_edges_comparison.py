import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_network_with_curves(ax, title, is_eml=False):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.3, title, fontsize=12, fontweight='bold', ha='center', color='#1E293B')
    
    # Layer layout: [2, 3, 1]
    layer_sizes = [2, 3, 1]
    positions = {
        0: [(2.0, 6.0), (2.0, 4.0)],          # Input layer nodes
        1: [(5.0, 7.0), (5.0, 5.0), (5.0, 3.0)], # Hidden layer nodes
        2: [(8.0, 5.0)]                       # Output layer node
    }
    
    # Draw edges first (so nodes are rendered on top)
    edge_color = '#94A3B8'
    curve_color = '#7C3AED' if is_eml else '#E11D48'
    
    # Perpendicular offset plotter
    for l in range(len(layer_sizes) - 1):
        src_nodes = positions[l]
        dst_nodes = positions[l+1]
        
        for s_idx, src in enumerate(src_nodes):
            for d_idx, dst in enumerate(dst_nodes):
                x1, y1 = src
                x2, y2 = dst
                
                # Draw the straight connection line as base
                ax.plot([x1, x2], [y1, y2], color=edge_color, lw=1.0, linestyle='--', alpha=0.7)
                
                # Generate curve points along the edge
                t = np.linspace(0.15, 0.85, 100)
                # Compute coordinates along the straight line segment
                x_line = (1 - t) * x1 + t * x2
                y_line = (1 - t) * y1 + t * y2
                
                # Perpendicular direction
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                nx = -dy / length
                ny = dx / length
                
                # Define different function profiles based on edge index to look realistic
                seed_val = s_idx * 3 + d_idx
                np.random.seed(seed_val)
                
                if is_eml:
                    # EML: Smooth exponential growth or log curves
                    if seed_val % 2 == 0:
                        # Exp curve
                        f = 0.45 * (np.exp(2.0 * t - 1.0) - t)
                    else:
                        # Log/Softplus curve
                        f = -0.4 * (np.log(t + 0.1) + 0.5)
                else:
                    # Standard KAN: Bumpy multi-modal spline grid curves
                    if seed_val % 2 == 0:
                        # Bumpy spline wave
                        f = 0.3 * np.sin(2.0 * np.pi * t) + 0.1 * np.cos(5.0 * np.pi * t)
                    else:
                        # Spline bump
                        f = 0.45 * np.exp(-((t - 0.5)/0.15)**2)
                
                # Apply perpendicular offset scaling
                x_curve = x_line + f * nx
                y_curve = y_line + f * ny
                
                # Plot the curve on top of the edge
                ax.plot(x_curve, y_curve, color=curve_color, lw=2.0)
                
    # Draw Nodes
    node_radius = 0.45
    # Input nodes
    for idx, (x, y) in enumerate(positions[0]):
        node = patches.Circle((x, y), node_radius, edgecolor='#2563EB', facecolor='#DBEAFE', linewidth=2, zorder=4)
        ax.add_patch(node)
        ax.text(x, y, f"$x_{idx+1}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#1E3A8A', zorder=5)
        
    # Hidden nodes
    for idx, (x, y) in enumerate(positions[1]):
        node = patches.Circle((x, y), node_radius, edgecolor='#475569', facecolor='#F1F5F9', linewidth=2, zorder=4)
        ax.add_patch(node)
        ax.text(x, y, f"$h_{idx+1}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#334155', zorder=5)
        
    # Output nodes
    for idx, (x, y) in enumerate(positions[2]):
        node = patches.Circle((x, y), node_radius, edgecolor='#10B981', facecolor='#D1FAE5', linewidth=2, zorder=4)
        ax.add_patch(node)
        ax.text(x, y, f"$y_{idx+1}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#065F46', zorder=5)

def main():
    print("Generating network edge comparison plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    plt.subplots_adjust(wspace=0.15)
    
    # Left Panel: Standard KAN [2, 3, 1] with spline edge curves
    draw_network_with_curves(axes[0], "Standard KAN: Bumpy Spline Curves on Edges", is_eml=False)
    
    # Right Panel: EML-KAN [2, 3, 1] with EML edge curves
    draw_network_with_curves(axes[1], "EML-KAN: Smooth Exp-Log Curves on Edges", is_eml=True)
    
    plt.suptitle("KAN Edge Comparison: Univariate Functions Plotted on Connection Edges", fontsize=15, fontweight='bold', color='#0F172A', y=0.98)
    
    dest_path = os.path.join(os.path.dirname(__file__), "kan_network_edges_comparison.png")
    plt.savefig(dest_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Network comparison plot successfully saved to: {dest_path}")

if __name__ == "__main__":
    main()

import os
import sys
# Injected path for root and core imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core')))

import os
import matplotlib.pyplot as plt
import numpy as np

def draw_kan_network():
    # Architecture dimensions
    layer_sizes = [4, 3, 2]
    num_layers = len(layer_sizes)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.axis('off')
    
    # Coordinates of nodes
    # Layer spacing along X-axis
    x_coords = np.linspace(0.1, 0.9, num_layers)
    
    node_positions = {}
    for l, size in enumerate(layer_sizes):
        # Center nodes vertically along Y-axis
        y_coords = np.linspace(0.2, 0.8, size)
        node_positions[l] = [(x_coords[l], y) for y in y_coords]
        
    # Draw connections (edges with activation function plots in the middle)
    np.random.seed(42)
    colors = ['#FF4E50', '#FC913A', '#45B6FE', '#37A124']
    
    # We will draw the edges
    for l in range(num_layers - 1):
        curr_nodes = node_positions[l]
        next_nodes = node_positions[l+1]
        
        for i, (x_start, y_start) in enumerate(curr_nodes):
            for j, (x_end, y_end) in enumerate(next_nodes):
                # Draw main connection line with low opacity
                ax.plot([x_start, x_end], [y_start, y_end], color='#CCCCCC', lw=1.5, alpha=0.6, zorder=1)
                
                # Plot a mini activation function in the middle of the edge
                # Midpoint of connection
                x_mid = 0.6 * x_start + 0.4 * x_end
                y_mid = 0.6 * y_start + 0.4 * y_end
                
                # Mini curve coordinates
                x_curve = np.linspace(-0.03, 0.03, 20)
                # Random looking RBF/sine activation function curve
                curve_type = np.random.choice(['rbf', 'quadratic', 'sine'])
                if curve_type == 'rbf':
                    y_curve = 0.02 * np.exp(-((x_curve) / 0.015)**2)
                elif curve_type == 'quadratic':
                    y_curve = 0.02 * (x_curve / 0.03)**2
                else:
                    y_curve = 0.02 * np.sin(x_curve * 100)
                    
                # Rotate and translate curve to align with connection angle
                angle = np.arctan2(y_end - y_start, x_end - x_start)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                x_rot = x_curve * cos_a - y_curve * sin_a + x_mid
                y_rot = x_curve * sin_a + y_curve * cos_a + y_mid
                
                edge_color = colors[(i + j) % len(colors)]
                ax.plot(x_rot, y_rot, color=edge_color, lw=2.0, zorder=2)
                
    # Draw nodes as circles
    for l, pos_list in node_positions.items():
        for i, (x, y) in enumerate(pos_list):
            circle = plt.Circle((x, y), 0.03, color='#1F2937', zorder=3)
            ax.add_patch(circle)
            inner_circle = plt.Circle((x, y), 0.025, color='#F3F4F6', zorder=4)
            ax.add_patch(inner_circle)
            
            # Label nodes
            ax.text(x, y, f"$x^{l}_{i+1}$", ha='center', va='center', fontsize=9, zorder=5, color='#1F2937', weight='bold')
            
    # Add title and labels
    plt.title("Kolmogorov-Arnold Network (KAN) Architecture\n(Edge-based Learnable 1D Activation Functions)", fontsize=14, fontweight='bold', pad=20, color='#1F2937')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='#CCCCCC', lw=1.5, label='Connection Edges'),
        plt.Line2D([0], [0], color='#FF4E50', lw=2, label='1D Activation Functions $\phi_{i,j}(x)$')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), frameon=True, facecolor='#F9FAFB')
    
    artifact_dir = r"C:\Users\karthikkrazy\.gemini\antigravity\brain\b61fde41-981b-4214-ae72-96441b49d932"
    os.makedirs(artifact_dir, exist_ok=True)
    save_path = os.path.join(artifact_dir, "kan_network_visualization.png")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"KAN network visualization saved to {save_path}")

if __name__ == "__main__":
    draw_kan_network()

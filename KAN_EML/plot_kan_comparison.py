import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    print("Generating schematic comparison of Standard KAN vs. EML-KAN...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    # -------------------------------------------------------------------------
    # Panel 1: Standard KAN Edge (Grid Spline basis)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9.3, "Standard KAN Connection (Grid-based Splines)", fontsize=13, fontweight='bold', ha='center', color='#1E293B')
    
    # Draw input node
    input_circle = patches.Circle((1.5, 5), 0.6, edgecolor='#2563EB', facecolor='#DBEAFE', linewidth=2)
    ax1.add_patch(input_circle)
    ax1.text(1.5, 5, "Input\n$x$", ha='center', va='center', fontsize=10, fontweight='bold', color='#1E3A8A')
    
    # Draw output node
    output_circle = patches.Circle((8.5, 5), 0.6, edgecolor='#10B981', facecolor='#D1FAE5', linewidth=2)
    ax1.add_patch(output_circle)
    ax1.text(8.5, 5, "Output\n$y$", ha='center', va='center', fontsize=10, fontweight='bold', color='#065F46')
    
    # Spline Grid Block (Middle)
    grid_rect = patches.FancyBboxPatch((4.0, 2.5), 2.0, 5.0, boxstyle="round,pad=0.2", edgecolor='#475569', facecolor='#F1F5F9', linewidth=1.5)
    ax1.add_patch(grid_rect)
    
    # Draw connections
    ax1.annotate("", xy=(3.8, 5), xytext=(2.2, 5), arrowprops=dict(arrowstyle="->", lw=1.5, color='#475569'))
    ax1.annotate("", xy=(7.8, 5), xytext=(6.2, 5), arrowprops=dict(arrowstyle="->", lw=1.5, color='#475569'))
    
    # Internal grid representation
    ax1.text(5.0, 7.0, "Spline Basis", fontsize=11, fontweight='bold', ha='center', color='#334155')
    ax1.text(5.0, 6.2, "SiLU base: $w_b \cdot x$", fontsize=9, ha='center', color='#475569')
    # Draw grid points
    ax1.text(5.0, 5.2, "Discrete Grid Points:", fontsize=9, ha='center', color='#475569')
    for idx, gp in enumerate([4.3, 4.6, 5.0, 5.4, 5.7]):
        pt = patches.Circle((gp, 4.6), 0.08, edgecolor='#E11D48', facecolor='#F87171')
        ax1.add_patch(pt)
    ax1.text(5.0, 3.8, "Grid Weights: $w_0 ... w_G$\n(Parameters $\propto G$)", fontsize=9, ha='center', va='center', color='#475569')
    
    # -------------------------------------------------------------------------
    # Panel 2: EML-KAN Edge (Analytic EML mixture)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9.3, "EML-KAN Connection (Analytic EML Primitive)", fontsize=13, fontweight='bold', ha='center', color='#1E293B')
    
    # Draw input node
    input_circle2 = patches.Circle((1.5, 5), 0.6, edgecolor='#2563EB', facecolor='#DBEAFE', linewidth=2)
    ax2.add_patch(input_circle2)
    ax2.text(1.5, 5, "Input\n$x$", ha='center', va='center', fontsize=10, fontweight='bold', color='#1E3A8A')
    
    # Draw output node
    output_circle2 = patches.Circle((8.5, 5), 0.6, edgecolor='#10B981', facecolor='#D1FAE5', linewidth=2)
    ax2.add_patch(output_circle2)
    ax2.text(8.5, 5, "Output\n$y$", ha='center', va='center', fontsize=10, fontweight='bold', color='#065F46')
    
    # EML Block (Middle)
    eml_rect = patches.FancyBboxPatch((4.0, 2.5), 2.0, 5.0, boxstyle="round,pad=0.2", edgecolor='#7C3AED', facecolor='#F3E8FF', linewidth=1.5)
    ax2.add_patch(eml_rect)
    
    # Draw connections
    ax2.annotate("", xy=(3.8, 5), xytext=(2.2, 5), arrowprops=dict(arrowstyle="->", lw=1.5, color='#475569'))
    ax2.annotate("", xy=(7.8, 5), xytext=(6.2, 5), arrowprops=dict(arrowstyle="->", lw=1.5, color='#475569'))
    
    # Internal EML details
    ax2.text(5.0, 7.0, "EML Primitive", fontsize=11, fontweight='bold', ha='center', color='#6B21A8')
    ax2.text(5.0, 6.0, "$\operatorname{eml}(u, v) = \exp(u) - \ln(v)$", fontsize=9, fontweight='bold', ha='center', color='#7C3AED')
    ax2.text(5.0, 5.0, "$u = a \cdot x + b$\n$v = \\text{softplus}(c \cdot x + d) + \\epsilon$", fontsize=9, ha='center', va='center', color='#581C87')
    ax2.text(5.0, 3.8, "Parameters:\n$w_{eml}, a, b, c, d$\n(Fixed size: 5 params)", fontsize=9, ha='center', va='center', color='#581C87')
    
    plt.suptitle("KAN Edge Architecture Comparison: Grid-based Splines vs. EML", fontsize=16, fontweight='bold', color='#0F172A', y=0.98)
    
    dest_path = os.path.join(os.path.dirname(__file__), "kan_comparison.png")
    plt.savefig(dest_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot successfully saved to: {dest_path}")

if __name__ == "__main__":
    main()

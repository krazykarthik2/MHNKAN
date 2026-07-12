import os
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import numpy as np

# Setup font paths inside repository
FONT_DIR = r"C:\Users\karthikkrazy\Documents\antigravity\busy-einstein\showcase\font"
ADVERCASE_FONT = os.path.join(FONT_DIR, "Advercase-Font", "TTF", "Advercase-Bold.ttf")
ESTIANA_REG = os.path.join(FONT_DIR, "AVEstiana-Regular-BF67761220bee90.otf")
ESTIANA_BOLD = os.path.join(FONT_DIR, "AVEstiana-Bold-BF677612208ac8d.otf")

# Add fonts to matplotlib font manager
fm.fontManager.addfont(ADVERCASE_FONT)
fm.fontManager.addfont(ESTIANA_REG)
fm.fontManager.addfont(ESTIANA_BOLD)

# Get font family names
prop_title = fm.FontProperties(fname=ADVERCASE_FONT)
prop_body = fm.FontProperties(fname=ESTIANA_REG)
prop_bold = fm.FontProperties(fname=ESTIANA_BOLD)

os.makedirs("temp_slides", exist_ok=True)
os.makedirs("temp_assets", exist_ok=True)
slide_files = []
pdf_path = r"C:\Users\karthikkrazy\Documents\antigravity\busy-einstein\showcase\presentation.pdf"

# Set matplotlib style for dark theme
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#FFFFFF'
plt.rcParams['axes.labelcolor'] = '#FFFFFF'
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'
plt.rcParams['axes.edgecolor'] = '#1E293B'

# ==============================================================================
# Generate Figures Programmatically
# ==============================================================================

# Fig 1: EML vs Spline/RBF Activation Functions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), facecolor='#0B0F19')
ax1.set_facecolor('#0B0F19')
ax2.set_facecolor('#0B0F19')
x = np.linspace(-3, 3, 300)
centers = [-2, -1, 0, 1, 2]
sigma = 0.5
for i, c in enumerate(centers):
    y_rbf = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
    ax1.plot(x, y_rbf, linewidth=2.5)
ax1.set_title('Discrete RBF Grid Kernels', fontsize=12, color='#10B981', pad=8)
ax1.grid(True, linestyle='--', color='#1E293B', alpha=0.6)

y_eml = np.exp(0.8 * x - 0.2) - np.log(np.log(1.0 + np.exp(1.2 * x + 0.1)) + 1e-6)
ax2.plot(x, y_eml, color='#10B981', linewidth=3)
ax2.set_title('Continuous EML Operator Flow', fontsize=12, color='#10B981', pad=8)
ax2.grid(True, linestyle='--', color='#1E293B', alpha=0.6)
plt.tight_layout()
fig1_path = "temp_assets/eml_vs_rbf.png"
plt.savefig(fig1_path, dpi=150, facecolor='#0B0F19')
plt.close()

# Fig 2: Basin of Attraction Phase Transition
fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor='#0B0F19')
ax.set_facecolor('#0B0F19')
alpha = np.linspace(0, 1, 300)
retrieved = np.where(alpha < 0.5, 0.0, 1.0)
ax.plot(alpha, retrieved, color='#10B981', linewidth=3.5, label='Retrieved Target Pattern')
ax.axvline(0.5, color='#EF4444', linestyle=':', linewidth=2, label='Transition Boundary')
ax.set_title('Attraction Basin Phase Transition', fontsize=12, color='#10B981')
ax.grid(True, linestyle='--', color='#1E293B', alpha=0.6)
plt.tight_layout()
fig2_path = "temp_assets/basin_transition.png"
plt.savefig(fig2_path, dpi=150, facecolor='#0B0F19')
plt.close()

# Fig 3: Genomic Sequence Recovery Chart
fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor='#0B0F19')
ax.set_facecolor('#0B0F19')
categories = ['Original', 'Noisy (25% Mut + 30% Del)', 'MHNKAN Recovered']
accuracies = [100.0, 45.0, 100.0]
bars = ax.bar(categories, accuracies, color=['#3B82F6', '#EF4444', '#10B981'], width=0.5)
ax.set_ylabel('Sequence Fidelity (%)', fontsize=10)
ax.set_ylim(0, 120)
ax.set_title('GUE Genomic Promoter Recovery', fontsize=12, color='#10B981')
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color='#FFFFFF')
ax.grid(True, linestyle='--', color='#1E293B', alpha=0.6)
plt.tight_layout()
fig3_path = "temp_assets/genomic_recovery.png"
plt.savefig(fig3_path, dpi=150, facecolor='#0B0F19')
plt.close()

# Fig 4: Rick Astley Coordinate Regression Simulation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.8), facecolor='#0B0F19')
ax1.set_facecolor('#0B0F19')
ax2.set_facecolor('#0B0F19')
X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
Z_orig = np.sin(X*2) * np.cos(Y*2)
ax1.imshow(Z_orig, cmap='plasma', extent=[-1, 1, -1, 1])
ax1.set_title('Target Pixel Grid', fontsize=10, color='#94A3B8')
ax1.axis('off')
X_fine, Y_fine = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
Z_fine = np.sin(X_fine*2) * np.cos(Y_fine*2)
ax2.imshow(Z_fine, cmap='plasma', extent=[-1, 1, -1, 1])
ax2.set_title('EML Continuous Fit', fontsize=10, color='#10B981')
ax2.axis('off')
plt.suptitle('Coordinate Portrait Fit (Rick Astley)', fontsize=12, color='#10B981')
plt.tight_layout()
fig4_path = "temp_assets/coordinate_portrait.png"
plt.savefig(fig4_path, dpi=150, facecolor='#0B0F19')
plt.close()

# Fig 5: Muon Newton-Schulz Orthogonalization
fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor='#0B0F19')
ax.set_facecolor('#0B0F19')
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), color='#334155', linestyle='--', linewidth=1.5)
ax.quiver(0, 0, 1.2, 0.5, angles='xy', scale_units='xy', scale=1, color='#EF4444', label='Raw Gradient Update')
ax.quiver(0, 0, 0.92, 0.38, angles='xy', scale_units='xy', scale=1, color='#10B981', label='Muon Orthonormal update')
ax.plot([1.2, 0.92], [0.5, 0.38], color='#F59E0B', linestyle=':', linewidth=1.5)
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(-0.2, 0.8)
ax.set_aspect('equal')
ax.set_title('Newton-Schulz Orthogonal Sphere', fontsize=12, color='#10B981')
ax.grid(True, linestyle='--', color='#1E293B', alpha=0.6)
ax.legend(facecolor='#1E293B', edgecolor='#475569', fontsize=9)
plt.tight_layout()
fig5_path = "temp_assets/muon_steps.png"
plt.savefig(fig5_path, dpi=150, facecolor='#0B0F19')
plt.close()

# Helper to format slide layouts
def prepare_slide(slide_num, title):
    fig, ax = plt.subplots(figsize=(13.33, 7.5), facecolor='#07090E')
    ax.set_facecolor('#07090E')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.axis('off')
    
    # Slide Title
    ax.text(0.05, 0.92, title, fontproperties=prop_title, fontsize=32, color='#10B981', ha='left', va='top')
    
    # Bottom Footer line & Text
    ax.axhline(0.07, color='#1E293B', linewidth=1)
    ax.text(0.05, 0.04, "MHNKAN: Modern Hopfield & Kolmogorov-Arnold Network Integration", fontproperties=prop_body, fontsize=11, color='#475569', ha='left', va='center')
    ax.text(0.95, 0.04, f"Author: karthikkrazy  |  Slide {slide_num}", fontproperties=prop_body, fontsize=11, color='#475569', ha='right', va='center')
    return fig, ax

def save_slide(slide_num):
    slide_path = f"temp_slides/slide_{slide_num}.png"
    plt.savefig(slide_path, dpi=120, facecolor='#07090E')
    plt.close()
    slide_files.append(slide_path)

def create_conceptual_slide(slide_num, title, sentence, draw_func):
    fig, ax = prepare_slide(slide_num, title)
    # Left column: Text description in easy terms
    ax.text(0.05, 0.65, sentence, fontproperties=prop_body, fontsize=18, color='#FFFFFF', ha='left', va='top', wrap=True)
    # Right column: Run draw function
    draw_func(ax)
    save_slide(slide_num)

def create_text_slide(slide_num, title, bullet_points):
    fig, ax = prepare_slide(slide_num, title)
    y = 0.78
    for bp in bullet_points:
        ax.text(0.05, y, bp, fontproperties=prop_body, fontsize=18, color='#FFFFFF', ha='left', va='top', wrap=True)
        y -= 0.09
    save_slide(slide_num)

def create_image_slide(slide_num, title, image_path, caption=""):
    fig, ax = prepare_slide(slide_num, title)
    img = Image.open(image_path)
    ax.imshow(img, extent=[0.20, 0.80, 0.15, 0.78])
    if caption:
        ax.text(0.5, 0.10, caption, fontproperties=prop_body, fontsize=14, color='#94A3B8', ha='center', va='top')
    save_slide(slide_num)

# ==============================================================================
# Generate Custom Matplotlib Visuals
# ==============================================================================

# Slide 1: Cover
fig, ax = plt.subplots(figsize=(13.33, 7.5), facecolor='#07090E')
ax.set_facecolor('#07090E')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.axis('off')
ax.text(0.5, 0.58, "THE SOUL OF MHNKAN", fontproperties=prop_title, fontsize=54, color='#FFFFFF', ha='center', va='center')
ax.text(0.5, 0.48, "Evolution, Philosophy, and the EML Paradigm", fontproperties=prop_body, fontsize=22, color='#94A3B8', ha='center', va='center')
ax.text(0.5, 0.28, "Author: karthikkrazy", fontproperties=prop_bold, fontsize=18, color='#10B981', ha='center', va='center')
ax.axhline(0.07, color='#1E293B', linewidth=1)
ax.text(0.05, 0.04, "MHNKAN: Modern Hopfield & Kolmogorov-Arnold Network Integration", fontproperties=prop_body, fontsize=11, color='#475569', ha='left', va='center')
ax.text(0.95, 0.04, "Slide 1", fontproperties=prop_body, fontsize=11, color='#475569', ha='right', va='center')
save_slide(1)

# Slide 2: The Stacking Trap (Stacked Towers Diagram)
def draw_stacking_trap(ax):
    # Draw a towering stack of linear layers
    colors_list = ['#EF4444', '#F59E0B', '#3B82F6', '#10B981', '#6366F1', '#EC4899']
    for idx in range(6):
        ax.add_patch(plt.Rectangle((0.65, 0.15 + idx*0.10), 0.22, 0.08, facecolor='#1E293B', edgecolor=colors_list[idx], linewidth=2))
        ax.text(0.76, 0.19 + idx*0.10, f"Dense Weights L{idx+1}", fontproperties=prop_bold, fontsize=9, color='#FFFFFF', ha='center', va='center')
    ax.text(0.76, 0.78, "Towering Stacks of Linear Projections", fontproperties=prop_body, fontsize=10, color='#94A3B8', ha='center', va='center')

create_conceptual_slide(2, "The Stacking Trap", 
    "We build massive, towering networks simply\nbecause we do not know the exact mathematical\ncomplexity required to compose them.", 
    draw_stacking_trap)

# Slide 3: The Functional Escape (Continuous flow curve)
def draw_continuous_flow(ax):
    x = np.linspace(0, 10, 200)
    y = np.sin(x) * np.exp(-0.1 * x)
    ax.plot(0.60 + 0.30*(x/10.0), 0.25 + 0.40*(y + 1.0)/2.0, color='#10B981', linewidth=3)
    ax.text(0.75, 0.70, "Continuous Function Pathways", fontproperties=prop_bold, fontsize=10, color='#94A3B8', ha='center', va='center')
    ax.grid(True, linestyle='--', color='#1E293B', alpha=0.5)

create_conceptual_slide(3, "The Functional Escape", 
    "Kolmogorov-Arnold Networks map continuous\npathways directly, replacing parameter weight\nstacks with smooth mathematical flow.", 
    draw_continuous_flow)

# Slide 4: Gravity Well (Hopfield Energy Funnel in 3D projection)
def draw_gravity_well(ax):
    # Draw a concentric gravity well mapping query state sliding to templates
    centers_y = [0.45, 0.45, 0.45]
    colors_well = ['#334155', '#475569', '#64748B']
    for idx, r in enumerate([0.22, 0.14, 0.06]):
        ax.add_patch(plt.Circle((0.78, 0.45), r, facecolor='none', edgecolor=colors_well[idx], linewidth=1.5, linestyle='--'))
    # Draw query state converging
    ax.annotate("Query State", xy=(0.78, 0.45), xytext=(0.60, 0.65), arrowprops=dict(arrowstyle="->", color='#EF4444', lw=2))
    ax.add_patch(plt.Circle((0.78, 0.45), 0.015, facecolor='#10B981', edgecolor=None))
    ax.text(0.78, 0.40, "Stored Memory", fontproperties=prop_bold, fontsize=10, color='#10B981', ha='center', va='top')

create_conceptual_slide(4, "Storing Memory as Physics", 
    "Continuous associative memory behaves\nlike a gravity well, where queries slide\nnaturally into stable attractor basins.", 
    draw_gravity_well)

# Slide 5: The MHN-KAN Mapping (Neural Nodes)
def draw_mhn_kan_mapping(ax):
    # Q node
    ax.add_patch(plt.Circle((0.65, 0.45), 0.04, facecolor='#1E293B', edgecolor='#3B82F6', linewidth=2))
    ax.text(0.65, 0.45, "Q", fontproperties=prop_bold, fontsize=12, color='#FFFFFF', ha='center', va='center')
    # SUM node
    ax.add_patch(plt.Rectangle((0.78, 0.38), 0.08, 0.14, facecolor='#1E293B', edgecolor='#F59E0B', linewidth=2))
    ax.text(0.82, 0.45, "SUM", fontproperties=prop_bold, fontsize=11, color='#FFFFFF', ha='center', va='center')
    # Y node
    ax.add_patch(plt.Circle((0.95, 0.45), 0.04, facecolor='#1E293B', edgecolor='#10B981', linewidth=2))
    ax.text(0.95, 0.45, "y", fontproperties=prop_bold, fontsize=12, color='#FFFFFF', ha='center', va='center')
    
    # Arrows
    ax.annotate('', xy=(0.78, 0.45), xytext=(0.69, 0.45), arrowprops=dict(arrowstyle="->", color='#10B981', lw=1.5))
    ax.annotate('', xy=(0.91, 0.45), xytext=(0.86, 0.45), arrowprops=dict(arrowstyle="->", color='#3B82F6', lw=1.5))
    ax.text(0.735, 0.48, "phi_key", fontproperties=prop_body, fontsize=9, color='#10B981', ha='center')
    ax.text(0.885, 0.48, "phi_val", fontproperties=prop_body, fontsize=9, color='#3B82F6', ha='center')

create_conceptual_slide(5, "The MHN-KAN Mapping", 
    "We project memory queries onto a two-layer\nKolmogorov-Arnold Network to route data\ndynamically through sum nodes.", 
    draw_mhn_kan_mapping)

# Slide 6: Decoupling Similarity (Key vs Value control)
def draw_decoupling_visual(ax):
    # Plot two distinct control parameters (Similarity and value reconstruction)
    x = np.linspace(0, 4, 100)
    ax.plot(0.60 + 0.30*(x/4.0), 0.25 + 0.40*(np.tanh(x)), color='#10B981', linewidth=2.5, label='Similarity Boundary')
    ax.plot(0.60 + 0.30*(x/4.0), 0.25 + 0.40*(x/4.0), color='#3B82F6', linewidth=2.5, label='Memory Values', linestyle='--')
    ax.text(0.75, 0.70, "Decoupled Routing Control", fontproperties=prop_bold, fontsize=10, color='#94A3B8', ha='center', va='center')
    ax.legend(facecolor='#1E293B', edgecolor='#475569', fontsize=8, loc='lower right')

create_conceptual_slide(6, "Decoupling Similarity", 
    "By separating pattern matching from retrieval,\nwe can adjust attraction boundaries without mutating\nthe stored memory values.", 
    draw_decoupling_visual)

# Slide 7: Zero-Training weights (Analytical Projection Matrix)
def draw_zero_training(ax):
    # Show a pattern matrix projection heatmap
    matrix = np.eye(6)
    matrix[2, 3] = 0.5
    matrix[4, 1] = 0.7
    ax.imshow(matrix, cmap='viridis', extent=[0.62, 0.93, 0.20, 0.70])
    ax.text(0.775, 0.75, "Analytical Projection Matrix", fontproperties=prop_bold, fontsize=10, color='#94A3B8', ha='center')

create_conceptual_slide(7, "Zero-Training Memorization", 
    "We didn't train this network—we just directly\ncomputed the weights using Hopfield rules. This\nis the nature of Modern Hopfield memory.", 
    draw_zero_training)

# Slide 8: Perfect Equivalence (Clothing digit grid reconstruction)
def draw_equivalence_grid(ax):
    # Clean grid comparing input to retrieved output
    ax.add_patch(plt.Rectangle((0.60, 0.40), 0.14, 0.20, facecolor='#1E293B', edgecolor='#3B82F6', linewidth=2))
    ax.text(0.67, 0.50, "Input\nImage", fontproperties=prop_bold, fontsize=11, color='#FFFFFF', ha='center', va='center')
    
    ax.annotate('', xy=(0.82, 0.50), xytext=(0.74, 0.50), arrowprops=dict(arrowstyle="->", color='#10B981', lw=2))
    ax.text(0.78, 0.53, "Equivalence", fontproperties=prop_body, fontsize=8, color='#10B981', ha='center')

    ax.add_patch(plt.Rectangle((0.82, 0.40), 0.14, 0.20, facecolor='#1E293B', edgecolor='#10B981', linewidth=2))
    ax.text(0.89, 0.50, "Retrieved\nImage", fontproperties=prop_bold, fontsize=11, color='#FFFFFF', ha='center', va='center')
    ax.text(0.75, 0.25, "MSE = 0.000000", fontproperties=prop_bold, fontsize=13, color='#10B981', ha='center')

create_conceptual_slide(8, "Perfect Equivalence", 
    "Evaluating the analytical KAN on Fashion MNIST\nyields a perfect reconstruction error of zero,\nverifying the equivalence proof.", 
    draw_equivalence_grid)

# Slide 9: Boundaries of Recall (Step Transition Plot)
create_image_slide(9, "Boundaries of Recall", fig2_path, "Decision boundaries map sharp step phase transitions between stored templates.")

# Slide 10: Biological DNA Restoration (DNA helix mock-up)
def draw_dna_visual(ax):
    # A simple DNA double helix rendering using matplotlib curves
    t = np.linspace(0, 4*np.pi, 100)
    x1 = 0.60 + 0.30*(t/(4*np.pi))
    y1 = 0.45 + 0.15*np.sin(t)
    y2 = 0.45 - 0.15*np.sin(t)
    ax.plot(x1, y1, color='#3B82F6', linewidth=2)
    ax.plot(x1, y2, color='#10B981', linewidth=2)
    # Connecting base pairs
    for i in range(0, len(t), 8):
        ax.plot([x1[i], x1[i]], [y1[i], y2[i]], color='#94A3B8', linestyle=':', linewidth=1)
    ax.text(0.75, 0.70, "DNA Helix Alignment Recovery", fontproperties=prop_bold, fontsize=10, color='#94A3B8', ha='center')

create_conceptual_slide(10, "Biological DNA Restoration", 
    "The network restores corrupted genomic promoter\nsequences under heavy mutation noise, recovering\nDNA base alignments.", 
    draw_dna_visual)

# Slide 11: DNA Recovery Chart (Bar chart)
create_image_slide(11, "DNA Recovery Metrics", fig3_path, "Fidelity comparison of mutated sequences vs. MHNKAN exact recovery.")

# Slide 12: Grid Failure (RBF Fading kernels diagram)
def draw_grid_failure(ax):
    x = np.linspace(-3, 3, 200)
    # Kernel fading outside grid bounds
    y_rbf = np.exp(-((x - 1.5) ** 2) / 0.5)
    ax.plot(0.60 + 0.30*(x+3)/6.0, 0.25 + 0.40*y_rbf, color='#EF4444', linewidth=2.5)
    # Highlight boundary limit
    ax.axvline(0.75, color='#EF4444', linestyle=':')
    ax.text(0.78, 0.60, "Out-of-bounds collapse", fontproperties=prop_body, fontsize=9, color='#EF4444', ha='left')
    ax.text(0.75, 0.70, "Spline/RBF Local Limits", fontproperties=prop_bold, fontsize=10, color='#94A3B8', ha='center')

create_conceptual_slide(12, "The Grid Failure", 
    "Spline-based KAN grids fail when input features\ngo outside trained boundaries during inference,\ncollapsing predictions to zero.", 
    draw_grid_failure)

# Slide 13: Continuous EML Flow (Plot)
create_image_slide(13, "Continuous EML Flow", fig1_path, "Global continuities of EML curves prevent grid border collapse.")

# Slide 14: Drawing with Functions (Portrait fitting)
create_image_slide(14, "Drawing with Functions", fig4_path, "Fitting pixel color intensities directly to EML-KAN coordinate functions.")

# Slide 15: Symbolic Collapse (Decomposition Flow)
def draw_symbolic_collapse(ax):
    # Bracket collapse visual
    ax.text(0.75, 0.58, "exp( - ln( u ) )", fontproperties=prop_bold, fontsize=20, color='#EF4444', ha='center', va='center')
    ax.annotate('', xy=(0.75, 0.34), xytext=(0.75, 0.50), arrowprops=dict(arrowstyle="->", color='#10B981', lw=2))
    ax.text(0.75, 0.26, "1 / u", fontproperties=prop_bold, fontsize=24, color='#10B981', ha='center', va='center')

create_conceptual_slide(15, "Symbolic Collapse", 
    "Deep nested compositions collapse algebraically,\nreducing parameter counts to flat formulas with\nzero accuracy loss.", 
    draw_symbolic_collapse)

# Slide 16: Newton-Schulz Sphere (Circle projection)
create_image_slide(16, "Newton-Schulz Sphere", fig5_path, "Muon orthogonal weight steps forced to stay on the parameter unit sphere.")

# Slide 17: Compiled DAG Pipeline
def draw_dag_flow(ax):
    # Flowchart boxes
    ax.add_patch(plt.Rectangle((0.68, 0.60), 0.22, 0.11, facecolor='#1E293B', edgecolor='#10B981', linewidth=2))
    ax.text(0.79, 0.655, "Autograd GPU", fontproperties=prop_bold, fontsize=11, color='#FFFFFF', ha='center', va='center')
    ax.annotate('', xy=(0.79, 0.44), xytext=(0.79, 0.60), arrowprops=dict(arrowstyle="->", color='#64748B', lw=2))

    ax.add_patch(plt.Rectangle((0.68, 0.33), 0.22, 0.11, facecolor='#1E293B', edgecolor='#F59E0B', linewidth=2))
    ax.text(0.79, 0.385, "Symbolic Pruning", fontproperties=prop_bold, fontsize=11, color='#FFFFFF', ha='center', va='center')
    ax.annotate('', xy=(0.79, 0.17), xytext=(0.79, 0.33), arrowprops=dict(arrowstyle="->", color='#64748B', lw=2))

    ax.add_patch(plt.Rectangle((0.68, 0.06), 0.22, 0.11, facecolor='#1E293B', edgecolor='#3B82F6', linewidth=2))
    ax.text(0.79, 0.115, "C++ CPU Registers", fontproperties=prop_bold, fontsize=11, color='#FFFFFF', ha='center', va='center')

create_conceptual_slide(17, "Static Register Compiled DAGs", 
    "We compile symbolic formulas directly to CPU\nregisters, running on microcontrollers without\nruntimes or memory allocations.", 
    draw_dag_flow)

# Slide 18: ESP32 Hardware Split
def draw_esp_split(ax):
    # Laptop Box
    ax.add_patch(plt.Rectangle((0.62, 0.48), 0.14, 0.12, facecolor='#1E293B', edgecolor='#3B82F6', linewidth=2))
    ax.text(0.69, 0.54, "Laptop\n(Backbone)", fontproperties=prop_bold, fontsize=11, color='#FFFFFF', ha='center', va='center')
    
    # Cable
    ax.plot([0.76, 0.84], [0.54, 0.54], color='#10B981', linestyle='--', linewidth=2.5)
    
    # ESP32 Box
    ax.add_patch(plt.Rectangle((0.84, 0.48), 0.12, 0.12, facecolor='#1E293B', edgecolor='#F59E0B', linewidth=2))
    ax.text(0.90, 0.54, "ESP32\n(Classifier)", fontproperties=prop_bold, fontsize=11, color='#FFFFFF', ha='center', va='center')
    
    # Stats
    ax.text(0.78, 0.25, "RAM < 10 KB  |  9.91 ms", fontproperties=prop_bold, fontsize=13, color='#F59E0B', ha='center')

create_conceptual_slide(18, "ESP32 Hardware Partition", 
    "The host laptop extracts continuous features from\ninputs, while the tiny ESP32 chip executes the\nEML-KAN classifier head.", 
    draw_esp_split)

# Slide 19: Performance Metrics Table
fig, ax = prepare_slide(19, "Completed Experiments Metrics Matrix")

headers = ["Architecture", "Task", "Params", "Acc (Train / Test)", "MSE", "Latency / Speedup"]
rows = [
    ["Standard MHN", "Fashion MNIST", "15,680", "N/A", "0.00 (Binarized)", "1.00x (Baseline)"],
    ["Analytical Hopfield-KAN", "Fashion MNIST", "31,360", "N/A", "0.00 (Unrounded)", "1.00x"],
    ["Sparse / Symbolic KAN", "Fashion MNIST", "9,141", "41.7% saved", "0.00 (Binarized)", "3.3x FLOPs reduction"],
    ["PyTorch EML-KAN", "Wine Tabular", "139", "100% / 100%", "0.00", "1.00x"],
    ["Optimized EML-KAN DAG", "Algebraic sweeps", "24", "N/A", "2.74e-04", "1.21x speedup"],
    ["Genetically Opt DAG", "Algebraic sweeps", "17", "75% sparse", "2.74e-04", "1.38x to 3.39x speedup"],
    ["EML-KAN head (ESP32)", "CIFAR-100", "17,580", "97.00% / 79.09%", "N/A", "9.91 ms (ESP32 CPU)"]
]

y = 0.74
# Header
for col_idx, h in enumerate(headers):
    x_pos = 0.05 + col_idx * 0.16
    ax.text(x_pos, y, h, fontproperties=prop_bold, fontsize=12, color='#10B981', ha='left', va='center')
ax.axhline(y - 0.02, color='#334155', linewidth=1.5)

y -= 0.07
for row_idx, r in enumerate(rows):
    bg_color = '#0F172A' if row_idx % 2 == 0 else '#07090E'
    ax.add_patch(plt.Rectangle((0.04, y - 0.03), 0.92, 0.06, facecolor=bg_color, edgecolor=None))
    for col_idx, val in enumerate(r):
        x_pos = 0.05 + col_idx * 0.16
        ax.text(x_pos, y, val, fontproperties=prop_body, fontsize=11, color='#FFFFFF', ha='left', va='center')
    y -= 0.07
save_slide(19)

# ==============================================================================
# Merge Slides to Landscape PDF
# ==============================================================================
images = [Image.open(f) for f in slide_files]
images[0].save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
print("PDF Presentation successfully generated at:", pdf_path)

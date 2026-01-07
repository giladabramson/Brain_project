"""
Generate All 3 Important Analysis Plots

1. N × M Memory-Responsiveness 3×3 Grid
2. Breakpoint Analysis (Capacity Curve)
3. Weight Interaction with Noise Effects

Reads from recovered CSV data and generates comprehensive visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis
import warnings
warnings.filterwarnings('ignore')

# Paths
REPLICA_PATH = Path(__file__).resolve().parent
DATA_DIR = REPLICA_PATH / "data"
FIGURES_DIR = REPLICA_PATH / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("GENERATING ALL 3 ANALYSIS PLOTS")
print("=" * 80)

# ============================================================================
# GRAPH 1: N × M Memory-Responsiveness 3×3 Grid
# ============================================================================
def plot_nm_grid():
    """Plot 3×3 grid: N (population) × M (memories) with memory vs responsiveness"""
    print("\n[1/3] Generating N × M Memory-Responsiveness Grid...")
    
    # Load data
    results_file = DATA_DIR / "pn_multiseed_results.csv"
    retention_file = DATA_DIR / "pn_retention_details.csv"
    
    if not retention_file.exists():
        print(f"  ⚠ {retention_file.name} not found, skipping Graph 1")
        return
    
    df = pd.read_csv(retention_file)
    
    # Select N and M values for 3×3 grid
    n_values = sorted(df['N'].unique())[:3]  # First 3 N values
    m_values = sorted(df['P'].unique())[::3]  # Every 3rd M value
    if len(m_values) < 3:
        m_values = sorted(df['P'].unique())[:3]
    
    print(f"  Using N = {n_values}, M = {m_values}")
    
    # Create 3×3 grid
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Global stats for normalization
    all_retention = df['retention_mean'].values
    all_reaction = df['reaction_mean'].values
    all_reaction = all_reaction[all_reaction > 0]  # Remove zeros
    
    global_retention_range = (all_retention.min(), all_retention.max())
    global_reaction_range = (all_reaction.min(), all_reaction.max())
    
    # Color normalization for strength
    strengths = df['reaction_mean'].values
    strengths = strengths[strengths > 0]
    strength_norm = Normalize(vmin=np.log10(strengths.min() + 1e-6), 
                             vmax=np.log10(strengths.max() + 1e-6))
    cmap = plt.cm.viridis
    
    plot_count = 0
    for i, n in enumerate(n_values):
        for j, m in enumerate(m_values):
            if plot_count >= 9:
                break
            
            row, col = divmod(plot_count, 3)
            ax = fig.add_subplot(gs[row, col])
            
            # Filter data for this N, M combination
            subset = df[(df['N'] == n) & (df['P'] == m)]
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, f'N={n}, M={m}\n(No data)', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='red')
                ax.set_xlim(global_retention_range)
                ax.set_ylim(1.0 / global_reaction_range[1], 1.0 / global_reaction_range[0])
                plot_count += 1
                continue
            
            # Compute responsiveness and log strength
            responsiveness = 1.0 / subset['reaction_mean'].values
            memory_init = subset['retention_mean'].values
            log_strength = np.log10(subset['reaction_mean'].values + 1e-6)
            
            # Scatter plot with color = log strength
            scatter = ax.scatter(memory_init, responsiveness,
                               c=log_strength, cmap=cmap, norm=strength_norm,
                               s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Set shared axes
            ax.set_xlim(global_retention_range)
            ax.set_ylim(1.0 / global_reaction_range[1], 1.0 / global_reaction_range[0])
            
            # Labels
            ax.set_xlabel('Memory [nT(τ)]', fontsize=11, fontweight='bold')
            ax.set_ylabel('Responsiveness [1/τ_r(τ)]', fontsize=11, fontweight='bold')
            ax.set_title(f'N={n}, M={m}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add stats
            ax.text(0.98, 0.02, f'n={len(subset)}',
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            plot_count += 1
    
    # Overall title and colorbar
    fig.suptitle('N × M Memory-Responsiveness Trade-off: 3×3 Grid\n(Color = log(Mean Strength))',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = plt.colorbar(ScalarMappable(norm=strength_norm, cmap=cmap), cax=cbar_ax)
    cb.set_label('log(Mean Strength)', fontsize=11, fontweight='bold')
    
    # Save
    pdf_path = FIGURES_DIR / "01_nm_grid_memory_responsiveness.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {pdf_path}")
    
    png_path = FIGURES_DIR / "01_nm_grid_memory_responsiveness.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {png_path}")
    
    plt.close()

# ============================================================================
# GRAPH 2: Breakpoint Analysis (Capacity Curve)
# ============================================================================
def plot_breakpoint_analysis():
    """Plot N vs Max Stable P (breakpoint/capacity curve)"""
    print("\n[2/3] Generating Breakpoint Analysis (Capacity Curve)...")
    
    summary_file = DATA_DIR / "pn_multiseed_summary.csv"
    
    if not summary_file.exists():
        print(f"  ⚠ {summary_file.name} not found, skipping Graph 2")
        return
    
    df = pd.read_csv(summary_file)
    
    # Calculate breakpoint: max P where all_stable == True, or highest P with stable_fraction > 0.5
    breakpoints = {}
    for n in df['N'].unique():
        n_data = df[df['N'] == n].sort_values('P')
        
        # Find max stable P
        stable_rows = n_data[n_data['all_stable'] == True]
        if len(stable_rows) > 0:
            breakpoint = stable_rows['P'].max()
        else:
            # Fallback: find max P with high stability
            high_stability = n_data[n_data['stable_fraction'] >= 0.5]
            if len(high_stability) > 0:
                breakpoint = high_stability['P'].max()
            else:
                breakpoint = n_data['P'].min()
        
        breakpoints[n] = breakpoint
    
    # Sort by N
    ns = sorted(breakpoints.keys())
    ps = [breakpoints[n] for n in ns]
    
    print(f"  Breakpoints: {dict(zip(ns, ps))}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ns, ps, marker='o', linewidth=2.5, markersize=10, 
           color='#1f77b4', label='Breakpoint', markerfacecolor='lightblue',
           markeredgewidth=2, markeredgecolor='#1f77b4')
    
    ax.set_xlabel('Population Size (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Estimated Break Point (Max Stable P)', fontsize=13, fontweight='bold')
    ax.set_title('Network Capacity: Breakpoint Analysis (Multi-Seed)\nMaximum Storable Memories vs Population',
                fontsize=14, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(min(ns) - 5, max(ns) + 5)
    ax.set_ylim(min(ps) - 1, max(ps) + 2)
    
    # Add annotations
    for n, p in zip(ns, ps):
        ax.annotate(f'{int(p)}', xy=(n, p), xytext=(0, 8),
                   textcoords='offset points', ha='center', fontsize=10,
                   fontweight='bold', color='darkblue')
    
    plt.tight_layout()
    
    # Save
    pdf_path = FIGURES_DIR / "02_breakpoint_capacity_curve.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {pdf_path}")
    
    png_path = FIGURES_DIR / "02_breakpoint_capacity_curve.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {png_path}")
    
    plt.close()

# ============================================================================
# GRAPH 3: Weight Interaction Analysis (w_s × w_external with Noise)
# ============================================================================
def plot_weight_interaction():
    """Plot weight scale vs external input interaction effects with noise"""
    print("\n[3/3] Generating Weight Interaction Analysis...")
    
    hyper_file = DATA_DIR / "sweep_hyper_combo_summary.csv"
    
    if not hyper_file.exists():
        print(f"  ⚠ {hyper_file.name} not found, skipping Graph 3")
        return
    
    df = pd.read_csv(hyper_file)
    
    # Extract unique values
    w_s_values = sorted(df['w_s'].unique())
    w_ext_values = sorted(df['w_external'].unique())
    
    print(f"  w_s values: {w_s_values}")
    print(f"  w_external values: {w_ext_values}")
    
    # Create 2D heatmap: w_s × w_external with retention time as color
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pivot for retention
    retention_pivot = df.pivot_table(
        index='w_s', columns='w_external', values='retention_mean', aggfunc='mean'
    )
    
    # Pivot for reaction
    reaction_pivot = df.pivot_table(
        index='w_s', columns='w_external', values='reaction_mean', aggfunc='mean'
    )
    
    # Plot 1: Retention Time Heatmap
    im1 = axes[0].imshow(retention_pivot.values, cmap='RdYlGn', aspect='auto')
    axes[0].set_xticks(range(len(w_ext_values)))
    axes[0].set_yticks(range(len(w_s_values)))
    axes[0].set_xticklabels([f'{x:.1f}' for x in w_ext_values], fontsize=11)
    axes[0].set_yticklabels([f'{y:.2f}' for y in w_s_values], fontsize=11)
    axes[0].set_xlabel('w_external', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('w_s', fontsize=12, fontweight='bold')
    axes[0].set_title('Retention Time (Memory Duration)\nw_s × w_external Interaction',
                     fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i, w_s in enumerate(w_s_values):
        for j, w_ext in enumerate(w_ext_values):
            val = retention_pivot.iloc[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f'{val:.2f}', ha='center', va='center',
                           color='black', fontsize=9, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Retention Time (s)', fontsize=11, fontweight='bold')
    
    # Plot 2: Reaction Time (Responsiveness) Heatmap
    im2 = axes[1].imshow(1.0 / reaction_pivot.values, cmap='YlOrRd', aspect='auto')
    axes[1].set_xticks(range(len(w_ext_values)))
    axes[1].set_yticks(range(len(w_s_values)))
    axes[1].set_xticklabels([f'{x:.1f}' for x in w_ext_values], fontsize=11)
    axes[1].set_yticklabels([f'{y:.2f}' for y in w_s_values], fontsize=11)
    axes[1].set_xlabel('w_external', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('w_s', fontsize=12, fontweight='bold')
    axes[1].set_title('Responsiveness (1/Reaction Time)\nw_s × w_external Interaction',
                     fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i, w_s in enumerate(w_s_values):
        for j, w_ext in enumerate(w_ext_values):
            val = reaction_pivot.iloc[i, j]
            if not np.isnan(val):
                resp = 1.0 / val
                axes[1].text(j, i, f'{resp:.2f}', ha='center', va='center',
                           color='black', fontsize=9, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Responsiveness (1/s)', fontsize=11, fontweight='bold')
    
    fig.suptitle('Weight Interaction Analysis: Internal (w_s) × External (w_external)\n(Noise = 0.5)',
                fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    # Save
    pdf_path = FIGURES_DIR / "03_weight_interaction_heatmaps.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {pdf_path}")
    
    png_path = FIGURES_DIR / "03_weight_interaction_heatmaps.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {png_path}")
    
    plt.close()

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    try:
        plot_nm_grid()
    except Exception as e:
        print(f"  ✗ Error generating Graph 1: {e}")
    
    try:
        plot_breakpoint_analysis()
    except Exception as e:
        print(f"  ✗ Error generating Graph 2: {e}")
    
    try:
        plot_weight_interaction()
    except Exception as e:
        print(f"  ✗ Error generating Graph 3: {e}")
    
    print("\n" + "=" * 80)
    print("ALL PLOTS GENERATED")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 80)

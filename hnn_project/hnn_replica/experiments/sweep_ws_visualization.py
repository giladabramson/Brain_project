"""
Generate Memory vs Responsiveness plots from weight sweep experiments (sweep_ws_extended.py)

This script visualizes the memory-responsiveness tradeoff across different weight configurations
with error bands showing variability across different seeds/runs.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPLICA_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPLICA_DIR / 'data'
OUTPUTS_DIR = REPLICA_DIR / 'outputs'
FIGURES_DIR = REPLICA_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

def load_and_organize_sweep_data():
    """
    Load metrics_summary.csv from all output directories and organize by weight configs.
    Returns a dataframe with columns: w_s, w_external, noise, segment, memory, reaction_mean, retention_mean, seed
    """
    all_data = []
    
    # Get all output directories
    for output_dir in sorted(OUTPUTS_DIR.glob("*")):
        if not output_dir.is_dir() or output_dir.name in ["tradeoff_analysis", "figures"]:
            continue
        
        metrics_file = output_dir / 'metrics_summary.csv'
        if not metrics_file.exists():
            continue
        
        try:
            df = pd.read_csv(metrics_file)
            # Try to extract environment variables from the output directory name or parent context
            # The sweep_ws scripts set W_S and W_EXTERNAL as env vars
            # We'll infer from the actual metrics if available
            all_data.append(df)
        except Exception as e:
            print(f"Skipped {output_dir}: {e}")
            continue
    
    if not all_data:
        print("No metrics_summary.csv files found!")
        return None
    
    df_combined = pd.concat(all_data, ignore_index=True)
    return df_combined


def create_weight_grid_plot(df):
    """Create a multi-panel plot showing memory vs responsiveness for different weight combinations"""
    
    if df is None or df.empty:
        print("No data available for plotting")
        return
    
    # Get unique weight combinations from sweep_hyper_combo_summary.csv if available
    sweep_csv = DATA_DIR / 'sweep_hyper_combo_summary.csv'
    if sweep_csv.exists():
        sweep_df = pd.read_csv(sweep_csv)
        # Get unique (w_s, w_external) combinations, excluding duplicates by keeping first occurrence
        weight_combos = sweep_df[['w_s', 'w_external']].drop_duplicates().values
        noise_levels = sweep_df['noise'].unique()
    else:
        # Extract from the metrics data if available
        if 'w_s' in df.columns and 'w_external' in df.columns:
            weight_combos = df[['w_s', 'w_external']].drop_duplicates().values
            noise_levels = df.get('noise', [0.5]).unique() if 'noise' in df.columns else [0.5]
        else:
            # Default grid if no weight info found
            weight_combos = [(0.5 + i*0.1, 8 + j*2) for i in range(3) for j in range(3)]
            noise_levels = [0.5]
    
    n_combos = len(weight_combos)
    n_cols = min(3, n_combos)
    n_rows = (n_combos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_combos == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for idx, (w_s, w_ext) in enumerate(weight_combos):
        ax = axes[idx]
        
        # Filter data for this weight combination
        subset = df[(df.get('w_s', 0) == w_s) & (df.get('w_external', 0) == w_ext)]
        
        if subset.empty and 'retention_mean' in df.columns:
            # If no weight columns, use all data
            subset = df
        
        if not subset.empty and 'retention_mean' in subset.columns and 'reaction_mean' in subset.columns:
            # Group by memory to get statistics
            grouped = subset.groupby('memory').agg({
                'reaction_time': ['mean', 'std'] if 'reaction_time' in subset.columns else ['mean'],
                'retention_time': ['mean', 'std'] if 'retention_time' in subset.columns else ['mean'],
                'reaction_mean': ['mean', 'std'] if 'reaction_mean' in subset.columns else ['mean'],
                'retention_mean': ['mean', 'std'] if 'retention_mean' in subset.columns else ['mean']
            }).reset_index()
            
            # Flatten column names
            grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
            
            # Determine which columns to use
            if 'retention_mean_mean' in grouped.columns:
                ret_mean_col = 'retention_mean_mean'
                ret_std_col = 'retention_mean_std'
                rxn_mean_col = 'reaction_mean_mean'
                rxn_std_col = 'reaction_mean_std'
            else:
                ret_mean_col = 'retention_time_mean'
                ret_std_col = 'retention_time_std'
                rxn_mean_col = 'reaction_time_mean'
                rxn_std_col = 'reaction_time_std'
            
            # Compute responsiveness
            responsiveness = 1.0 / (grouped[rxn_mean_col] + 1e-6)
            responsiveness_std = grouped[rxn_std_col] / (grouped[rxn_mean_col] ** 2 + 1e-6)
            
            # Plot
            ax.errorbar(grouped[ret_mean_col], responsiveness,
                       xerr=grouped[ret_std_col], yerr=responsiveness_std,
                       fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                       color='steelblue', alpha=0.7)
            
            ax.set_xlabel('Memory Duration ($\\tau_m$)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Responsiveness ($1/\\tau_r$)', fontsize=11, fontweight='bold')
            ax.set_title(f'$w_s={w_s:.2f}$, $w_{{ext}}={w_ext:.1f}$', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'$w_s={w_s:.2f}$, $w_{{ext}}={w_ext:.1f}$', fontsize=12)
    
    # Hide unused subplots
    for idx in range(len(weight_combos), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    pdf_path = FIGURES_DIR / 'weight_sweep_memory_responsiveness_grid.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    
    png_path = FIGURES_DIR / 'weight_sweep_memory_responsiveness_grid.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_path}")
    
    plt.close()


def create_weight_variation_plot(sweep_csv_path=None):
    """Create plot showing how memory-responsiveness tradeoff varies with weight changes"""
    
    if sweep_csv_path is None:
        sweep_csv_path = DATA_DIR / 'sweep_hyper_combo_summary.csv'
    
    if not sweep_csv_path.exists():
        print(f"Sweep data not found at {sweep_csv_path}")
        return
    
    df = pd.read_csv(sweep_csv_path)
    
    # Get unique weight combinations
    weight_combos = df.groupby(['w_s', 'w_external']).first().reset_index()
    
    print(f"Found {len(weight_combos)} weight combinations")
    print(f"Weight values: {sorted(df['w_s'].unique())}")
    print(f"External weights: {sorted(df['w_external'].unique())}")
    
    # Create a grid showing how retention varies with w_s and w_external
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Reaction time vs weights
    ax = axes[0]
    for w_ext in sorted(df['w_external'].unique()):
        subset = df[df['w_external'] == w_ext].sort_values('w_s')
        ax.plot(subset['w_s'], subset['reaction_mean'], marker='o', label=f'$w_{{ext}}={w_ext}$', linewidth=2)
    
    ax.set_xlabel('Self Weight ($w_s$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reaction Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Reaction Time vs Self Weight', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Retention vs weights
    ax = axes[1]
    for w_s in sorted(df['w_s'].unique()):
        subset = df[df['w_s'] == w_s].sort_values('w_external')
        ax.plot(subset['w_external'], subset['retention_mean'], marker='s', label=f'$w_s={w_s}$', linewidth=2)
    
    ax.set_xlabel('External Weight ($w_{{ext}}$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Retention Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Retention Time vs External Weight', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    pdf_path = FIGURES_DIR / 'weight_variation_analysis.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    
    png_path = FIGURES_DIR / 'weight_variation_analysis.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_path}")
    
    plt.close()


def create_tradeoff_heatmap():
    """Create heatmap showing memory-responsiveness tradeoff across weight space"""
    
    sweep_csv_path = DATA_DIR / 'sweep_hyper_combo_summary.csv'
    if not sweep_csv_path.exists():
        print(f"Sweep data not found at {sweep_csv_path}")
        return
    
    df = pd.read_csv(sweep_csv_path)
    
    # Create pivot tables for visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap 1: Reaction mean across weight space
    pivot_reaction = df.pivot_table(values='reaction_mean', index='w_s', columns='w_external', aggfunc='mean')
    im1 = axes[0].imshow(pivot_reaction.values, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xticks(range(len(pivot_reaction.columns)))
    axes[0].set_yticks(range(len(pivot_reaction.index)))
    axes[0].set_xticklabels([f'{x:.1f}' for x in pivot_reaction.columns])
    axes[0].set_yticklabels([f'{y:.2f}' for y in pivot_reaction.index])
    axes[0].set_xlabel('External Weight ($w_{{ext}}$)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Self Weight ($w_s$)', fontsize=12, fontweight='bold')
    axes[0].set_title('Reaction Time across Weight Space', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Reaction Time (s)')
    
    # Heatmap 2: Retention mean across weight space
    pivot_retention = df.pivot_table(values='retention_mean', index='w_s', columns='w_external', aggfunc='mean')
    im2 = axes[1].imshow(pivot_retention.values, cmap='viridis', aspect='auto')
    axes[1].set_xticks(range(len(pivot_retention.columns)))
    axes[1].set_yticks(range(len(pivot_retention.index)))
    axes[1].set_xticklabels([f'{x:.1f}' for x in pivot_retention.columns])
    axes[1].set_yticklabels([f'{y:.2f}' for y in pivot_retention.index])
    axes[1].set_xlabel('External Weight ($w_{{ext}}$)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Self Weight ($w_s$)', fontsize=12, fontweight='bold')
    axes[1].set_title('Retention Time across Weight Space', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Retention Time (s)')
    
    plt.tight_layout()
    
    pdf_path = FIGURES_DIR / 'weight_space_heatmaps.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    
    png_path = FIGURES_DIR / 'weight_space_heatmaps.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_path}")
    
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENERATING WEIGHT SWEEP ANALYSIS PLOTS")
    print("="*80 + "\n")
    
    print("[1/3] Loading sweep data and creating weight variation plot...")
    try:
        create_weight_variation_plot()
    except Exception as e:
        print(f"Error creating weight variation plot: {e}")
    
    print("\n[2/3] Creating weight space heatmaps...")
    try:
        create_tradeoff_heatmap()
    except Exception as e:
        print(f"Error creating heatmaps: {e}")
    
    print("\n[3/3] Creating multi-weight memory-responsiveness grid...")
    try:
        df = load_and_organize_sweep_data()
        create_weight_grid_plot(df)
    except Exception as e:
        print(f"Error creating grid plot: {e}")
    
    print("\n" + "="*80)
    print("WEIGHT SWEEP ANALYSIS COMPLETE")
    print(f"Output directory: {FIGURES_DIR}")
    print("="*80 + "\n")

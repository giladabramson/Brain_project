"""
Generate 3x3 Grid of Memory vs Responsiveness Tradeoff Plots

This script creates a grid visualization showing the tradeoff across
different experimental conditions or parameter values.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec

# Configuration
REPLICA_PATH = Path(__file__).parent / "hnn_replica"
OUTPUT_ROOT = REPLICA_PATH / "outputs"
RESULTS_DIR = REPLICA_PATH / "outputs" / "tradeoff_analysis"
RESULTS_DIR.mkdir(exist_ok=True)

def load_all_metrics():
    """Load metrics_per_trial.csv from all output directories"""
    all_data = []
    
    for output_dir in sorted(OUTPUT_ROOT.glob("*")):
        if not output_dir.is_dir() or output_dir.name == "tradeoff_analysis":
            continue
        
        metrics_file = output_dir / "metrics_per_trial.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            # Add experiment identifier
            df['experiment'] = output_dir.name
            all_data.append(df)
    
    if not all_data:
        print("No metrics_per_trial.csv files found!")
        return None
    
    return pd.concat(all_data, ignore_index=True)

def plot_3x3_grid(data):
    """Create a 3x3 grid of memory vs responsiveness plots"""
    
    # Group experiments
    experiments = sorted(data['experiment'].unique())
    n_experiments = min(len(experiments), 9)  # Max 9 for 3x3 grid
    experiments = experiments[:n_experiments]
    
    # Create 3x3 grid
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Global min/max for shared axes
    all_retention = data['retention_time']
    all_reaction = data['reaction_time']
    global_retention_range = (all_retention.min(), all_retention.max())
    global_reaction_range = (all_reaction.min(), all_reaction.max())
    
    # Create subplot for each experiment
    for idx, exp in enumerate(experiments):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        exp_data = data[data['experiment'] == exp]
        
        # Compute responsiveness
        responsiveness = 1.0 / exp_data['reaction_time']
        
        # Scatter plot
        ax.scatter(exp_data['retention_time'], responsiveness, 
                  alpha=0.5, s=20, color='steelblue', edgecolors='none')
        
        # Set shared axes limits
        ax.set_xlim(global_retention_range)
        ax.set_ylim(1.0 / global_reaction_range[1], 1.0 / global_reaction_range[0])
        
        # Labels
        ax.set_xlabel('Memory Duration (s)', fontsize=11)
        ax.set_ylabel('Responsiveness (1/s)', fontsize=11)
        ax.set_title(f'Experiment {idx+1}\n({exp[:15]}...)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add count
        ax.text(0.98, 0.02, f'n={len(exp_data)}', 
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('Memory vs Responsiveness Trade-off: 3×3 Grid of Experiments', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    pdf_path = Path(__file__).parent / "responsiveness_vs_memory_grid_3x3_shared_axes.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    
    # Also save PNG
    png_path = Path(__file__).parent / "responsiveness_vs_memory_grid_3x3_shared_axes.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Loading metrics from all output directories...")
    data = load_all_metrics()
    
    if data is not None:
        print(f"Loaded {len(data)} trials")
        print(f"Found {len(data['experiment'].unique())} experiments")
        
        plot_3x3_grid(data)
    else:
        print("No data to plot.")

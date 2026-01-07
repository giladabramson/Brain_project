"""
Generate Memory vs Responsiveness Tradeoff Plot from HNN Experiment Data

This script aggregates metrics across all output folders and creates the 
memory vs responsiveness (reaction time) tradeoff visualization.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_ROOT = Path(__file__).parent / "outputs"
RESULTS_DIR = OUTPUT_ROOT / "tradeoff_analysis"
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
            all_data.append(df)
    
    if not all_data:
        print("No metrics_per_trial.csv files found!")
        return None
    
    return pd.concat(all_data, ignore_index=True)

def compute_responsiveness(reaction_time):
    """Compute responsiveness as inverse of reaction time"""
    return 1.0 / reaction_time

def plot_tradeoff(data):
    """Plot memory vs responsiveness"""
    # Aggregate by memory
    summary = data.groupby('memory').agg({
        'reaction_time': ['mean', 'std'],
        'retention_time': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['memory', 'reaction_mean', 'reaction_std', 
                       'retention_mean', 'retention_std']
    
    # Compute responsiveness (inverse of reaction time)
    summary['responsiveness'] = 1.0 / summary['reaction_mean']
    summary['responsiveness_std'] = summary['reaction_std'] / (summary['reaction_mean'] ** 2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot tradeoff curve
    ax.errorbar(summary['retention_mean'], summary['responsiveness'], 
                xerr=summary['retention_std'], yerr=summary['responsiveness_std'],
                fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                label='Memory vs Responsiveness Tradeoff', color='steelblue')
    
    # Labels and formatting
    ax.set_xlabel('Memory Duration ($\\tau_m$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Responsiveness ($1/\\tau_r$)', fontsize=14, fontweight='bold')
    ax.set_title('Memory vs Responsiveness Tradeoff in HNN', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Save figure
    fig_path = RESULTS_DIR / "memory_responsiveness_tradeoff.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    
    # Also save PNG
    png_path = RESULTS_DIR / "memory_responsiveness_tradeoff.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {png_path}")
    
    # Save summary data
    summary_path = RESULTS_DIR / "tradeoff_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"✓ Saved: {summary_path}")
    
    plt.show()
    
    return summary

if __name__ == "__main__":
    print("Loading metrics from all output directories...")
    data = load_all_metrics()
    
    if data is not None:
        print(f"Loaded {len(data)} trials")
        print(f"Memory conditions: {sorted(data['memory'].unique())}")
        
        summary = plot_tradeoff(data)
        print("\nTrade-off Summary:")
        print(summary)
    else:
        print("No data to plot.")

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import os
import glob

# Grid search parameters - Experiment 1: Vary w_external, constant w_s
num_iterations = 100
fixed_noise = 0.5
fixed_w_s = 0.5
w_external_values = np.linspace(0.25, 2.5, num_iterations)

# Create results directory
results_dir = 'grid_search_results'
os.makedirs(results_dir, exist_ok=True)

# Run grid search
results = []
total_runs = len(w_external_values)
run_count = 0

for w_external in w_external_values:
    w_s = fixed_w_s
    run_count += 1
    print(f"Run {run_count}/{total_runs}: w_external={w_external:.2f}, w_s={w_s:.2f}, noise={fixed_noise}")
    
    try:
        # Set environment variables and run the script
        env = os.environ.copy()
        env['W_EXTERNAL'] = str(w_external)
        env['W_S'] = str(w_s)
        env['NOISE_LEVEL'] = str(fixed_noise)
        env['TRIAL_SAMPLES'] = '50'  # Keep existing default
        env['INPUT_LOG_TRIALS'] = '0'  # Disable detailed logging for grid search
        env['INPUT_PLOT_TRIALS'] = '0'  # Disable plotting for grid search
        env['INPUT_MODE'] = 'deterministic'  # Use single memory only
        
        # Run the script
        result = subprocess.run(
            [sys.executable, 'hnn_replica/MainHNN_SHB.py'],
            env=env,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise Exception(f"Script failed: {result.stderr}")
        
        # Look for the output directory created by the script
        output_dirs = sorted(glob.glob('hnn_replica/outputs/*'), key=os.path.getmtime)
        if output_dirs:
            latest_output_dir = output_dirs[-1]
            # Load metrics from CSV
            metrics_file = os.path.join(latest_output_dir, 'metrics_summary.csv')
            reaction_time = np.nan
            retention_time = np.nan
            
            if os.path.exists(metrics_file):
                import pandas as pd
                metrics_df = pd.read_csv(metrics_file)
                if not metrics_df.empty:
                    reaction_time = metrics_df['reaction_mean'].iloc[0]
                    retention_time = metrics_df['retention_mean'].iloc[0]
        else:
            reaction_time = np.nan
            retention_time = np.nan
        
        results.append({
            'w_external': w_external,
            'w_s': w_s,
            'noise': fixed_noise,
            'reaction_time': reaction_time,
            'retention_time': retention_time
        })
        
        print(f"  -> Reaction: {reaction_time:.4f}s, Retention: {retention_time:.4f}s")
        
    except Exception as e:
        print(f"Error with w_external={w_external:.2f}, w_s={w_s:.2f}: {e}")
        results.append({
            'w_external': w_external,
            'w_s': w_s,
            'noise': fixed_noise,
            'error': str(e)
        })

# Save all results summary
summary_file = os.path.join(results_dir, 'grid_search_summary.npy')
np.save(summary_file, results)

# Create combined PDF - add plots sequentially with parameter annotations
from PyPDF2 import PdfReader, PdfWriter
from matplotlib.backends.backend_pdf import PdfPages

print("\nCreating combined PDF with overlap plots and parameter annotations...")

# Find all overlap plot PDFs from the output directories
output_dirs = sorted(glob.glob('hnn_replica/outputs/*'), key=os.path.getmtime)

# Get the last total_runs directories
selected_dirs = output_dirs[-total_runs:]

# Create output PDF
pdf_path = os.path.join(results_dir, 'grid_search_overlaps.pdf')
writer = PdfWriter()

# Process each plot (no grouping needed for 10 plots)
for page_idx in range(0, total_runs):
    # Add annotation page for each plot
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title
    fig.suptitle(f'Configuration {page_idx + 1} of {total_runs}', fontsize=18, y=0.98)
    
    # Add parameter info
    run_info = results[page_idx] if page_idx < len(results) else {}
    w_ext = run_info.get('w_external', 'N/A')
    w_s_val = run_info.get('w_s', 'N/A')
    
    # Format values safely
    w_ext_str = f'{w_ext:.2f}' if isinstance(w_ext, (int, float)) else str(w_ext)
    w_s_str = f'{w_s_val:.2f}' if isinstance(w_s_val, (int, float)) else str(w_s_val)
    
    ax.text(0.5, 0.5, f'w_external = {w_ext_str}\nw_s = {w_s_str}\nnoise = {fixed_noise}', 
           ha='center', va='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save annotation to temporary PDF
    temp_pdf = os.path.join(results_dir, 'temp_annotation.pdf')
    plt.tight_layout(rect=[0, 0.9, 1, 0.96])
    plt.savefig(temp_pdf, format='pdf')
    plt.close(fig)
    
    # Add annotation page
    reader = PdfReader(temp_pdf)
    writer.add_page(reader.pages[0])
    
    # Add the plot PDF
    output_dir = selected_dirs[page_idx]
    
    # Look for the overlap plot PDF
    overlap_pdf = None
    for file in os.listdir(output_dir):
        if file.startswith('TrajectoryOverlap') and file.endswith('.pdf'):
            overlap_pdf = os.path.join(output_dir, file)
            break
    
    if overlap_pdf and os.path.exists(overlap_pdf):
        reader = PdfReader(overlap_pdf)
        writer.add_page(reader.pages[0])
        
        run_info = results[page_idx] if page_idx < len(results) else {}
        w_ext = run_info.get('w_external', 'N/A')
        w_s_val = run_info.get('w_s', 'N/A')
        print(f"  Added plot {page_idx+1}: w_external={w_ext}, w_s={w_s_val}")

# Write the combined PDF
with open(pdf_path, 'wb') as output_file:
    writer.write(output_file)

# Cleanup temp file
if os.path.exists(temp_pdf):
    os.remove(temp_pdf)

print(f"\nCombined PDF created: {pdf_path}")
print(f"Each plot has an annotation page followed by the full-size plot.")

# Create summary table
import pandas as pd
summary_data = []
for idx, r in enumerate(results, 1):
    summary_data.append({
        'Config': idx,
        'w_external': r['w_external'],
        'w_s': r['w_s'],
        'noise': r.get('noise', fixed_noise),
        'reaction_time': r.get('reaction_time', np.nan),
        'retention_time': r.get('retention_time', np.nan)
    })

df = pd.DataFrame(summary_data)
summary_csv = os.path.join(results_dir, 'metrics_table.csv')
df.to_csv(summary_csv, index=False)

print(f"\n{'='*80}")
print("METRICS SUMMARY TABLE")
print(f"{'='*80}")
print(df.to_string(index=False))
print(f"\n{'='*80}")
print(f"Table saved to: {summary_csv}")
print(f"Grid search complete. Results saved in {results_dir}/")
os.system(f'start {pdf_path}')

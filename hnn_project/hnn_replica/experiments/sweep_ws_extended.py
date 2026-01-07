import os
import subprocess
from pathlib import Path
import csv
import numpy as np

REPLICA_DIR = Path(__file__).resolve().parent
PYTHON = Path('/home/yoyo7/brain_project/.venv/bin/python')
OUTPUTS_DIR = REPLICA_DIR / 'outputs'

# Helper to run a single configuration

def run_config(w_s, w_external, noise):
    env = os.environ.copy()
    env['W_S'] = str(w_s)
    env['W_EXTERNAL'] = str(w_external)
    env['NOISE_LEVEL'] = str(noise)
    before = set(p.name for p in OUTPUTS_DIR.iterdir()) if OUTPUTS_DIR.exists() else set()
    subprocess.run([str(PYTHON), 'MainHNN_SHB.py'], cwd=str(REPLICA_DIR), env=env, check=True)
    after = set(p.name for p in OUTPUTS_DIR.iterdir())
    new_dirs = sorted(list(after - before))
    if new_dirs:
        latest_name = new_dirs[-1]
    else:
        latest_name = max(after, key=lambda name: name)
    latest_dir = OUTPUTS_DIR / latest_name
    metrics_path = latest_dir / 'metrics_summary.csv'
    if not metrics_path.exists():
        raise RuntimeError(f'Metrics missing for {latest_dir}')
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            def parse_float(value):
                value = value.strip()
                if not value:
                    return float('nan')
                return float(value)

            data.append({
                'w_s': w_s,
                'w_external': w_external,
                'noise': noise,
                'segment': int(row['segment']),
                'memory': int(row['memory']),
                'reaction_mean': parse_float(row['reaction_mean']),
                'retention_mean': parse_float(row['retention_mean'])
            })
    return data

# Sweep 1: lower noise (0.2) with paired w_external = 1 / w_s
low_noise = 0.2
pairs = [0.5, 0.75, 1.0, 1.25, 1.5]
results = []
print('--- Sweep A: noise=0.2, w_external=1/w_s ---')
for w_s in pairs:
    w_external = 1.0 / w_s
    results.extend(run_config(w_s, w_external, low_noise))

# Compute correlations for sweep A
segments = sorted(set(r['segment'] for r in results))
print('\nCorrelation (retention vs w_s) at noise=0.2:')
for seg in segments:
    ws_vals = [r['w_s'] for r in results if r['segment'] == seg and not np.isnan(r['retention_mean'])]
    retention_vals = [r['retention_mean'] for r in results if r['segment'] == seg and not np.isnan(r['retention_mean'])]
    corr = np.corrcoef(ws_vals, retention_vals)[0, 1] if len(set(ws_vals)) > 1 else float('nan')
    print(f'  Segment {seg}: corr = {corr:.3f}')

print('\nDetailed results (noise=0.2):')
for r in results:
    print(f"w_s={r['w_s']:.2f}, w_ext={r['w_external']:.2f}, segment={r['segment']}, "
          f"reaction_mean={r['reaction_mean']:.3f}s, retention_mean={r['retention_mean']:.3f}s")

# Sweep 2: same noise (0.5) but varying product w_s * w_external
print('\n--- Sweep B: noise=0.5, varied products ---')
noise_base = 0.5
configs = [
    (0.6, 1.5),
    (0.8, 1.4),
    (1.0, 1.1),
    (1.2, 0.9),
    (1.4, 0.7)
]
results_b = []
for w_s, w_external in configs:
    results_b.extend(run_config(w_s, w_external, noise_base))

segments_b = sorted(set(r['segment'] for r in results_b))
print('\nCorrelation (retention vs w_s) at noise=0.5 with varied products:')
for seg in segments_b:
    ws_vals = [r['w_s'] for r in results_b if r['segment'] == seg and not np.isnan(r['retention_mean'])]
    retention_vals = [r['retention_mean'] for r in results_b if r['segment'] == seg and not np.isnan(r['retention_mean'])]
    corr = np.corrcoef(ws_vals, retention_vals)[0, 1] if len(set(ws_vals)) > 1 else float('nan')
    print(f'  Segment {seg}: corr = {corr:.3f}')

print('\nDetailed results (noise=0.5 varied products):')
for r in results_b:
    print(f"w_s={r['w_s']:.2f}, w_ext={r['w_external']:.2f}, segment={r['segment']}, "
          f"reaction_mean={r['reaction_mean']:.3f}s, retention_mean={r['retention_mean']:.3f}s")

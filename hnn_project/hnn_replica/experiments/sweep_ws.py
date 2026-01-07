import os
import subprocess
from pathlib import Path
import csv
import numpy as np

REPLICA_DIR = Path(__file__).resolve().parent
PYTHON = Path('/home/yoyo7/brain_project/.venv/bin/python')
OUTPUTS_DIR = REPLICA_DIR / 'outputs'

pairs = [0.5, 0.75, 1.0, 1.25, 1.5]
results = []

for w_s in pairs:
    w_external = 1.0 / w_s
    env = os.environ.copy()
    env['W_S'] = str(w_s)
    env['W_EXTERNAL'] = str(w_external)
    before = set(p.name for p in OUTPUTS_DIR.iterdir()) if OUTPUTS_DIR.exists() else set()
    print(f'Running with w_s={w_s:.2f}, w_external={w_external:.2f}')
    subprocess.run([str(PYTHON), 'MainHNN_SHB.py'], cwd=str(REPLICA_DIR), env=env, check=True)
    after = set(p.name for p in OUTPUTS_DIR.iterdir())
    new_dirs = sorted(list(after - before))
    if not new_dirs:
        raise RuntimeError('No new output directory detected')
    latest_dir = OUTPUTS_DIR / new_dirs[-1]
    metrics_path = latest_dir / 'metrics_summary.csv'
    if not metrics_path.exists():
        raise RuntimeError(f'Metrics file missing for run {latest_dir}')
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'w_s': w_s,
                'w_external': w_external,
                'segment': int(row['segment']),
                'memory': int(row['memory']),
                'reaction_mean': float(row['reaction_mean']),
                'retention_mean': float(row['retention_mean'])
            })

segments = sorted({r['segment'] for r in results})
print('\nCorrelation summary (retention vs w_s):')
for segment in segments:
    ws_vals = [r['w_s'] for r in results if r['segment'] == segment]
    retention_vals = [r['retention_mean'] for r in results if r['segment'] == segment]
    if len(set(ws_vals)) < 2:
        corr = float('nan')
    else:
        corr = np.corrcoef(ws_vals, retention_vals)[0, 1]
    print(f'  Segment {segment}: corr = {corr:.3f}')

print('\nDetailed results:')
for r in results:
    print(f"w_s={r['w_s']:.2f}, w_ext={r['w_external']:.2f}, segment={r['segment']}, "
          f"reaction_mean={r['reaction_mean']:.3f}s, retention_mean={r['retention_mean']:.3f}s")

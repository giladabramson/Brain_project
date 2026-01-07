import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPLICA_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = REPLICA_DIR / "outputs"
PYTHON = Path(sys.executable)


def _latest_output(before: Iterable[str]) -> Path:
    before_set = set(before)
    if OUTPUTS_DIR.exists():
        after_set = {p.name for p in OUTPUTS_DIR.iterdir()}
    else:
        raise RuntimeError("Outputs directory missing after run")
    new_dirs = sorted(after_set - before_set)
    chosen = new_dirs[-1] if new_dirs else max(after_set)
    return OUTPUTS_DIR / chosen


def _read_metrics(directory: Path) -> List[Dict]:
    metrics_path = directory / "metrics_summary.csv"
    if not metrics_path.exists():
        raise RuntimeError(f"metrics_summary.csv not found in {directory}")
    with metrics_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        records = []
        for row in reader:
            records.append(
                {
                    "segment": int(row["segment"]),
                    "memory": int(row["memory"]),
                    "reaction_mean": float(row["reaction_mean"]) if row["reaction_mean"] else float("nan"),
                    "retention_mean": float(row["retention_mean"]) if row["retention_mean"] else float("nan"),
                }
            )
    return records


def _read_metadata(directory: Path) -> Dict:
    meta_path = directory / "run_metadata.json"
    if meta_path.exists():
        with meta_path.open() as fh:
            return json.load(fh)
    return {}


def run_config(w_s: float, w_external: float, noise: float, trials: int = 10) -> Dict:
    env = os.environ.copy()
    env["W_S"] = f"{w_s}"
    env["W_EXTERNAL"] = f"{w_external}"
    env["NOISE_LEVEL"] = f"{noise}"
    env["TRIAL_SAMPLES"] = str(trials)
    env.setdefault("STIM_DURATION", "2.0")

    before = []
    if OUTPUTS_DIR.exists():
        before = [p.name for p in OUTPUTS_DIR.iterdir()]

    subprocess.run([str(PYTHON), "MainHNN_SHB.py"], cwd=str(REPLICA_DIR), env=env, check=True)

    latest_dir = _latest_output(before)
    metrics = _read_metrics(latest_dir)
    metadata = _read_metadata(latest_dir)
    metadata.update(
        {
            "w_s": w_s,
            "w_external": w_external,
            "noise": noise,
            "output_dir": str(latest_dir.relative_to(REPLICA_DIR)),
        }
    )
    return {"metadata": metadata, "metrics": metrics}


def sweep(
    configs: Iterable[Tuple[float, float]],
    noise: float,
    trials: int = 10,
) -> List[Dict]:
    results = []
    for idx, (w_s, w_external) in enumerate(configs, 1):
        print(f"[{idx}] Running w_s={w_s:.3f}, w_ext={w_external:.3f}, noise={noise:.3f}")
        run_result = run_config(w_s, w_external, noise, trials=trials)
        results.append(run_result)
        meta = run_result["metadata"]
        print(f"    -> outputs in {meta['output_dir']}")
        for metric in run_result["metrics"]:
            print(
                f"       segment={metric['segment']} reaction_mean={metric['reaction_mean']:.3f}s "
                f"retention_mean={metric['retention_mean']:.3f}s"
            )
    return results


def main():
    noise = 0.5
    trials = 10
    combos = [
        # w_s held at 1.0, vary w_external
        (0.5, 6),
        (0.5, 8),
        (0.5, 10),
        (0.5, 12),
        (0.5, 14),
        # w_external held at 1.0, vary w_s
        (0.3, 10),
        (0.4, 10),
        (0.5, 10),
        (0.6, 10),
        (0.7, 10),
    ]
    results = sweep(combos, noise=noise, trials=trials)
    summary_path = REPLICA_DIR / "sweep_hyper_combo_summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["w_s", "w_external", "noise", "segment", "reaction_mean", "retention_mean", "output_dir"]
        )
        for result in results:
            meta = result["metadata"]
            for metric in result["metrics"]:
                writer.writerow(
                    [
                        meta["w_s"],
                        meta["w_external"],
                        meta["noise"],
                        metric["segment"],
                        metric["reaction_mean"],
                        metric["retention_mean"],
                        meta["output_dir"],
                    ]
                )
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

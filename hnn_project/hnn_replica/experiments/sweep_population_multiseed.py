"""Run a fixed grid of (N, P) Hopfield configurations across multiple seeds.

The script executes `MainHNN_SHB.py` for every combination of neuron count `N`
and stored memories `P` in the provided grids, repeating each config for a list
of random seeds. Results are aggregated to estimate the maximum number of
memories that can be retained (the "break point") for every population size.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Ensure we can render plots without a display server.
import matplotlib.pyplot as plt

REPLICA_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = REPLICA_DIR / "outputs"
PYTHON = Path(sys.executable)
DEFAULT_N_VALUES = [32, 40, 48, 56, 64, 72, 80, 96, 112, 128]
DEFAULT_P_VALUES = [5, 6, 8, 10, 12, 14, 16, 18, 19, 20]
DEFAULT_SEEDS = [101, 202, 303]


def _parse_int_list(raw: str, label: str) -> List[int]:
    if not raw:
        raise ValueError(f"{label} list cannot be empty")
    result: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {label}: '{token}'") from exc
        if value <= 0:
            raise ValueError(f"{label} values must be positive: {value}")
        result.append(value)
    if not result:
        raise ValueError(f"No valid {label} values parsed")
    return sorted(set(result))


def _snapshot_outputs() -> List[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    return [p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]


def _latest_output(before: Sequence[Path]) -> Path:
    before_names = {p.name for p in before}
    if not OUTPUTS_DIR.exists():
        raise RuntimeError("Outputs directory missing after simulation run")
    after = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]
    if not after:
        raise RuntimeError("No output directories produced by the run")
    new_dirs = [p for p in after if p.name not in before_names]
    candidates = new_dirs or after
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _read_metrics(directory: Path) -> List[Dict]:
    metrics_path = directory / "metrics_summary.csv"
    if not metrics_path.exists():
        return []
    rows: List[Dict] = []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "segment": int(row["segment"]),
                    "memory": int(row["memory"]),
                    "reaction_mean": float(row["reaction_mean"]) if row["reaction_mean"] else float("nan"),
                    "retention_mean": float(row["retention_mean"]) if row["retention_mean"] else float("nan"),
                }
            )
    return rows


def _read_metadata(directory: Path) -> Dict:
    meta_path = directory / "run_metadata.json"
    if not meta_path.exists():
        return {}
    import json

    with meta_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _evaluate_retention(metrics: List[Dict], metadata: Dict) -> Dict:
    tail_time = max(metadata.get("segment_duration", 0.0) - metadata.get("stim_duration", 0.0), 0.0)
    dt = metadata.get("dt", 0.01)
    tolerance = max(dt * 5, 0.05)
    stable = True
    min_retention = float("inf")
    for metric in metrics:
        retention = metric["retention_mean"]
        if math.isnan(retention) or retention + tolerance < tail_time:
            stable = False
        if not math.isnan(retention):
            min_retention = min(min_retention, retention)
    if min_retention == float("inf"):
        min_retention = float("nan")
    return {
        "tail_time": tail_time,
        "tolerance": tolerance,
        "stable": stable,
        "min_retention": min_retention,
    }


def _run_single(
    neuron_count: int,
    memory_count: int,
    noise: float,
    trials: int,
    w_s: float,
    w_external: float,
    stim_duration: float,
    t_end: float,
    seed: int,
    save_plots: bool,
    num_segments: int,
) -> Tuple[Dict, Dict]:
    env = os.environ.copy()
    env.update(
        {
            "HNN_N": str(neuron_count),
            "NUM_MEMORIES": str(memory_count),
            "NOISE_LEVEL": str(noise),
            "W_S": str(w_s),
            "W_EXTERNAL": str(w_external),
            "TRIAL_SAMPLES": str(trials),
            "INPUT_MODE": "deterministic",
            "DETERMINISTIC_GAIN": "3.0",
            "INPUT_LOG_TRIALS": "0",
            "INPUT_PLOT_TRIALS": "0",
            "INPUT_PLOT_SEGMENTS": "0",
            "SAVE_PLOTS": "1" if save_plots else "0",
            "RANDOM_SEED": str(seed),
            "STIM_DURATION": str(stim_duration),
            "T_END": str(t_end),
            "NUM_SEGMENTS": str(num_segments),
        }
    )
    before = _snapshot_outputs()
    subprocess.run([
        str(PYTHON),
        "MainHNN_SHB.py",
    ], cwd=str(REPLICA_DIR), env=env, check=True)
    latest = _latest_output(before)
    metrics = _read_metrics(latest)
    metadata = _read_metadata(latest)
    metadata.setdefault("segment_duration", t_end)
    metadata.setdefault("stim_duration", stim_duration)
    metadata.setdefault("dt", 0.01)
    metadata["output_dir"] = str(latest.relative_to(REPLICA_DIR))
    return metrics, metadata


def _write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid sweep over N/P combos with multiple seeds")
    parser.add_argument(
        "--n-values",
        default=",".join(str(v) for v in DEFAULT_N_VALUES),
        help="Comma-separated neuron counts (default: 10 values between 32 and 128)",
    )
    parser.add_argument(
        "--p-values",
        default=",".join(str(v) for v in DEFAULT_P_VALUES),
        help="Comma-separated memory counts (default: 10 values between 5 and 20)",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(v) for v in DEFAULT_SEEDS),
        help="Comma-separated random seeds per configuration",
    )
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level passed to MainHNN_SHB")
    parser.add_argument("--trials", type=int, default=6, help="Number of trials (TRIAL_SAMPLES)")
    parser.add_argument("--w-s", dest="w_s", type=float, default=1.0, help="Short-term weight scalar")
    parser.add_argument("--w-ext", dest="w_external", type=float, default=1.5, help="External drive scalar")
    parser.add_argument("--stim-duration", type=float, default=2.0, help="Stimulus duration in seconds")
    parser.add_argument("--t-end", type=float, default=10.0, help="Segment duration in seconds")
    parser.add_argument(
        "--num-segments",
        type=int,
        default=1,
        help="Input segments per run (default 1, i.e., single memory cue)",
    )
    parser.add_argument(
        "--break-threshold",
        type=float,
        default=1.0,
        help="Fraction of seeds that must remain stable to count as retained",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Keep overlap PDFs (disabled by default for speed)",
    )
    return parser


def _plot_breakpoints(path: Path, data: Dict[int, float], p_min: int, p_max: int) -> None:
    xs = sorted(data.keys())
    ys = [data[n] for n in xs]
    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, marker="o", linewidth=2, color="#1f77b4")
    plt.xlabel("Neuron count N")
    plt.ylabel("Estimated break point (max stable P)")
    plt.ylim(p_min - 1, p_max + 1)
    plt.xlim(min(xs) - 2, max(xs) + 2)
    plt.title("Break points per population size (multi-seed)")
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        n_values = _parse_int_list(args.n_values, "N")
        p_values = _parse_int_list(args.p_values, "P")
        seeds = _parse_int_list(args.seeds, "seed")
    except ValueError as exc:
        parser.error(str(exc))
    combos = [(n, p) for n, p in product(n_values, p_values) if p < n]
    skipped = [(n, p) for n, p in product(n_values, p_values) if p >= n]
    if skipped:
        skip_preview = ", ".join(f"(N={n}, P={p})" for n, p in skipped[:6])
        print(f"Skipping {len(skipped)} invalid combos where P >= N: {skip_preview}...")
    if not combos:
        parser.error("No valid (N, P) combinations after filtering by P < N")
    print(f"Running {len(combos)} configurations x {len(seeds)} seeds (total {len(combos) * len(seeds)} runs)...")
    per_run_rows: List[Dict] = []
    aggregate: Dict[Tuple[int, int], Dict[str, float]] = {}
    total_runs = len(combos) * len(seeds)
    run_index = 0
    for n, p in combos:
        for seed in seeds:
            run_index += 1
            print(f"[{run_index:4d}/{total_runs}] N={n:3d}, P={p:2d}, seed={seed}")
            metrics, metadata = _run_single(
                neuron_count=n,
                memory_count=p,
                noise=args.noise,
                trials=args.trials,
                w_s=args.w_s,
                w_external=args.w_external,
                stim_duration=args.stim_duration,
                t_end=args.t_end,
                seed=seed,
                save_plots=args.save_plots,
                num_segments=args.num_segments,
            )
            retention = _evaluate_retention(metrics, metadata)
            per_run_rows.append(
                {
                    "N": n,
                    "P": p,
                    "seed": seed,
                    "stable": retention["stable"],
                    "tail_time": retention["tail_time"],
                    "min_retention": retention["min_retention"],
                    "output_dir": metadata.get("output_dir", ""),
                }
            )
            key = (n, p)
            slot = aggregate.setdefault(
                key,
                {"passes": 0, "runs": 0, "min_retention": float("inf"), "tail_time": retention["tail_time"]},
            )
            slot["runs"] += 1
            slot["passes"] += int(retention["stable"])
            if not math.isnan(retention["min_retention"]):
                slot["min_retention"] = min(slot["min_retention"], retention["min_retention"])
    for slot in aggregate.values():
        if slot["min_retention"] == float("inf"):
            slot["min_retention"] = float("nan")
    per_run_path = REPLICA_DIR / "pn_multiseed_results.csv"
    _write_csv(
        per_run_path,
        per_run_rows,
        ["N", "P", "seed", "stable", "tail_time", "min_retention", "output_dir"],
    )
    summary_rows: List[Dict] = []
    stable_fractions: Dict[Tuple[int, int], float] = {}
    for n in n_values:
        for p in p_values:
            key = (n, p)
            slot = aggregate.get(key)
            if not slot:
                continue
            fraction = slot["passes"] / slot["runs"] if slot["runs"] else 0.0
            stable_fractions[key] = fraction
            summary_rows.append(
                {
                    "N": n,
                    "P": p,
                    "stable_fraction": fraction,
                    "all_stable": slot["passes"] == slot["runs"],
                    "min_retention": slot["min_retention"],
                    "tail_time": slot["tail_time"],
                }
            )
    summary_path = REPLICA_DIR / "pn_multiseed_summary.csv"
    _write_csv(summary_path, summary_rows, ["N", "P", "stable_fraction", "all_stable", "min_retention", "tail_time"])
    breakpoint_map: Dict[int, float] = {}
    for n in n_values:
        entries = [(p, stable_fractions.get((n, p), 0.0)) for p in p_values]
        best = max((p for p, frac in entries if frac >= args.break_threshold), default=float("nan"))
        breakpoint_map[n] = best
    plot_path = REPLICA_DIR / "pn_breakpoints.png"
    _plot_breakpoints(plot_path, breakpoint_map, min(p_values), max(p_values))
    print(f"\nPer-run metrics saved to {per_run_path}")
    print(f"Aggregate stability summary saved to {summary_path}")
    print(f"Break point plot written to {plot_path}")


if __name__ == "__main__":
    main()

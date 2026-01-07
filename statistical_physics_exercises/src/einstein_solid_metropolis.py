from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


class MetropolisEinsteinSolid:
    """Single Einstein solid evolved with a Metropolis heat-bath algorithm."""

    def __init__(
        self,
        num_oscillators: int = 100,
        theta: float = 2.5,
        rng: np.random.Generator | None = None,
    ):
        if num_oscillators <= 0:
            raise ValueError("num_oscillators must be positive")
        if theta <= 0:
            raise ValueError("theta must be positive")
        self.num_oscillators = int(num_oscillators)
        self.theta = float(theta)
        self.energies = np.zeros(self.num_oscillators, dtype=int)
        self.rng = rng or np.random.default_rng()
        self._accept_prob = math.exp(-1.0 / self.theta)
        self._total_energy = 0

    def total_energy(self) -> int:
        return self._total_energy

    def step(self) -> int:
        """Perform one Metropolis trial. Returns the total-energy change."""
        index = int(self.rng.integers(0, self.num_oscillators))
        delta = -1 if self.rng.random() < 0.5 else 1

        total_delta = 0
        if delta == -1:
            if self.energies[index] > 0:
                self.energies[index] -= 1
                total_delta = -1
        else:
            if self.rng.random() <= self._accept_prob:
                self.energies[index] += 1
                total_delta = 1

        self._total_energy += total_delta
        return total_delta


@dataclass
class SimulationResults:
    theta: float
    num_oscillators: int
    total_steps: int
    tracked_index: int
    total_energy_history: np.ndarray
    particle_energy_history: np.ndarray
    particle_energy_counts: np.ndarray
    energy_snapshots: dict[int, np.ndarray]
    sample_steps: list[int]


def _ensure_capacity(array: np.ndarray, new_size: int) -> np.ndarray:
    if new_size <= array.size:
        return array
    extended = np.zeros(new_size, dtype=array.dtype)
    extended[: array.size] = array
    return extended


def run_simulation(
    steps: int = 10_000_000,
    theta: float = 2.5,
    sample_steps: Iterable[int] | None = None,
    tracked_index: int = 0,
) -> SimulationResults:
    if steps <= 0:
        raise ValueError("steps must be positive")

    simulator = MetropolisEinsteinSolid(theta=theta)
    tracked_index = int(tracked_index)
    if not 0 <= tracked_index < simulator.num_oscillators:
        raise IndexError("tracked_index out of range")

    default_samples = [2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000]
    sample_steps = (
        sorted({s for s in (sample_steps or default_samples) if 1 <= s <= steps})
        or [steps]
    )

    total_energy_history = np.empty(steps, dtype=np.int32)
    particle_energy_history = np.empty(steps, dtype=np.uint16)
    max_energy_guess = int(math.ceil(10 * theta))
    particle_energy_counts = np.zeros(max(1, max_energy_guess + 1), dtype=np.int64)
    energy_snapshots: dict[int, np.ndarray] = {}

    sample_iter = iter(sample_steps)
    try:
        next_sample = next(sample_iter)
    except StopIteration:
        next_sample = None

    total_energy = simulator.total_energy()
    for step in range(1, steps + 1):
        total_energy += simulator.step()
        total_energy_history[step - 1] = total_energy

        tracked_energy = simulator.energies[tracked_index]
        if tracked_energy >= particle_energy_counts.size:
            particle_energy_counts = _ensure_capacity(
                particle_energy_counts, tracked_energy + 1
            )
        particle_energy_counts[tracked_energy] += 1
        particle_energy_history[step - 1] = tracked_energy

        if next_sample is not None and step == next_sample:
            counts = np.bincount(simulator.energies)
            energy_snapshots[step] = counts
            try:
                next_sample = next(sample_iter)
            except StopIteration:
                next_sample = None

    return SimulationResults(
        theta=theta,
        num_oscillators=simulator.num_oscillators,
        total_steps=steps,
        tracked_index=tracked_index,
        total_energy_history=total_energy_history,
        particle_energy_history=particle_energy_history,
        particle_energy_counts=particle_energy_counts,
        energy_snapshots=energy_snapshots,
        sample_steps=list(sample_steps),
    )


def plot_total_energy(results: SimulationResults) -> None:
    steps = np.arange(1, results.total_energy_history.size + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, results.total_energy_history, linewidth=1.0)
    plt.xlabel("Metropolis step")
    plt.ylabel("Total energy $q_{\\text{tot}}$")
    plt.title(f"Total energy trace (θ = {results.theta:.2f})")
    plt.tight_layout()
    output_path = Path(__file__).with_name("einstein_total_energy.png")
    plt.savefig(output_path, dpi=150)


def plot_energy_snapshots(results: SimulationResults) -> None:
    if not results.energy_snapshots:
        return

    max_bins = max(arr.size for arr in results.energy_snapshots.values())
    energies = np.arange(max_bins)

    plt.figure(figsize=(8, 4))
    aggregated = np.zeros(max_bins, dtype=float)
    samples_used = 0
    for step in results.sample_steps:
        counts = results.energy_snapshots.get(step)
        if counts is None:
            continue
        padded = np.zeros(max_bins, dtype=np.int64)
        padded[: counts.size] = counts
        fractions = padded / results.num_oscillators
        aggregated += fractions
        samples_used += 1
        mask = fractions > 0
        if np.any(mask):
            plt.semilogy(
                energies[mask],
                fractions[mask],
                marker="o",
                markersize=3,
                linewidth=1.0,
                label=f"Step {step:,}",
            )

    if samples_used:
        mean_fraction = aggregated / samples_used
        mask = mean_fraction > 0
        if np.any(mask):
            slope, intercept = np.polyfit(
                energies[mask],
                np.log(mean_fraction[mask]),
                1,
            )
            fit_values = np.exp(intercept + slope * energies[mask])
            plt.semilogy(
                energies[mask],
                fit_values,
                linestyle="--",
                linewidth=2.0,
                label=f"Exp fit (θ≈{-1/slope:.2f})",
            )

    plt.xlabel("Energy quanta q")
    plt.ylabel("Fraction of oscillators")
    plt.title("Energy distributions across the solid (semi-log)")
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(1e-2, None)
    plt.tight_layout()
    base = Path(__file__).with_name("einstein_solid_energy_distributions")
    plt.savefig(base.with_suffix(".png"), dpi=150)
    plt.yscale("linear")
    plt.ylim(0, None)
    plt.savefig(base.with_name(base.stem + "_linear.png"), dpi=150)


def plot_tracked_particle_counts(results: SimulationResults) -> None:
    counts = results.particle_energy_counts
    nonzero = counts > 0
    if not np.any(nonzero):
        return

    energies = np.arange(counts.size)[nonzero]
    fractions = counts[nonzero] / results.total_steps

    plt.figure(figsize=(8, 4))
    plt.bar(energies, fractions, width=0.8)
    mask = fractions > 0
    fit_label = None
    fit_values = None
    if np.any(mask):
        slope, intercept = np.polyfit(
            energies[mask],
            np.log(fractions[mask]),
            1,
        )
        fit_values = np.exp(intercept + slope * energies[mask])
        fit_label = f"Exp fit (θ≈{-1/slope:.2f})"
        plt.plot(
            energies[mask],
            fit_values,
            color="C1",
            linestyle="--",
            linewidth=1.5,
            label=fit_label,
        )
    plt.xlabel("Energy quanta q")
    plt.ylabel("Fraction of steps")
    plt.title(
        f"Energy occupancy for oscillator {results.tracked_index} "
        f"(θ = {results.theta:.2f})"
    )
    if fit_label:
        plt.legend()
    plt.tight_layout()
    output_path = Path(__file__).with_name("einstein_particle_energy_counts.png")
    plt.savefig(output_path, dpi=150)

    plt.figure(figsize=(8, 4))
    plt.plot(energies, fractions, marker="o", linewidth=1.0)
    if fit_values is not None:
        plt.plot(
            energies[mask],
            fit_values,
            color="C1",
            linestyle="--",
            linewidth=1.5,
            label=fit_label,
        )
    plt.yscale("log")
    plt.xlabel("Energy quanta q")
    plt.ylabel("Fraction of steps")
    plt.title(
        f"Energy occupancy (curve) for oscillator {results.tracked_index} "
        f"(θ = {results.theta:.2f})"
    )
    if fit_label:
        plt.legend()
    plt.tight_layout()
    output_path = Path(__file__).with_name("einstein_particle_energy_counts_line.png")
    plt.savefig(output_path, dpi=150)


if __name__ == "__main__":
    simulation = run_simulation()
    plot_total_energy(simulation)
    plot_energy_snapshots(simulation)
    plot_tracked_particle_counts(simulation)

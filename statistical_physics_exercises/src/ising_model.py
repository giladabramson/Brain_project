"""
Skeleton implementation for a 2D Ising model simulation.
"""

from __future__ import annotations

import numpy as np


def initialize_lattice(size: int, seed: int | None = None) -> np.ndarray:
    """Return a random spin configuration (+1 / -1) on a square lattice."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(size, size))


def total_energy(lattice: np.ndarray, coupling: float = 1.0) -> float:
    """Compute the total energy using nearest-neighbor interactions."""
    up = np.roll(lattice, shift=1, axis=0)
    down = np.roll(lattice, shift=-1, axis=0)
    left = np.roll(lattice, shift=1, axis=1)
    right = np.roll(lattice, shift=-1, axis=1)

    interaction = lattice * (up + down + left + right)
    return -0.5 * coupling * interaction.sum()


def metropolis_step(
    lattice: np.ndarray,
    beta: float,
    rng: np.random.Generator | None = None,
) -> None:
    """Perform an in-place Metropolis update on a randomly chosen spin."""
    if rng is None:
        rng = np.random.default_rng()

    size = lattice.shape[0]
    i = rng.integers(size)
    j = rng.integers(size)

    spin = lattice[i, j]
    neighbors = (
        lattice[(i + 1) % size, j]
        + lattice[(i - 1) % size, j]
        + lattice[i, (j + 1) % size]
        + lattice[i, (j - 1) % size]
    )

    delta_e = 2 * spin * neighbors
    if delta_e <= 0 or rng.random() < np.exp(-beta * delta_e):
        lattice[i, j] = -spin


if __name__ == "__main__":
    lattice = initialize_lattice(size=32, seed=42)
    beta = 0.4
    for _ in range(10_000):
        metropolis_step(lattice, beta)

    print(f"Final energy: {total_energy(lattice):.2f}")

import numpy as np

from statistical_physics_exercises.src.ising_model import (
    initialize_lattice,
    total_energy,
)


def test_initialize_lattice_shapes_and_values():
    lattice = initialize_lattice(size=8, seed=1)
    assert lattice.shape == (8, 8)
    assert set(np.unique(lattice)) <= {-1, 1}


def test_energy_invariant_under_global_flip():
    lattice = initialize_lattice(size=8, seed=2)
    flipped = -lattice
    assert total_energy(lattice) == total_energy(flipped)

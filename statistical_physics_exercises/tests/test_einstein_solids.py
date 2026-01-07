import numpy as np

from statistical_physics_exercises.src.einstein_solids import (
    CoupledEinsteinSolids,
    EinsteinSolid,
    create_default_simulation,
)


def test_random_initial_distribution_respects_totals():
    rng = np.random.default_rng(0)
    solid = EinsteinSolid.with_random_energy(num_oscillators=10, total_energy=20, rng=rng)
    assert solid.total_energy() == 20
    assert solid.energies.shape == (10,)


def test_step_conserves_total_energy():
    rng = np.random.default_rng(1)
    simulator = create_default_simulation(num_oscillators=5, energy_a=4, energy_b=2, rng=rng)
    total_before = simulator.solid_a.total_energy() + simulator.solid_b.total_energy()
    simulator.step()

    total_after = simulator.solid_a.total_energy() + simulator.solid_b.total_energy()
    assert total_before == total_after
    assert len(simulator.history_direction) == 1


def test_run_records_history():
    rng = np.random.default_rng(2)
    simulator = create_default_simulation(num_oscillators=5, energy_a=4, energy_b=0, rng=rng)
    simulator.run(steps=10, record_every=2)
    assert len(simulator.history_a) == len(simulator.history_b) == 1 + 10 // 2
    assert len(simulator.history_direction) == 10

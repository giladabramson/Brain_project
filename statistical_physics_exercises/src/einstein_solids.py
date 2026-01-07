import numpy as np


class EinsteinSolid:
    """Stores oscillator energies for a single solid."""

    def __init__(self, num_oscillators: int, energies: np.ndarray):
        energies = np.asarray(energies, dtype=int)
        if num_oscillators <= 0:
            raise ValueError("num_oscillators must be positive")
        if energies.shape != (num_oscillators,):
            raise ValueError("energies must have shape (num_oscillators,)")
        if np.any(energies < 0):
            raise ValueError("energies must be non-negative")
        self.num_oscillators = int(num_oscillators)
        self.energies = energies

    @classmethod
    def with_random_energy(
        cls,
        num_oscillators: int,
        total_energy: int,
        rng: np.random.Generator | None = None,
    ) -> "EinsteinSolid":
        if total_energy < 0:
            raise ValueError("total_energy must be non-negative")
        rng = rng or np.random.default_rng()
        energies = np.zeros(num_oscillators, dtype=int)
        for _ in range(total_energy):
            index = int(rng.integers(0, num_oscillators))
            energies[index] += 1
        return cls(num_oscillators, energies)

    def total_energy(self) -> int:
        return int(self.energies.sum())


class CoupledEinsteinSolids:
    """Trades single energy quanta between two solids."""

    def __init__(
        self,
        solid_a: EinsteinSolid,
        solid_b: EinsteinSolid,
        rng: np.random.Generator | None = None,
    ):
        self.solid_a = solid_a
        self.solid_b = solid_b
        self.rng = rng or np.random.default_rng()
        self.history_a = [solid_a.total_energy()]
        self.history_b = [solid_b.total_energy()]
        self.history_direction: list[str] = []

    def step(self) -> None:
        total_slots = self.solid_a.num_oscillators + self.solid_b.num_oscillators

        donor_slot = int(self.rng.integers(0, total_slots))
        receiver_slot = int(self.rng.integers(0, total_slots))

        if donor_slot < self.solid_a.num_oscillators:
            donor_solid = self.solid_a
            donor_index = donor_slot
        else:
            donor_solid = self.solid_b
            donor_index = donor_slot - self.solid_a.num_oscillators

        if receiver_slot < self.solid_a.num_oscillators:
            receiver_solid = self.solid_a
            receiver_index = receiver_slot
        else:
            receiver_solid = self.solid_b
            receiver_index = receiver_slot - self.solid_a.num_oscillators

        if donor_solid.energies[donor_index] > 0:
            donor_solid.energies[donor_index] -= 1
            receiver_solid.energies[receiver_index] += 1

            if donor_solid is self.solid_a and receiver_solid is self.solid_b:
                self.history_direction.append("A->B")
            elif donor_solid is self.solid_b and receiver_solid is self.solid_a:
                self.history_direction.append("B->A")
            else:
                self.history_direction.append("none")
        else:
            self.history_direction.append("none")

    def run(self, steps: int, record_every: int = 1) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if record_every <= 0:
            raise ValueError("record_every must be positive")
        for step_index in range(1, steps + 1):
            self.step()
            if step_index % record_every == 0:
                self.history_a.append(self.solid_a.total_energy())
                self.history_b.append(self.solid_b.total_energy())


def create_default_simulation(
    num_oscillators: int = 100,
    energy_a: int = 300,
    energy_b: int = 0,
    rng: np.random.Generator | None = None,
) -> CoupledEinsteinSolids:
    rng = rng or np.random.default_rng()
    solid_a = EinsteinSolid.with_random_energy(num_oscillators, energy_a, rng)
    solid_b = EinsteinSolid.with_random_energy(num_oscillators, energy_b, rng)
    return CoupledEinsteinSolids(solid_a, solid_b, rng)

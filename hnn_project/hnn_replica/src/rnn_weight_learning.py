import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Set seeds
torch.manual_seed(42)
np.random.seed(42)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DT = 0.01
T_TOTAL = 10.0
T_PRESENTATION = 2.0
N_NEURONS = 64

TIMESTEPS_TOTAL = int(T_TOTAL / DT)
TIMESTEPS_PRESENTATION = int(T_PRESENTATION / DT)

BATCH_SIZE = 2
N_EPOCHS = 2000
LEARNING_RATE = 0.005
L2_LAMBDA = 1e-6

ALPHA_RETENTION = 1.0
BETA_REACTION = 10.0

RESPONSE_THRESHOLD = 0.7
RETENTION_THRESHOLD = 0.6

EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments" / "rnn_weight_learning"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class VanillaRNN(nn.Module):
    """Vanilla RNN with configurable activation and symmetric W_self."""

    def __init__(self, n_neurons: int, activation: str = "tanh"):
        super().__init__()
        self.n = n_neurons
        self.activation = activation

        # Parameterize W_self via a free matrix; use symmetric view property.
        self.W_self_lower = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.1)
        self.W_input = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.1)

    @property
    def W_self(self):
        # Symmetric weights like Hopfield
        return (self.W_self_lower + self.W_self_lower.T) / 2.0

    def forward(self, input_sequence: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, timesteps, _ = input_sequence.shape

        h = torch.zeros(batch_size, self.n, device=input_sequence.device) if initial_state is None else initial_state
        states = []

        for t in range(timesteps):
            x_t = input_sequence[:, t, :]
            pre = h @ self.W_self.t() + x_t @ self.W_input.t()
            if self.activation == "tanh":
                h = torch.tanh(pre)
            elif self.activation == "relu":
                h = torch.relu(pre)
            elif self.activation == "linear":
                h = pre
            elif self.activation == "sigmoid":
                h = torch.sigmoid(pre)
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
            states.append(h)

        return torch.stack(states, dim=1)


# ----------------------------------------------------------------------------
# Data generators and metrics
# ----------------------------------------------------------------------------
def generate_memory_patterns(batch_size: int, n_neurons: int) -> torch.Tensor:
    mem = torch.where(torch.rand(batch_size, n_neurons) > 0.5, torch.ones(batch_size, n_neurons), -torch.ones(batch_size, n_neurons))
    return mem


def generate_input_sequence(memory_patterns: torch.Tensor, timesteps_presentation: int, timesteps_total: int) -> torch.Tensor:
    batch, n = memory_patterns.shape
    x = torch.zeros(batch, timesteps_total, n)
    gain = torch.rand(batch, 1, 1) * 1.5 + 2.0
    x[:, :timesteps_presentation, :] = memory_patterns.unsqueeze(1) * gain
    return x


def compute_overlap(states: torch.Tensor, memory_patterns: torch.Tensor) -> torch.Tensor:
    batch, timesteps, n = states.shape
    mem = memory_patterns.unsqueeze(1).expand(batch, timesteps, n)
    overlap = torch.sum(mem * states, dim=2) / n
    return torch.abs(overlap)


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
def compute_loss_option_a(overlaps: torch.Tensor, alpha: float, beta: float, timesteps_presentation: int):
    pres = overlaps[:, :timesteps_presentation]
    ret = overlaps[:, timesteps_presentation:]

    retention_loss = -torch.mean(ret)
    reaction_loss = -torch.mean(torch.max(pres, dim=1)[0])
    loss = alpha * retention_loss + beta * reaction_loss

    metrics = {
        "retention_loss": retention_loss.item(),
        "reaction_loss": reaction_loss.item(),
        "avg_retention_overlap": -retention_loss.item(),
        "avg_max_presentation_overlap": -reaction_loss.item(),
    }
    return loss, metrics


def compute_fixed_point_loss(model: VanillaRNN, memory_patterns: torch.Tensor, gamma: float = 0.5):
    pre = memory_patterns @ model.W_self.t()
    if model.activation == "tanh":
        next_state = torch.tanh(pre)
    elif model.activation == "relu":
        next_state = torch.relu(pre)
    elif model.activation == "linear":
        next_state = pre
    elif model.activation == "sigmoid":
        next_state = torch.sigmoid(pre)
    else:
        raise ValueError(f"Unknown activation: {model.activation}")
    fp_loss = torch.mean((next_state - memory_patterns) ** 2)
    return gamma * fp_loss


# ----------------------------------------------------------------------------
# Training & evaluation
# ----------------------------------------------------------------------------
def train_rnn(
    loss_option: str = "a",
    n_epochs: int = N_EPOCHS,
    lr: float = LEARNING_RATE,
    fixed_memory: torch.Tensor | None = None,
    activation: str = "tanh",
    target_retention: float = 0.99,
    target_reaction: float = 0.99,
    patience: int = 20,
):
    print("\n" + "=" * 80)
    print(f"TRAINING RNN WITH LOSS OPTION {loss_option.upper()} - ACTIVATION: {activation.upper()}")
    print("=" * 80 + "\n")
    print(f"Network size: {N_NEURONS}")
    print(f"Activation: {activation}")
    print(f"Total time: {T_TOTAL}s ({TIMESTEPS_TOTAL} steps)")
    print(f"Presentation: {T_PRESENTATION}s ({TIMESTEPS_PRESENTATION} steps)")
    n_memories = fixed_memory.shape[0] if fixed_memory is not None else BATCH_SIZE
    print(f"Training mode: alternating over {n_memories} memories (one per epoch) with batch=1")
    print(f"Loss weights: alpha={ALPHA_RETENTION}, beta={BETA_REACTION}")
    print(f"L2 regularization: lambda={L2_LAMBDA}\n")

    model = VanillaRNN(N_NEURONS, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = compute_loss_option_a

    if fixed_memory is None:
        fixed_memory = generate_memory_patterns(BATCH_SIZE, N_NEURONS)

    n_memories = fixed_memory.shape[0]
    for i in range(min(n_memories, 5)):  # Show first 5 memories
        print(f"Memory {i+1} (first 10): {fixed_memory[i, :10].tolist()}")
    if n_memories > 5:
        print(f"... and {n_memories - 5} more memories\n")
    else:
        print()

    history = {
        "total_loss": [],
        "retention_loss": [],
        "reaction_loss": [],
        "avg_retention": [],
        "avg_reaction": [],
    }

    converged_epoch: int | None = None
    streak = 0

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        mem_idx = epoch % fixed_memory.shape[0]
        memory_to_train = fixed_memory[mem_idx:mem_idx + 1, :]

        input_seq = generate_input_sequence(memory_to_train, TIMESTEPS_PRESENTATION, TIMESTEPS_TOTAL)
        states = model(input_seq)
        overlaps = compute_overlap(states, memory_to_train)

        loss, metrics = loss_fn(overlaps, ALPHA_RETENTION, BETA_REACTION, TIMESTEPS_PRESENTATION)
        fp_loss = compute_fixed_point_loss(model, memory_to_train, gamma=0.5)
        # L2 on parameters (use underlying parameter for symmetric W_self)
        l2_reg = L2_LAMBDA * (torch.sum(model.W_self_lower ** 2) + torch.sum(model.W_input ** 2))
        total_loss = loss + fp_loss + l2_reg

        total_loss.backward()
        optimizer.step()

        history["total_loss"].append(total_loss.item())
        history["retention_loss"].append(metrics["retention_loss"])
        history["reaction_loss"].append(metrics["reaction_loss"])
        history["avg_retention"].append(metrics["avg_retention_overlap"])
        history["avg_reaction"].append(metrics["avg_max_presentation_overlap"])

        if metrics["avg_retention_overlap"] >= target_retention and metrics["avg_max_presentation_overlap"] >= target_reaction:
            streak += 1
            if streak >= patience:
                converged_epoch = epoch + 1
                print(f"Converged at epoch {converged_epoch} (patience {patience}, targets R={target_retention}, Q={target_reaction})")
                break
        else:
            streak = 0

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:5d}/{n_epochs} | Loss: {total_loss.item():.4f} | Retention: {history['avg_retention'][-1]:.3f} | Reaction: {history['avg_reaction'][-1]:.3f}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    if converged_epoch is not None:
        print(f"Early stop: met strict convergence at epoch {converged_epoch}")
    else:
        print("Early stop: not reached; ran full schedule")
    print("=" * 80 + "\n")
    return model, history


def plot_training_history(history: dict, suffix: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(history["total_loss"])
    axes[0, 0].set_title("Total Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["retention_loss"], label="Retention Loss")
    axes[0, 1].plot(history["reaction_loss"], label="Reaction Loss")
    axes[0, 1].set_title("Loss Components", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history["avg_retention"], label="Retention (avg)")
    axes[1, 0].axhline(y=RETENTION_THRESHOLD, color="orange", linestyle="--", label=f"Threshold ({RETENTION_THRESHOLD})")
    axes[1, 0].set_title("Retention Overlap", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Overlap")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["avg_reaction"], label="Max Presentation (avg)")
    axes[1, 1].axhline(y=RESPONSE_THRESHOLD, color="r", linestyle="--", label=f"Threshold ({RESPONSE_THRESHOLD})")
    axes[1, 1].set_title("Reaction Overlap", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Overlap")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = EXPERIMENT_DIR / f"training_history_{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {fname}")
    plt.close()


def visualize_learned_weights(model: VanillaRNN, suffix: str):
    W_self = model.W_self.detach().cpu().numpy()
    W_input = model.W_input.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = axes[0].imshow(W_self, cmap="RdBu_r", aspect="auto")
    axes[0].set_title("W_self (symmetric)", fontweight="bold")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(W_input, cmap="RdBu_r", aspect="auto")
    axes[1].set_title("W_input", fontweight="bold")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    fname = EXPERIMENT_DIR / f"learned_weights_{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {fname}")
    plt.close()

    print("Weight Statistics:")
    print(f"  W_self  - Mean: {W_self.mean():.4f}, Std: {W_self.std():.4f}")
    print(f"  W_input - Mean: {W_input.mean():.4f}, Std: {W_input.std():.4f}")


def test_model(model: VanillaRNN, suffix: str, fixed_memory: torch.Tensor):
    print("\n" + "=" * 80)
    print("TESTING TRAINED MODEL ON THE LEARNED MEMORY")
    print("=" * 80 + "\n")

    input_seq = generate_input_sequence(fixed_memory, TIMESTEPS_PRESENTATION, TIMESTEPS_TOTAL)
    with torch.no_grad():
        states = model(input_seq)
        overlaps = compute_overlap(states, fixed_memory)

    time = np.arange(TIMESTEPS_TOTAL) * DT
    n_mem = overlaps.shape[0]
    fig, axes = plt.subplots(n_mem, 1, figsize=(12, 5 * n_mem))
    if n_mem == 1:
        axes = [axes]

    ret_list, react_list = [], []
    for i in range(n_mem):
        curve = overlaps[i].cpu().numpy()
        ret = curve[TIMESTEPS_PRESENTATION:].mean()
        react = curve[:TIMESTEPS_PRESENTATION].max()
        ret_list.append(ret)
        react_list.append(react)

        axes[i].plot(time, curve, linewidth=2, label=f"Memory {i + 1}")
        axes[i].axvline(x=T_PRESENTATION, color="gray", linestyle="--", label="End of presentation")
        axes[i].axhline(y=RESPONSE_THRESHOLD, color="r", linestyle="--", alpha=0.5, label="Response threshold")
        axes[i].axhline(y=RETENTION_THRESHOLD, color="orange", linestyle="--", alpha=0.5, label="Retention threshold")
        axes[i].set_title(f"Memory {i + 1}: Reaction={react:.3f}, Retention={ret:.3f}", fontweight="bold")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Overlap")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = EXPERIMENT_DIR / f"test_trajectory_{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {fname}")
    plt.close()

    print("Test Results:")
    for i in range(n_mem):
        print(f"  Memory {i + 1} - Reaction: {react_list[i]:.3f}, Retention: {ret_list[i]:.3f}")
    print(f"  Average Reaction: {np.mean(react_list):.3f}")
    print(f"  Average Retention: {np.mean(ret_list):.3f}")


def test_model_with_noise(model: VanillaRNN, suffix: str, fixed_memory: torch.Tensor, noise_std: float = 0.1):
    """Test network robustness to noisy memory inputs (true Hopfield capacity test)."""
    print("\n" + "=" * 80)
    print(f"TESTING TRAINED MODEL WITH NOISY INPUT (noise_std={noise_std})")
    print("=" * 80 + "\n")

    n_mem = fixed_memory.shape[0]
    
    # Add noise to memories for testing
    noisy_memories = fixed_memory + torch.randn_like(fixed_memory) * noise_std
    noisy_memories = torch.clamp(noisy_memories, -1.0, 1.0)  # Keep in valid range
    
    # Diagnostic: measure actual noise corruption
    noise_applied = noisy_memories - fixed_memory
    corruption_l2 = torch.norm(noise_applied, p=2, dim=1).mean().item()
    avg_per_neuron = torch.abs(noise_applied).mean().item()
    print(f"Noise Statistics (with clamping):")
    print(f"  L2 norm per pattern: {corruption_l2:.4f}")
    print(f"  Mean absolute corruption per neuron: {avg_per_neuron:.4f}")
    print(f"  Requested std: {noise_std:.4f}")
    
    # Measure initial overlap (noisy patterns vs clean patterns)
    initial_overlap = torch.sum(noisy_memories * fixed_memory, dim=1) / fixed_memory.shape[1]
    print(f"  Initial overlap (noisy vs clean): {initial_overlap.mean().item():.4f}")
    print()
    
    input_seq = generate_input_sequence(noisy_memories, TIMESTEPS_PRESENTATION, TIMESTEPS_TOTAL)
    
    with torch.no_grad():
        states = model(input_seq)
        # Compute overlap with CLEAN patterns (not noisy)
        overlaps = compute_overlap(states, fixed_memory)

    time = np.arange(TIMESTEPS_TOTAL) * DT
    fig, axes = plt.subplots(n_mem, 1, figsize=(12, 5 * n_mem))
    if n_mem == 1:
        axes = [axes]

    ret_list, react_list = [], []
    for i in range(n_mem):
        curve = overlaps[i].cpu().numpy()
        ret = curve[TIMESTEPS_PRESENTATION:].mean()
        react = curve[:TIMESTEPS_PRESENTATION].max()
        ret_list.append(ret)
        react_list.append(react)

        axes[i].plot(time, curve, linewidth=2, label=f"Memory {i + 1} (converged from noisy input)")
        axes[i].axvline(x=T_PRESENTATION, color="gray", linestyle="--", label="End of presentation")
        axes[i].axhline(y=RESPONSE_THRESHOLD, color="r", linestyle="--", alpha=0.5, label="Response threshold")
        axes[i].axhline(y=RETENTION_THRESHOLD, color="orange", linestyle="--", alpha=0.5, label="Retention threshold")
        axes[i].set_title(f"Memory {i + 1}: Reaction={react:.3f}, Retention={ret:.3f}", fontweight="bold")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Overlap (with clean pattern)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = EXPERIMENT_DIR / f"test_trajectory_noisy_{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {fname}")
    plt.close()

    print(f"Test Results (input corrupted by noise_std={noise_std}):")
    for i in range(n_mem):
        print(f"  Memory {i + 1} - Reaction: {react_list[i]:.3f}, Retention: {ret_list[i]:.3f}")
    print(f"  Average Reaction: {np.mean(react_list):.3f}")
    print(f"  Average Retention: {np.mean(ret_list):.3f}")


def test_model_with_bit_flip(model: VanillaRNN, suffix: str, fixed_memory: torch.Tensor, flip_rate: float = 0.1):
    """Test with actual bit flips (more realistic corruption)."""
    print("\n" + "=" * 80)
    print(f"TESTING TRAINED MODEL WITH BIT FLIP (flip_rate={flip_rate})")
    print("=" * 80 + "\n")

    n_mem = fixed_memory.shape[0]
    
    # Create bit-flipped version (actually flip bits, not add noise)
    flipped_memories = fixed_memory.clone()
    for i in range(n_mem):
        flip_mask = torch.rand(fixed_memory.shape[1]) < flip_rate
        flipped_memories[i, flip_mask] *= -1.0  # Flip these bits
    
    # Diagnostic
    bit_diff = (flipped_memories != fixed_memory).float().mean().item()
    initial_overlap = torch.sum(flipped_memories * fixed_memory, dim=1) / fixed_memory.shape[1]
    print(f"Bit Flip Statistics:")
    print(f"  Fraction flipped: {bit_diff:.4f} (expected {flip_rate:.4f})")
    print(f"  Initial overlap (flipped vs clean): {initial_overlap.mean().item():.4f}")
    print()
    
    input_seq = generate_input_sequence(flipped_memories, TIMESTEPS_PRESENTATION, TIMESTEPS_TOTAL)
    
    with torch.no_grad():
        states = model(input_seq)
        overlaps = compute_overlap(states, fixed_memory)

    time = np.arange(TIMESTEPS_TOTAL) * DT
    fig, axes = plt.subplots(n_mem, 1, figsize=(12, 5 * n_mem))
    if n_mem == 1:
        axes = [axes]

    ret_list, react_list = [], []
    for i in range(n_mem):
        curve = overlaps[i].cpu().numpy()
        ret = curve[TIMESTEPS_PRESENTATION:].mean()
        react = curve[:TIMESTEPS_PRESENTATION].max()
        ret_list.append(ret)
        react_list.append(react)

        axes[i].plot(time, curve, linewidth=2, label=f"Memory {i + 1}")
        axes[i].axvline(x=T_PRESENTATION, color="gray", linestyle="--", label="End of presentation")
        axes[i].axhline(y=RESPONSE_THRESHOLD, color="r", linestyle="--", alpha=0.5, label="Response threshold")
        axes[i].axhline(y=RETENTION_THRESHOLD, color="orange", linestyle="--", alpha=0.5, label="Retention threshold")
        axes[i].set_title(f"Memory {i + 1}: Reaction={react:.3f}, Retention={ret:.3f}", fontweight="bold")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Overlap (with clean pattern)")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = EXPERIMENT_DIR / f"test_trajectory_bitflip_{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {fname}")
    plt.close()

    print(f"Test Results (input corrupted by bit flips, rate={flip_rate}):")
    for i in range(n_mem):
        print(f"  Memory {i + 1} - Reaction: {react_list[i]:.3f}, Retention: {ret_list[i]:.3f}")
    print(f"  Average Reaction: {np.mean(react_list):.3f}")
    print(f"  Average Retention: {np.mean(ret_list):.3f}")



def run_tanh_two_memory():
    print("\n" + "=" * 80)
    print("RNN WEIGHT LEARNING EXPERIMENT (tanh, 2 memories)")
    print("=" * 80)

    fixed_memory = generate_memory_patterns(BATCH_SIZE, N_NEURONS)
    print("\n>>> Training...")
    model, history = train_rnn(loss_option="a", fixed_memory=fixed_memory, activation="tanh")
    plot_training_history(history, "option_a_tanh")
    visualize_learned_weights(model, "option_a_tanh")
    test_model(model, "option_a_tanh", fixed_memory)


def run_tanh_five_memory():
    print("\n" + "=" * 80)
    print("RNN WEIGHT LEARNING EXPERIMENT (tanh, 5 memories)")
    print("=" * 80)

    fixed_memory = generate_memory_patterns(5, N_NEURONS)  # 5 memory patterns
    print("\n>>> Training...")
    model, history = train_rnn(loss_option="a", fixed_memory=fixed_memory, activation="tanh")
    plot_training_history(history, "option_a_tanh_5mem")
    visualize_learned_weights(model, "option_a_tanh_5mem")
    test_model(model, "option_a_tanh_5mem", fixed_memory)


def run_tanh_memory_sweep(memory_counts: list[int] | None = None):
    if memory_counts is None:
        memory_counts = [5, 10, 15, 20]

    for count in memory_counts:
        print("\n" + "#" * 80)
        print(f"RUN {count} MEMORIES")
        print("#" * 80)
        fixed_memory = generate_memory_patterns(count, N_NEURONS)
        suffix = f"option_a_tanh_{count}mem"
        model, history = train_rnn(
            loss_option="a",
            fixed_memory=fixed_memory,
            activation="tanh",
            target_retention=0.99,
            target_reaction=0.99,
            patience=20,
        )
        plot_training_history(history, suffix)
        visualize_learned_weights(model, suffix)
        test_model(model, suffix, fixed_memory)


def run_tanh_five_memory_with_noise(noise_std: float = 0.1):
    print("\n" + "=" * 80)
    print(f"RNN WEIGHT LEARNING EXPERIMENT (tanh, 5 memories, with noise std={noise_std})")
    print("=" * 80)

    fixed_memory = generate_memory_patterns(5, N_NEURONS)
    print("\n>>> Training (300 epochs)...")
    model, history = train_rnn(
        loss_option="a",
        n_epochs=300,
        fixed_memory=fixed_memory,
        activation="tanh",
        target_retention=0.99,
        target_reaction=0.99,
        patience=20,
    )
    plot_training_history(history, "option_a_tanh_5mem_noise")
    visualize_learned_weights(model, "option_a_tanh_5mem_noise")
    
    print("\n>>> Testing with clean patterns...")
    test_model(model, "option_a_tanh_5mem_noise", fixed_memory)
    
    print(f"\n>>> Testing with noisy patterns (Gaussian noise_std={noise_std})...")
    test_model_with_noise(model, "option_a_tanh_5mem_noise", fixed_memory, noise_std=noise_std)
    
    print(f"\n>>> Testing with bit flips (flip_rate=0.1)...")
    test_model_with_bit_flip(model, "option_a_tanh_5mem_noise", fixed_memory, flip_rate=0.1)


if __name__ == "__main__":
    run_tanh_five_memory_with_noise(noise_std=0.5)

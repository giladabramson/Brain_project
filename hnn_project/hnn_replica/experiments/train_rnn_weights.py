"""
Train a vanilla RNN to learn optimal weight matrices (W_self, W_input) that 
maximize both memory retention and reaction speed.

Based on HNN parameters:
- dt = 0.01
- Total time = 10s
- Presentation time = 2s
- Activation: tanh (smooth version of sign function)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# HNN-based Parameters
# ============================================================================
DT = 0.01  # Time step (from HNN)
T_TOTAL = 10.0  # Total simulation time (seconds)
T_PRESENTATION = 2.0  # Input presentation duration (seconds)
N_NEURONS = 64  # Network size (increased for multi-pattern storage)

# Compute timesteps
TIMESTEPS_TOTAL = int(T_TOTAL / DT)  # 1000 timesteps
TIMESTEPS_PRESENTATION = int(T_PRESENTATION / DT)  # 200 timesteps

# Training parameters
BATCH_SIZE = 2  # Two memory patterns (no noise, learn multiple memories)
N_EPOCHS = 300  # Train longer for single memory
LEARNING_RATE = 0.005  # Higher learning rate for faster convergence
L2_LAMBDA = 1e-06  # Even lower L2 to allow larger weights for stability

# Loss weights (from HNN analysis: retention~8s, reaction~0.2-2s)
# We weight reaction more heavily since it's smaller scale
ALPHA_RETENTION = 1.0
BETA_REACTION = 10.0  # Restore reaction term

# Thresholds (from HNN code)
RESPONSE_THRESHOLD = 0.7  # Overlap threshold for "successful reaction"
RETENTION_THRESHOLD = 0.6  # Overlap threshold for "good retention"

# Output directory
EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments" / "rnn_weight_learning"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Vanilla RNN Model
# ============================================================================
class VanillaRNN(nn.Module):
    """
    Simple RNN with explicit weight matrices:
    - W_self: recurrent/internal connections (n×n)
    - W_input: external input connections (n×n)
    
    No trainable readout - overlap is computed directly with memory patterns.
    """
    
    def __init__(self, n_neurons, activation='tanh'):
        super().__init__()
        self.n = n_neurons
        self.activation = activation
        
        # W_self: parameterize with lower triangle + diagonal, enforce symmetry
        # This matches Hopfield networks which use symmetric weights
        self.W_self_lower = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.1)
        self.W_input = nn.Parameter(torch.randn(n_neurons, n_neurons) * 0.1)
    
    @property
    def W_self(self):
        """Make W_self symmetric: W = (W_lower + W_lower.T) / 2"""
        return (self.W_self_lower + self.W_self_lower.T) / 2
        
    def forward(self, input_sequence, initial_state=None):
        """
        Forward pass through time.
        
        Args:
            input_sequence: (batch, timesteps, n_neurons)
            initial_state: (batch, n_neurons) or None
            
        Returns:
            states: (batch, timesteps, n_neurons) - all hidden states
        """
        batch_size, timesteps, _ = input_sequence.shape
        
        # Initialize state
        if initial_state is None:
            h = torch.zeros(batch_size, self.n, device=input_sequence.device)
        else:
            h = initial_state
            
        # Store all states for overlap computation
        states = []
        
        # Run through time
        for t in range(timesteps):
            x_t = input_sequence[:, t, :]  # (batch, n_neurons)
            
            # RNN update with configurable activation
            pre_activation = h @ self.W_self.t() + x_t @ self.W_input.t()
            
            if self.activation == 'tanh':
                h = torch.tanh(pre_activation)
            elif self.activation == 'relu':
                h = torch.relu(pre_activation)
            elif self.activation == 'linear':
                h = pre_activation
            elif self.activation == 'sigmoid':
                h = torch.sigmoid(pre_activation)
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
            
            states.append(h)
            
        # Stack: (batch, timesteps, n_neurons)
        states = torch.stack(states, dim=1)
        
        return states


# ============================================================================
# Data Generation
# ============================================================================
def generate_memory_patterns(batch_size, n_neurons):
    """
    Generate random binary memory patterns (like HNN).
    
    Returns:
        memories: (batch, n_neurons) with values in {-1, +1}
    """
    return torch.randint(0, 2, (batch_size, n_neurons)).float() * 2 - 1


def generate_input_sequence(memory_patterns, timesteps_presentation, timesteps_total):
    """
    Generate input sequence that presents memory patterns for T_presentation,
    then turns off.
    
    Args:
        memory_patterns: (batch, n_neurons)
        timesteps_presentation: int
        timesteps_total: int
        
    Returns:
        input_seq: (batch, timesteps_total, n_neurons)
    """
    batch_size, n_neurons = memory_patterns.shape
    input_seq = torch.zeros(batch_size, timesteps_total, n_neurons)
    
    # Present memory during presentation period with some gain
    # (matching HNN's input gains between 2.0-3.5 for target memory)
    gain = torch.rand(batch_size, 1, 1) * 1.5 + 2.0  # Random gain [2.0, 3.5]
    input_seq[:, :timesteps_presentation, :] = memory_patterns.unsqueeze(1) * gain
    
    return input_seq


# ============================================================================
# Overlap Computation (matching HNN)
# ============================================================================
def compute_overlap(states, memory_patterns):
    """
    Compute overlap between network states and memory patterns.
    
    Matching HNN: overlap = (1/N) * |dot(memory, tanh(5*state))|
    But states are already tanh'd in RNN, so: overlap = (1/N) * |dot(memory, state)|
    
    Args:
        states: (batch, timesteps, n_neurons)
        memory_patterns: (batch, n_neurons)
        
    Returns:
        overlaps: (batch, timesteps) - overlap at each timestep
    """
    batch_size, timesteps, n_neurons = states.shape
    
    # Expand memory for broadcasting: (batch, 1, n_neurons)
    mem = memory_patterns.unsqueeze(1)
    
    # Dot product: (batch, timesteps)
    overlap = torch.sum(mem * states, dim=2) / n_neurons
    
    # Absolute value (matching HNN)
    overlap = torch.abs(overlap)
    
    return overlap


# ============================================================================
# Loss Function
# ============================================================================
def compute_loss_option_a(overlaps, alpha, beta, timesteps_presentation):
    """
    Option A: Average-based loss
    
    - Retention: Maximize average overlap AFTER presentation
    - Reaction: Minimize time to reach threshold DURING presentation
    
    Args:
        overlaps: (batch, timesteps)
        alpha: retention weight
        beta: reaction weight
        timesteps_presentation: when presentation ends
        
    Returns:
        loss: scalar
        metrics: dict with individual components
    """
    # Split into presentation and retention periods
    overlap_presentation = overlaps[:, :timesteps_presentation]  # (batch, T_pres)
    overlap_retention = overlaps[:, timesteps_presentation:]  # (batch, T_ret)
    
    # Retention loss: negative mean overlap (we want to maximize)
    retention_loss = -torch.mean(overlap_retention)
    
    # Reaction loss: negative max overlap (we want high overlap during presentation)
    # Alternative: could use time to threshold
    reaction_loss = -torch.mean(torch.max(overlap_presentation, dim=1)[0])
    
    # Total loss with L2 regularization (added externally)
    loss = alpha * retention_loss + beta * reaction_loss
    
    metrics = {
        'retention_loss': retention_loss.item(),
        'reaction_loss': reaction_loss.item(),
        'avg_retention_overlap': -retention_loss.item(),
        'avg_max_presentation_overlap': -reaction_loss.item(),
    }
    
    return loss, metrics


def compute_loss_option_b(overlaps, alpha, beta, timesteps_presentation):
    """
    Option B: Min/max-based loss (stricter)
    
    - Retention: Maximize MINIMUM overlap after presentation (worst case)
    - Reaction: Maximize MAXIMUM overlap during presentation
    
    Args:
        overlaps: (batch, timesteps)
        alpha: retention weight
        beta: reaction weight
        timesteps_presentation: when presentation ends
        
    Returns:
        loss: scalar
        metrics: dict with individual components
    """
    overlap_presentation = overlaps[:, :timesteps_presentation]
    overlap_retention = overlaps[:, timesteps_presentation:]
    
    # Retention: maximize worst-case retention
    min_retention = torch.min(overlap_retention, dim=1)[0]  # (batch,)
    retention_loss = -torch.mean(min_retention)
    
    # Reaction: maximize best overlap during presentation
    max_presentation = torch.max(overlap_presentation, dim=1)[0]  # (batch,)
    reaction_loss = -torch.mean(max_presentation)
    
    loss = alpha * retention_loss + beta * reaction_loss
    
    metrics = {
        'retention_loss': retention_loss.item(),
        'reaction_loss': reaction_loss.item(),
        'min_retention_overlap': -retention_loss.item(),
        'max_presentation_overlap': -reaction_loss.item(),
    }
    
    return loss, metrics


def compute_fixed_point_loss(model, memory_patterns, gamma=0.1):
    """
    Explicitly enforce that memory patterns are fixed points:
    m ≈ tanh(W_self @ m)
    
    Args:
        model: VanillaRNN instance
        memory_patterns: (batch, n_neurons)
        gamma: weight for fixed-point loss
    
    Returns:
        loss: scalar fixed-point loss
    """
    # Compute next state if we start from memory pattern
    pre_activation = memory_patterns @ model.W_self.t()
    
    if model.activation == 'tanh':
        next_state = torch.tanh(pre_activation)
    elif model.activation == 'relu':
        next_state = torch.relu(pre_activation)
    elif model.activation == 'linear':
        next_state = pre_activation
    elif model.activation == 'sigmoid':
        next_state = torch.sigmoid(pre_activation)
    else:
        raise ValueError(f"Unknown activation: {model.activation}")
    
    # Fixed point loss: penalize deviation from memory pattern
    fp_loss = torch.mean((next_state - memory_patterns) ** 2)
    
    return gamma * fp_loss


# ============================================================================
# Training Loop
# ============================================================================
def train_rnn(loss_option='a', n_epochs=N_EPOCHS, lr=LEARNING_RATE, fixed_memory=None, activation='tanh'):
    """
    Train the RNN to learn optimal weight matrices.
    
    Args:
        loss_option: 'a' or 'b' - which loss formulation to use
        n_epochs: number of training epochs
        lr: learning rate
        fixed_memory: pregenerated fixed memory pattern to train on
        activation: 'tanh', 'relu', 'linear', or 'sigmoid'
    """
    print(f"\n{'='*80}")
    print(f"TRAINING RNN WITH LOSS OPTION {loss_option.upper()} - ACTIVATION: {activation.upper()}")
    print(f"{'='*80}\n")
    print(f"Network size: {N_NEURONS}")
    print(f"Activation: {activation}")
    print(f"Total time: {T_TOTAL}s ({TIMESTEPS_TOTAL} steps)")
    print(f"Presentation: {T_PRESENTATION}s ({TIMESTEPS_PRESENTATION} steps)")
    print(f"Training mode: {BATCH_SIZE} FIXED MEMORIES (no noise)")
    print(f"Loss weights: alpha={ALPHA_RETENTION}, beta={BETA_REACTION}")
    print(f"L2 regularization: lambda={L2_LAMBDA}\n")
    
    # Initialize model with specified activation
    model = VanillaRNN(N_NEURONS, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Choose loss function
    loss_fn = compute_loss_option_a if loss_option == 'a' else compute_loss_option_b
    
    # Use the passed fixed memory pattern
    if fixed_memory is None:
        print("Generating fixed memory pattern...")
        fixed_memory = generate_memory_patterns(1, N_NEURONS)
    print(f"Memory pattern (first 10 values): {fixed_memory[0, :10].tolist()}\n")
    
    # Training history
    history = {
        'total_loss': [],
        'retention_loss': [],
        'reaction_loss': [],
        'avg_retention': [],
        'avg_reaction': [],
    }
    
    # Training loop
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Randomly select one memory from the batch each epoch
        mem_idx = epoch % fixed_memory.shape[0]  # Cycle through memories
        memory_to_train = fixed_memory[mem_idx:mem_idx+1, :]  # (1, n_neurons)
        
        # Generate input sequence with single memory
        input_seq = generate_input_sequence(
            memory_to_train, 
            TIMESTEPS_PRESENTATION, 
            TIMESTEPS_TOTAL
        )
        
        # Forward pass
        states = model(input_seq)
        
        # Compute overlaps with the trained memory
        overlaps = compute_overlap(states, memory_to_train)
        
        # Compute loss
        loss, metrics = loss_fn(
            overlaps, 
            ALPHA_RETENTION, 
            BETA_REACTION, 
            TIMESTEPS_PRESENTATION
        )
        
        # Add fixed-point loss (enforce patterns are attractors)
        fp_loss = compute_fixed_point_loss(model, memory_to_train, gamma=0.5)
        
        # Add L2 regularization
        l2_reg = L2_LAMBDA * (torch.sum(model.W_self ** 2) + torch.sum(model.W_input ** 2))
        total_loss = loss + fp_loss + l2_reg
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Record history
        history['total_loss'].append(total_loss.item())
        history['retention_loss'].append(metrics['retention_loss'])
        history['reaction_loss'].append(metrics['reaction_loss'])
        
        if loss_option == 'a':
            history['avg_retention'].append(metrics['avg_retention_overlap'])
            history['avg_reaction'].append(metrics['avg_max_presentation_overlap'])
        else:
            history['avg_retention'].append(metrics['min_retention_overlap'])
            history['avg_reaction'].append(metrics['max_presentation_overlap'])
        
        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:5d}/{n_epochs} | "
                  f"Loss: {total_loss.item():.4f} | "
                  f"Retention: {history['avg_retention'][-1]:.3f} | "
                  f"Reaction: {history['avg_reaction'][-1]:.3f}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")
    
    return model, history


# ============================================================================
# Visualization
# ============================================================================
def plot_training_history(history, loss_option):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('Total Loss (including L2 reg)', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Individual losses
    axes[0, 1].plot(history['retention_loss'], label='Retention')
    axes[0, 1].plot(history['reaction_loss'], label='Reaction')
    axes[0, 1].set_title('Loss Components', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Retention performance
    axes[1, 0].plot(history['avg_retention'])
    metric_name = 'Average' if loss_option == 'a' else 'Minimum'
    axes[1, 0].set_title(f'{metric_name} Retention Overlap', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Overlap')
    axes[1, 0].axhline(y=RETENTION_THRESHOLD, color='r', linestyle='--', 
                       label=f'Threshold ({RETENTION_THRESHOLD})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reaction performance
    axes[1, 1].plot(history['avg_reaction'])
    metric_name = 'Max Presentation' if loss_option == 'a' else 'Max Presentation'
    axes[1, 1].set_title(f'{metric_name} Overlap', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Overlap')
    axes[1, 1].axhline(y=RESPONSE_THRESHOLD, color='r', linestyle='--', 
                       label=f'Threshold ({RESPONSE_THRESHOLD})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = EXPERIMENT_DIR / f'training_history_option_{loss_option}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def visualize_learned_weights(model, loss_option):
    """Visualize the learned weight matrices."""
    W_self = model.W_self.detach().cpu().numpy()
    W_input = model.W_input.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # W_self
    im1 = axes[0].imshow(W_self, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Learned W_self (Recurrent)', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Neuron j')
    axes[0].set_ylabel('Neuron i')
    plt.colorbar(im1, ax=axes[0], label='Weight')
    
    # W_input
    im2 = axes[1].imshow(W_input, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Learned W_input (External)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Input j')
    axes[1].set_ylabel('Neuron i')
    plt.colorbar(im2, ax=axes[1], label='Weight')
    
    plt.tight_layout()
    
    filename = EXPERIMENT_DIR / f'learned_weights_option_{loss_option}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    # Print statistics
    print(f"\nWeight Statistics:")
    print(f"  W_self  - Mean: {W_self.mean():.4f}, Std: {W_self.std():.4f}")
    print(f"  W_input - Mean: {W_input.mean():.4f}, Std: {W_input.std():.4f}")


def test_model(model, loss_option, fixed_memory=None):
    """Test the trained model on the trained memory pattern."""
    print(f"\n{'='*80}")
    print("TESTING TRAINED MODEL ON THE LEARNED MEMORY")
    print(f"{'='*80}\n")
    
    # Use the same fixed memory it was trained on
    if fixed_memory is None:
        memory = generate_memory_patterns(1, N_NEURONS)
    else:
        memory = fixed_memory
    input_seq = generate_input_sequence(memory, TIMESTEPS_PRESENTATION, TIMESTEPS_TOTAL)
    
    # Run model
    with torch.no_grad():
        states = model(input_seq)
        overlaps = compute_overlap(states, memory)
    
    # Plot results
    time = np.arange(TIMESTEPS_TOTAL) * DT
    
    # Handle multiple memories
    n_memories = overlaps.shape[0]
    
    fig, axes = plt.subplots(n_memories, 1, figsize=(12, 5*n_memories))
    if n_memories == 1:
        axes = [axes]
    
    retention_list = []
    reaction_list = []
    
    for mem_idx in range(n_memories):
        overlap_curve = overlaps[mem_idx].cpu().numpy()
        retention_overlap = overlap_curve[TIMESTEPS_PRESENTATION:].mean()
        max_presentation_overlap = overlap_curve[:TIMESTEPS_PRESENTATION].max()
        
        retention_list.append(retention_overlap)
        reaction_list.append(max_presentation_overlap)
        
        axes[mem_idx].plot(time, overlap_curve, linewidth=2, label=f'Memory {mem_idx+1}')
        axes[mem_idx].axvline(x=T_PRESENTATION, color='gray', linestyle='--', label='End of presentation')
        axes[mem_idx].axhline(y=RESPONSE_THRESHOLD, color='r', linestyle='--', alpha=0.5, label='Response threshold')
        axes[mem_idx].axhline(y=RETENTION_THRESHOLD, color='orange', linestyle='--', alpha=0.5, label='Retention threshold')
        axes[mem_idx].set_xlabel('Time (s)', fontweight='bold')
        axes[mem_idx].set_ylabel('Overlap', fontweight='bold')
        axes[mem_idx].set_title(f'Memory {mem_idx+1}: Retention={retention_overlap:.3f}, Reaction={max_presentation_overlap:.3f}', 
                                fontweight='bold', fontsize=12)
        axes[mem_idx].legend()
        axes[mem_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = EXPERIMENT_DIR / f'test_trajectory_option_{loss_option}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()
    
    # Print metrics
    print(f"Test Results:")
    for mem_idx in range(n_memories):
        print(f"  Memory {mem_idx+1} - Reaction: {reaction_list[mem_idx]:.3f}, Retention: {retention_list[mem_idx]:.3f}")
    print(f"  Average Reaction: {np.mean(reaction_list):.3f}")
    print(f"  Average Retention: {np.mean(retention_list):.3f}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Import from hnn_replica/src
    from pathlib import Path
    import sys
    SRC = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(SRC))

    from rnn_weight_learning import (
        generate_memory_patterns,
        N_NEURONS,
        BATCH_SIZE,
        train_rnn,
        plot_training_history,
        visualize_learned_weights,
        test_model,
    )

    print("\n" + "="*80)
    print("RNN WEIGHT LEARNING RUNNER (tanh, 2 memories)")
    print("="*80)

    fixed_memory = generate_memory_patterns(BATCH_SIZE, N_NEURONS)
    print("\n>>> Training (tanh)...\n")
    model, history = train_rnn(loss_option='a', fixed_memory=fixed_memory, activation='tanh')
    plot_training_history(history, 'option_a_tanh')
    visualize_learned_weights(model, 'option_a_tanh')
    test_model(model, 'option_a_tanh', fixed_memory)

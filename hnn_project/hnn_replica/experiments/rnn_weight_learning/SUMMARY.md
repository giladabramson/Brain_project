# RNN Weight Learning: Progress Summary

Date: 2026-01-07

## Goals
- Learn `W_self` (recurrent) and `W_input` (external) matrices that maximize:
  - Reaction: high overlap during stimulus presentation
  - Retention: high overlap after stimulus ends
- Use a vanilla RNN with fixed overlap readout (no trainable decoder), guided by HNN settings.

## Key Steps & Findings

1. Initial Implementation (single memory)
- Built vanilla RNN: `h_t = tanh(W_self h_{t-1} + W_input x_t)`.
- Loss Option A: average retention + max presentation.
- Bug found: memories were re-sampled every epoch → impossible to converge.
- Fix: train on one fixed memory across all epochs.
- Result: perfect reaction and retention (≈100%) on single memory.

2. Two Memories (batch of 2)
- Training on both memories simultaneously (batch) → average retention ≈ 56%, reaction ≈ 100%.
- Per-memory: one memory retained well (~75%), the other poorly (~37%).
- Insight: learning multi-pattern stable attractors via gradient descent is hard with tanh dynamics.

3. Activation Sweep
- Tested `tanh`, `relu`, `linear`, `sigmoid`.
- `tanh`: best stability (≈56% retention before improvements).
- `relu`, `linear`: diverged (NaNs / exponential growth) with recurrence.
- `sigmoid`: stable but poor retention (~31%).
- Conclusion: `tanh` is appropriate; the issue is not activation alone.

4. Training Regimen Change
- Alternate memories per epoch (train one at a time): focus improves optimization signal.

5. Structural Improvements (breakthrough)
- Enforce symmetric `W_self` (Hopfield-like energy landscape).
- Add explicit fixed-point loss: minimize `||tanh(W_self m) - m||^2`.
- Reduce `L2` regularization further to allow stronger stabilizing weights.
- Result: tanh + symmetry + fixed-point loss → reaction ≈ 100%, retention ≈ 99.8% for both memories.

## Current Implementation
- Consolidated module: `src/rnn_weight_learning.py`
  - Symmetric `W_self` via property over parameter `W_self_lower`.
  - Losses: Option A + fixed-point loss.
  - Alternating-memory training loop.
  - Plotting and testing helpers.
- Runner: `experiments/train_rnn_weights.py` imports module and runs tanh, 2 memories.

## Recommended Next Steps
- Compare learned `W_self` to analytical Hopfield weight matrix (`sum m_i ⊗ m_i / N`).
- Scale to more memories (e.g., 3–5) and observe capacity vs retention.
- Introduce controlled noise during presentation to test robustness.
- Explore different fixed-point weights and retention thresholds.

## Results Artifacts
- Saved under `experiments/rnn_weight_learning/`:
  - `training_history_option_a_tanh.png`
  - `learned_weights_option_a_tanh.png`
  - `test_trajectory_option_a_tanh.png`

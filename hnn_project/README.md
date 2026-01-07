# Hopfield Neural Network Project

This project contains two versions of the Hopfield Neural Network implementation:

## Structure

### `hnn_replica/` - Active Development Version
The main working version with ongoing improvements and experiments.

**Core Components (`src/`):**
- `HNN_Gen.py` - Hopfield network class with memory pattern generation
- `Eul_May.py` - Euler-Maruyama SDE integrator with noise handling  
- `HNPlot.py` - Plotting and CSV export utilities
- `MainHNN_SHB.py` - Main experiment orchestration script

**Experiments (`experiments/`):**
- `sweep_population.py` - Population size parameter sweep
- `sweep_population_multiseed.py` - Multi-seed population experiments
- `sweep_ws.py` - Short-term weight parameter sweep
- `sweep_ws_extended.py` - Extended weight parameter analysis
- `sweep_hyper_combo.py` - Combined hyperparameter search
- `run_grid_search.py` - Systematic grid search over parameters

**Outputs:**
- `outputs/` - Timestamped experiment results
- `data/` - CSV files with trajectories and error bands
- `figures/` - PDF visualizations

### `hnn_original/` - Reference Implementation
Original version kept for comparison and validation.

## Running Experiments

### Basic Usage
```bash
cd hnn_replica/src
python MainHNN_SHB.py
```

### Environment Variables
Control experiment parameters via environment variables:
- `HNN_N` - Population size (default: 64)
- `HNN_P` - Number of memory patterns
- `W_EXTERNAL` - External drive weight (default: 1.5)
- `W_S` - Short-term weight scalar (default: 1.0)
- `NOISE_LEVEL` - Noise magnitude (default: 0.0)
- `TRIAL_SAMPLES` - Number of trials per condition
- `INPUT_MODE` - 'deterministic' or other modes

### Parameter Sweeps
```bash
cd experiments

# Sweep population sizes
python sweep_population.py --n-values 32,64,128 --trials 12

# Sweep weight parameters
python sweep_ws.py

# Multi-seed experiments
python sweep_population_multiseed.py
```

### Grid Search
```bash
cd experiments
python run_grid_search.py
```
Results saved to `grid_search_results/` with CSV summaries.

## Results

Experimental outputs are organized by timestamp in `outputs/YYYYMMDD-HHMMSS-microseconds/`:
- `metrics.json` - Quantitative performance metrics
- `metadata.json` - Configuration parameters
- Individual trial data and plots (if enabled)

## Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- numpy
- matplotlib
- (see requirements.txt for complete list)

## Notes

Both `hnn_replica/` and `hnn_original/` are maintained to:
- Track improvements and changes
- Validate new features against baseline
- Support reproducibility of earlier results

See `FILE_OVERVIEW.md` in each directory for detailed file descriptions.

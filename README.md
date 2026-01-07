# Brain Project Workspace

This workspace contains multiple neuroscience and computational modeling projects.

## Project Organization

### 📊 HNN Project (`hnn_project/`)
Hopfield Neural Network implementations and experiments.

- **`hnn_replica/`** - Main working version with improvements and experiments
  - `src/` - Core HNN implementation (HNN_Gen, Eul_May, HNPlot, MainHNN_SHB)
  - `experiments/` - Parameter sweeps and grid search scripts
  - `outputs/` - Simulation results organized by timestamp
  - `data/` - CSV trajectories and error bands
  - `figures/` - Generated plots and visualizations

- **`hnn_original/`** - Reference implementation for comparison
  - Same structure as replica for easy comparison

### 🧠 Pareto fMRI Project (`pareto_fmri/`)
Analysis toolkit for Pareto optimality in functional brain connectomes.

- `data_preperation/` - Data preprocessing pipelines (HCP, ADNI, ADHD-200)
- `figures/` - Figure generation scripts for publications
- `ParTI-py/` - Pareto optimality analysis tools
- `utils/` - Shared utilities and constants

See [pareto_fmri/README.md](pareto_fmri/README.md) for detailed usage.

### ⚛️ Statistical Physics (`statistical_physics_exercises/`)
Numerical explorations of statistical physics concepts.

- `src/` - Simulation modules
- `notebooks/` - Jupyter notebooks for interactive work
- `data/` - Input datasets
- `tests/` - Unit tests

### 🛠️ Tools (`tools/`)
Shared utilities across projects.

- `gpt_helpers/` - Node.js GPT-based code assistants

## Getting Started

Each project has its own `requirements.txt`. Install dependencies for the project you're working on:

```bash
# For HNN project
pip install -r hnn_project/hnn_replica/requirements.txt

# For Pareto fMRI
pip install -r pareto_fmri/requirements.txt

# For Statistical Physics
pip install -r statistical_physics_exercises/requirements.txt
```

## Environment

- Python 3.10+ recommended
- Virtual environment: `.venv/` (git-ignored)
- Conda environment: `.conda/` (git-ignored)

## Quick Reference

### HNN Experiments
```bash
# Run a parameter sweep
cd hnn_project/hnn_replica/experiments
python sweep_population.py --n-values 32,64,128

# Run grid search
python run_grid_search.py
```

### Pareto Analysis
```bash
cd pareto_fmri
# See README.md for data preparation and figure generation
```

---

**Last organized:** January 7, 2026

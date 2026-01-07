# Code Organization Summary
## January 7, 2026

### ✅ What Was Organized

Successfully reorganized the Brain_Project workspace into a clean, logical structure.

---

## 📂 New Structure

```
Brain_Project/
├── .gitignore              [NEW] - Git ignore patterns
├── README.md               [NEW] - Main workspace documentation
│
├── hnn_project/            [NEW] - Hopfield Neural Network workspace
│   ├── README.md           [NEW] - HNN project documentation
│   │
│   ├── hnn_replica/        [ORGANIZED] - Main working version
│   │   ├── src/            - Core HNN code (HNN_Gen, Eul_May, HNPlot, MainHNN_SHB)
│   │   ├── experiments/    - Sweep scripts & run_grid_search.py
│   │   ├── data/           - CSV trajectories and results
│   │   ├── figures/        - PDF plots
│   │   ├── outputs/        - Timestamped experiment results
│   │   ├── requirements.txt
│   │   └── FILE_OVERVIEW.md
│   │
│   └── hnn_original/       [ORGANIZED] - Reference implementation
│       ├── src/            - Original core code
│       ├── experiments/    - (empty - no sweep scripts in original)
│       ├── data/           - Original CSV files
│       ├── figures/        - Original PDF plots
│       ├── outputs/        - Original results
│       ├── reports/        - Hyperparameter analysis docs
│       ├── scripts/        - Utility scripts (md_to_pdf.py)
│       ├── requirements.txt
│       └── FILE_OVERVIEW.md
│
├── pareto_fmri/            [RENAMED from pareto/]
│   ├── data_preperation/   - HCP, ADNI, ADHD-200 preprocessing
│   ├── figures/            - Publication figure generation
│   ├── ParTI-py/           - Pareto analysis tools
│   ├── utils/              - Shared utilities
│   ├── requirements.txt
│   ├── README.md
│   └── CONTENTS.md
│
├── statistical_physics_exercises/  [UNCHANGED]
│   ├── src/
│   ├── notebooks/
│   ├── data/
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
└── tools/                  [NEW] - Shared utilities
    └── gpt_helpers/        [CONSOLIDATED from 3 locations]
        ├── gpt-chat.js
        ├── gpt-helper.js
        ├── package.json
        └── node_modules/

```

---

## 🗑️ Cleaned Up

1. **Removed duplicate gpt_helpers**: Consolidated from 3 locations (root, hnn_replica, original_hnn) → `tools/gpt_helpers/`
2. **Removed all __pycache__**: Python bytecode cache directories throughout workspace
3. **Removed duplicate pareto**: Nested copies in `original_hnn/pareto/` and `hnn_replica/pareto/`
4. **Removed empty directories**: `hnn_replica/` and `original_hnn/` (after moving content)
5. **Cleaned IDE folders**: Extra `.idea/` and `.vscode/` in subdirectories

---

## 📋 Key Changes

### HNN Project Organization
- **Separated concerns**: Core code (`src/`), experiments, data, figures
- **Both versions preserved**: `hnn_replica/` (working) and `hnn_original/` (reference)
- **Experiments consolidated**: All sweep scripts and grid search in `experiments/`
- **Clear hierarchy**: Easy to find code vs results vs documentation

### Pareto Project
- **Renamed for clarity**: `pareto/` → `pareto_fmri/` (clearer project name)
- **Removed duplicates**: Cleaned nested copies from HNN directories

### Shared Tools
- **Centralized helpers**: All gpt_helpers now in `tools/gpt_helpers/`
- **Single source**: No more maintaining 3 copies of the same scripts

---

## 📌 Important Notes

1. **Both HNN versions preserved**: 
   - `hnn_replica/` - Your active development version with improvements
   - `hnn_original/` - Reference implementation for validation

2. **All data preserved**: CSV files, figures, and outputs moved intact

3. **Experiment scripts organized**: All in `hnn_replica/experiments/` including:
   - `run_grid_search.py`
   - `sweep_population.py`
   - `sweep_population_multiseed.py`
   - `sweep_ws.py`
   - `sweep_ws_extended.py`
   - `sweep_hyper_combo.py`

4. **Git-friendly**: Added `.gitignore` with patterns for:
   - Python cache (`__pycache__/`)
   - Virtual environments (`.venv/`, `.conda/`)
   - IDE folders (`.idea/`, `.vscode/`)
   - Large output files

---

## 🚀 Next Steps

You can now:
- Run experiments from `hnn_project/hnn_replica/experiments/`
- Find core code in `hnn_project/hnn_replica/src/`
- Compare with original in `hnn_project/hnn_original/`
- Access pareto tools in `pareto_fmri/`
- Use shared helpers from `tools/gpt_helpers/`

All documentation is in place with README files at appropriate levels!

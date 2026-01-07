# Brain Project File Overview

## Root Level
- `Eul_May.py` — Euler–Maruyama integrator for the Hopfield network SDE, including additive and multiplicative noise handling.
- `HNN_Gen.py` — Defines the `HNN` class used to create Hopfield networks, generate binary memory patterns, and set initial conditions.
- `HNPlot.py` — Plotting and export utilities that turn simulation outputs into overlap figures plus CSV error bands.
- `MainHNN_SHB.py` — Main experiment script wiring together network generation, stochastic integration, and plotting for stimulus-driven Hopfield runs.
- `TrajectoriesAdd_PP_1_m.csv` — Saved mean overlap trajectories from additive-input simulations with a single stimulus sequence.
- `TrajectoriesAdd_noise_2_m.csv` — Exported additive-input overlaps for noisy trials (variant “2_m”).
- `TrajectoriesAdd_noise_3_m.csv` — Exported additive-input overlaps for noisy trials (variant “3_m”).
- `TrajectoriesMult_noise_3_c.csv` — Exported multiplicative-input overlaps for noisy trials (variant “3_c”).
- `TrajectoryOverlap_Add_noise_2_m.pdf` — Figure showing additive-input overlap curves for the matching CSV run.
- `TrajectoryOverlap_Add_noise_3_m.pdf` — Figure showing additive-input overlap curves for the “3_m” noisy condition.
- `TrajectoryOverlap_Mult_noise_3_c.pdf` — Figure showing multiplicative-input overlap curves for the “3_c” noisy condition.
- `__pycache__/` — Python bytecode cache produced when running the Hopfield scripts.
- `err1.csv` — Lower error-band values (mean minus std) exported alongside overlap trajectories.
- `err2.csv` — Upper error-band values (mean plus std) exported alongside overlap trajectories.
- `gpt-chat.js` — Node.js CLI that opens a terminal chat with the OpenAI API using the `openai` package.
- `gpt-helper.js` — Node.js helper that can ask GPT for explanations or rewrite files via command-line commands.
- `node_modules/` — Installed npm dependencies (primarily `openai`); auto-generated and not project logic.
- `package-lock.json` — Locked dependency graph for the Node.js helpers.
- `package.json` — npm manifest enabling the GPT helper utilities and declaring the `openai` dependency.
- `pareto/` — Standalone toolkit for reproducing Pareto optimality analyses on fMRI datasets.

## pareto Top-Level
- `pareto/README.md` — High-level instructions for recreating the Pareto fMRI paper’s figures and analyses.
- `pareto/CONTENTS.md` — Auto-generated inventory of the `pareto` directory.
- `pareto/requirements.txt` — Python dependency pins for the Pareto workflows.
- `pareto/comdepri.mplstyle` — Matplotlib style sheet applied across Pareto figures.

## pareto/utils/
- `pareto/utils/__init__.py` — Package marker for shared Pareto utilities.
- `pareto/utils/constants.py` — Central configuration (labels, palettes, filesystem paths, feature metadata) used throughout the Pareto code.
- `pareto/utils/utils.py` — Statistical helpers and reporting utilities (correlations, regression summaries, plotting aids, table formatting).
- `pareto/utils/__pycache__/` — Bytecode cache for the utilities package.

## pareto/data_preperation/
- `pareto/data_preperation/collect_pvals.py` — CLI that gathers archetype p-values from ParTI result folders into a single CSV.
- `pareto/data_preperation/combine_connectomes.py` — Script that averages multiple connectivity feature tables into one combined dataset.
- `pareto/data_preperation/create_random_data.py` — Generates synthetic correlation features from random covariance models for benchmarking.
- `pareto/data_preperation/data_peeling.py` — Iteratively peels convex-hull layers from connectivity data and records layer membership and volumes.
- `pareto/data_preperation/download_data.py` — AWS S3 downloader for HCP resting-state or task data with support for multiple sessions/phases.
- `pareto/data_preperation/filter_enrichment_features.py` — Filters and groups behavioral enrichment variables, removing highly correlated fields.

## pareto/data_preperation/adhd/
- `pareto/data_preperation/adhd/collect_subjects_adhd.py` — Builds subject lists for the ADHD-200 cohort.
- `pareto/data_preperation/adhd/download_adhd_data.py` — Fetches ADHD-200 derivatives from the public S3 bucket.
- `pareto/data_preperation/adhd/normalize_adhd.py` — Ensures ADHD data follow the expected directory structure for subsequent preprocessing.

## pareto/data_preperation/adni/
- `pareto/data_preperation/adni/collect_subjects_adni.py` — Aggregates ADNI participant metadata for processing.
- `pareto/data_preperation/adni/create_options_adni.py` — Generates subject-specific `options.yaml` files reflecting ADNI scan parameters.
- `pareto/data_preperation/adni/move_invalid_sessions.py` — Moves or flags ADNI sessions that fail preprocessing quality checks.

## pareto/data_preperation/matrices2features/
- `pareto/data_preperation/matrices2features/preprocessing_HCP_matrices.py` — Converts HCP connectivity matrices into feature vectors used by ParTI.
- `pareto/data_preperation/matrices2features/preprocessing_schaefer_matrices.py` — Extracts features from Schaefer-parcellated matrices.
- `pareto/data_preperation/matrices2features/preprocessing_simple_matrices.py` — Generic matrix-to-feature conversion for simpler datasets.

## pareto/data_preperation/preprocess_bids_vol/
- `pareto/data_preperation/preprocess_bids_vol/CiftiUtils.py` — CIFTI helper routines for volumetric BIDS preprocessing.
- `pareto/data_preperation/preprocess_bids_vol/CorrMatrixCreator.py` — Builds volumetric correlation matrices split by functional networks.
- `pareto/data_preperation/preprocess_bids_vol/options.yaml` — Template configuration describing preprocessing parameters.
- `pareto/data_preperation/preprocess_bids_vol/opts_utils.py` — Utilities for reading and writing preprocessing option files.
- `pareto/data_preperation/preprocess_bids_vol/ParetoMain.py` — Entry point orchestrating volumetric BIDS preprocessing and matrix generation.
- `pareto/data_preperation/preprocess_bids_vol/RestPreprocessing.py` — Pipeline steps for denoising and preparing rest fMRI volumes.

## pareto/data_preperation/preprocess_hcp_surface/
- `pareto/data_preperation/preprocess_hcp_surface/CiftiUtils.py` — Surface-mode CIFTI utilities for HCP data.
- `pareto/data_preperation/preprocess_hcp_surface/CorrMatrixCreator.py` — Produces surface-based 17×17 correlation matrices.
- `pareto/data_preperation/preprocess_hcp_surface/HCP1200_IDs.txt` — Subject ID list used when downloading or preprocessing HCP surface data.
- `pareto/data_preperation/preprocess_hcp_surface/options.yaml` — Default preprocessing configuration for surface data.
- `pareto/data_preperation/preprocess_hcp_surface/opts_utils.py` — Helpers for managing surface preprocessing options.
- `pareto/data_preperation/preprocess_hcp_surface/ParetoMain.py` — Surface preprocessing driver mirroring the volumetric pipeline.
- `pareto/data_preperation/preprocess_hcp_surface/RestPreprocessing.py` — Resting-state surface processing steps.

## pareto/data_preperation/preprocess_hcp_vol/
- `pareto/data_preperation/preprocess_hcp_vol/CiftiUtils.py` — HCP volumetric CIFTI helpers.
- `pareto/data_preperation/preprocess_hcp_vol/CorrMatrixCreator.py` — Builds volumetric correlation matrices for HCP datasets.
- `pareto/data_preperation/preprocess_hcp_vol/HCP1200_IDs.txt` — Subject ID list for the volumetric pipeline.
- `pareto/data_preperation/preprocess_hcp_vol/options.yaml` — Configuration template for volumetric preprocessing.
- `pareto/data_preperation/preprocess_hcp_vol/opts_utils.py` — Option management utilities shared inside the volumetric workflow.
- `pareto/data_preperation/preprocess_hcp_vol/ParetoMain.py` — Alternate driver specialized for volumetric HCP preprocessing.
- `pareto/data_preperation/preprocess_hcp_vol/RestPreprocessing.py` — Resting-state volumetric cleaning steps.

## pareto/data_preperation/tasks/
- `pareto/data_preperation/tasks/create_time_locked_dynamics.py` — Produces event-aligned connectivity windows for task fMRI.
- `pareto/data_preperation/tasks/extract_brain_regressors.py` — Pulls nuisance regressors (WM/CSF) from task data.
- `pareto/data_preperation/tasks/prepare_task_conditions.py` — Organizes task condition metadata to feed later analyses.

## pareto/data_preperation/windows/
- `pareto/data_preperation/windows/combine_windows.py` — Merges left/right phase windowed connectivity tables.
- `pareto/data_preperation/windows/generate_individual_windows_connectome.py` — Builds per-subject sliding-window connectomes.

## pareto/figures/
- `pareto/figures/arch3_anatomical_analysis.py` — Anatomical comparisons for the three-arch Pareto model.
- `pareto/figures/create_graph_measures_table.py` — Computes graph-theoretic measures used across multiple figures.
- `pareto/figures/figs_archs.py` — Generates the main multi-arch structural plots.
- `pareto/figures/fig_adhd.py` — Recreates ADHD-related figures (group comparisons, enrichment summaries).
- `pareto/figures/fig_adni.py` — Builds ADNI cohort figures, including diagnostic group analyses.
- `pareto/figures/fig_enrichment.py` — Visualizes enrichment statistics for behavioral and clinical measures.
- `pareto/figures/fig_model.py` — Simulates an autapse model and produces the corresponding manuscript panels.
- `pareto/figures/fig_task.py` — Generates HCP task-based Pareto figures and time-locked dynamics panels.
- `pareto/figures/make_SI.py` — Produces supplementary information figures; sections enabled via manual toggles.
- `pareto/figures/plotters.py` — Shared plotting helpers (layouts, annotations, save helpers) used by the figure scripts.
- `pareto/figures/task_discrimination.py` — Implements task-condition decoding analyses for manuscript panels.

## pareto/ParTI-py/
- `pareto/ParTI-py/README.md` — Overview and usage instructions for the ParTI analysis toolkit.
- `pareto/ParTI-py/comdepri.mplstyle` — Matplotlib style specific to ParTI figure aesthetics.
- `pareto/ParTI-py/distance_to_archetypes.py` — Distance computations between datapoints and learned archetypes.
- `pareto/ParTI-py/main.py` — Command-line interface that loads features, runs ParTI archetype discovery, and saves outputs.
- `pareto/ParTI-py/parti.py` — Core ParTI implementation (archetype finding, bootstrapping, enrichment tests, permutation logic).
- `pareto/ParTI-py/requirements.txt` — Dependencies required to run the ParTI toolkit standalone.
- `pareto/ParTI-py/stat_utils.py` — Statistical helper functions (confidence ellipses, FDR correction) used by ParTI plots.
- `pareto/ParTI-py/unmixing.py` — SISAL/VCA unmixing algorithms and supporting math utilities.
- `pareto/ParTI-py/viz_utils.py` — Visualization and export routines for ParTI results (figures, CSV exports, palettes).

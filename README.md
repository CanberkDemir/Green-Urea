# Script Runner Guide

This README focuses on the scripts and notebooks that a person is likely to run from this workspace. It is not a full code reference; it is a practical guide for getting data, training surrogates, solving the IPPS model, and plotting results.



Notes:

- The Aspen runners require Aspen Plus on Windows.
- The IPPS script uses Pyomo and Gurobi.
- Some visualization paths use Graphviz.

## Recommended Run Order

For the current ReLU-surrogate workflow, the usual order is:

1. Generate ammonia Aspen data with `run_ammoniaF_grid.py`
2. Generate urea Aspen data with `run_ureaF_grid.py`
3. Train and plot surrogates with `surrogate_train_and_plot.py`
4. Optionally inspect a surrogate with `plot_surrogate_graphviz_surface.py`
5. Solve the planning model with `ipps_green_urea_fixed_operating_point_relu_omlt.py`
6. Plot or inspect electricity balance with `plot_electricity_balance.ipynb`

## Main Scripts You Might Run

### 1. `run_ammoniaF_grid.py`

Purpose: build a full-factorial Aspen case grid for the ammonia flowsheet and run all cases, writing live results to CSV.

Typical commands:

```powershell
python .\run_ammoniaF_grid.py --init-only
python .\run_ammoniaF_grid.py --generate-only --resolution 5
python .\run_ammoniaF_grid.py --resolution 5 --visible --case-timeout 600
```

What it creates/uses:

- Uses `ammoniaF.bkp`
- Creates or updates `ammoniaF_inputs.csv`
- Creates or updates `ammoniaF_outputs.csv`
- Writes `ammoniaF_case_grid.csv`
- Writes `ammoniaF_results_live.csv`

Important options:

- `--init-only`: create blank template CSVs and stop
- `--generate-only`: build the case grid and stop before Aspen runs
- `--resolution N`: points per active input
- `--visible`: show Aspen while running
- `--case-timeout SEC`: restart Aspen if a case hangs too long
- `--force-kill-aspen-on-timeout`: use only if Aspen is frozen and you have no other Aspen work open
- `--overwrite-templates`: replace existing input/output templates

Notes:

- The script auto-computes some derived ammonia-related quantities instead of treating them as independent inputs/outputs.
- It converts ammonia heat-duty outputs from `cal/s` to `MMkcal/hr` before writing results.

### 2. `run_ureaF_grid.py`

Purpose: build a full-factorial Aspen case grid for the urea flowsheet and run all cases, writing live results to CSV.

Typical commands:

```powershell
python .\run_ureaF_grid.py --init-only
python .\run_ureaF_grid.py --generate-only --resolution 5
python .\run_ureaF_grid.py --resolution 5 --visible --case-timeout 600
```

What it creates/uses:

- Uses `ureaF.bkp`
- Creates or updates `ureaF_inputs.csv`
- Creates or updates `ureaF_outputs.csv`
- Writes `ureaF_case_grid.csv`
- Writes `ureaF_results_live.csv`

Important options:

- Same CLI pattern as `run_ammoniaF_grid.py`

Notes:

- The script auto-computes some urea component-flow and product-output values.
- It currently leaves urea heat duties in the Aspen-reported basis when `UREA_HEAT_DUTIES_ALREADY_MMKCAL_PER_HR=True`.

### 3. `surrogate_train_and_plot.py`

Purpose: train or reuse the current ReLU ANN surrogate models based on the per-output settings in `UNIT_TRAINING_OVERRIDES`, then generate the visualization assets.

Typical commands:

```powershell
python .\surrogate_train_and_plot.py
python .\surrogate_train_and_plot.py --hide-other-points
```

What it creates/uses:

- Reads `ammoniaF_results_live.csv`
- Reads `ureaF_results_live.csv`
- Reads `ammoniaF_case_grid.csv` when available
- Writes trained bundles into `trained_unit_surrogates\`
- Writes plot assets under `surrogate_visualizations\` and/or `trained_unit_surrogates\...`

When to use it:

- Use this as the main training entry point for the current Keras/ReLU surrogate workflow.
- Adjust `UNIT_TRAINING_OVERRIDES` inside the script if you want some outputs retrained and others reused from saved bundles.

### 4. `plot_surrogate_graphviz_surface.py`

Purpose: render Graphviz network diagrams and data-vs-surface views for one surrogate unit or one specific output slice.

Typical commands:

```powershell
python .\plot_surrogate_graphviz_surface.py --unit ammoniaF_unit
python .\plot_surrogate_graphviz_surface.py --unit ammoniaF_unit --output ammonia_kgph --x Ft --y Fh2
python .\plot_surrogate_graphviz_surface.py --unit ureaF_unit --output pure_urea_kgph 
```

Useful options:

- `--unit`: `ammoniaF_unit` or `ureaF_unit`
- `--output`: plot only one target
- `--x` and `--y`: choose the 2D slice axes
- `--model-dir`: choose which saved bundle directory to read
- `--out-dir`: choose where figures are written
- `--grid-points`: surface resolution
- `--max-points`: maximum scatter points shown
- `--show-other-points`: overlay off-slice points
- `--bundle-mode auto|saved|retrain`: control bundle loading/retraining behavior

### 5. `ipps_green_urea_fixed_operating_point_relu_omlt.py`

Purpose: solve the green urea IPPS planning model using the trained ReLU surrogates embedded with OMLT.

Typical command:

```powershell
python .\ipps_green_urea_fixed_operating_point_relu_omlt.py
```

What it uses:

- Trained surrogate bundles in `trained_unit_surrogates\`
- Wind data in `uk_hornsea2_wind_availability_2024.csv`

What it writes:

- A planning-results CSV such as `ipps_solution_smallhorizon_free_grid.csv`

Notes:

- This script is configured by constants near the top of the file, including horizon settings, grid mode, solver name, and warm-start behavior.
- Edit the script if you want to change `GRID_MODE`, `USE_FULL_YEAR`, `TIME_LIMIT_SEC`, or the output filename.

### 6. `plot_electricity_balance.ipynb`

Purpose: inspect the IPPS output visually, especially electricity production, consumption, hot utility, and battery behavior.

How to use it:

- Open the notebook in Jupyter or VS Code
- Point it at the desired IPPS solution CSV if needed
- Run the cells to generate the plots

Best time to use it:

- After `ipps_green_urea_fixed_operating_point_relu_omlt.py` has written a solution CSV

## Other Runnable Scripts

These can be run directly, but they are usually secondary tools rather than the main day-to-day entry points.

### `surrogate_functions.py`

Purpose: the core library for the current ReLU surrogate workflow. It can also be run directly to train both unit surrogates from the available results CSVs.

Typical command:

```powershell
python .\surrogate_functions.py
```

Use this when:

- You are training the current ReLU/Keras workflow

If you are not sure which training script to use, prefer `surrogate_train_and_plot.py` for the current workflow.

## Support Modules You Usually Do Not Run Directly

### `aspen_grid_runner.py`

Shared engine used by the Aspen runner scripts to:

- create template CSVs
- generate full-factorial case grids
- run Aspen case batches
- resume and validate live results CSVs

Most users should not call this file directly; use `run_ammoniaF_grid.py` or `run_ureaF_grid.py`.

### `plotting_compat.py`

Small compatibility/helper module used by plotting code to normalize column names and compute fallback electric-load terms.

Most users should import it indirectly through plotting workflows rather than run it directly.

## Quick Start

If you just want the shortest practical path:

```powershell
python .\run_ammoniaF_grid.py --resolution 5 --visible
python .\run_ureaF_grid.py --resolution 5 --visible
python .\surrogate_train_and_plot.py
python .\ipps_green_urea_fixed_operating_point_relu_omlt.py
```

Then open:

- `plot_electricity_balance.ipynb`
- or the files under `surrogate_visualizations\`

## Important Data Files

You will see these files appear repeatedly across the workflow:

- `ammoniaF_inputs.csv`
- `ammoniaF_outputs.csv`
- `ammoniaF_case_grid.csv`
- `ammoniaF_results_live.csv`
- `ureaF_inputs.csv`
- `ureaF_outputs.csv`
- `ureaF_case_grid.csv`
- `ureaF_results_live.csv`
- `trained_unit_surrogates\ammoniaF_unit.joblib`
- `trained_unit_surrogates\ureaF_unit.joblib`
- `uk_hornsea2_wind_availability_2024.csv`
- `ipps_solution_smallhorizon_free_grid.csv`

## Which Script Should I Use?

- Need Aspen data for ammonia: `run_ammoniaF_grid.py`
- Need Aspen data for urea: `run_ureaF_grid.py`
- Need the current surrogate workflow: `surrogate_train_and_plot.py`
- Need one focused surrogate visualization: `plot_surrogate_graphviz_surface.py`
- Need the optimization/planning solve: `ipps_green_urea_fixed_operating_point_relu_omlt.py`
- Need post-processing plots: `plot_electricity_balance.ipynb`
- Need legacy HyperplaneTree experiments: `HT_train_surrogates.py`

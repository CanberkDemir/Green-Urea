# Script Runner Guide

## Environment Setup

This project uses a local Python 3.11 virtual environment (`.venv`).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-light.txt
pip install -r requirements-heavy.txt
```

Set the Gurobi license file if needed:

```powershell
$env:GRB_LICENSE_FILE = "$PWD\gurobi.lic"
```

Notes:

- The Aspen runners require Aspen Plus on Windows.
- The IPPS script uses Pyomo and Gurobi.
- Some visualization paths use Graphviz.
- All commands below assume you run them from the repo root.

## Repo Layout

The repo is organized by pipeline stage:

```
aspen/          Aspen models, grid runners, and raw Aspen inputs/outputs (aspen/data/)
surrogates/     Surrogate training/plotting scripts and trained artifacts (surrogates/trained/)
optimization/   IPPS planning model, wind data, and solution CSVs (optimization/solutions/)
figures/        Notebook, electricity-balance views, and paper figures/scripts (figures/paper/)
```

## Recommended Run Order

For the current ReLU-surrogate workflow, the usual order is:

1. Generate ammonia Aspen data with `aspen/run_ammoniaF_grid.py`
2. Generate urea Aspen data with `aspen/run_ureaF_grid.py`
3. Train and plot surrogates with `surrogates/surrogate_train_and_plot.py`
4. Optionally inspect a surrogate with `surrogates/plot_surrogate_graphviz_surface.py`
5. Solve the planning model with `optimization/ipps_green_urea_fixed_operating_point_relu_omlt.py`
6. Plot or inspect electricity balance with `figures/plot_electricity_balance.ipynb`

## Main Scripts You Might Run

### 1. `aspen/run_ammoniaF_grid.py`

Purpose: build a full-factorial Aspen case grid for the ammonia flowsheet and run all cases, writing live results to CSV.

Typical commands:

```powershell
python .\aspen\run_ammoniaF_grid.py --init-only
python .\aspen\run_ammoniaF_grid.py --generate-only --resolution 5
python .\aspen\run_ammoniaF_grid.py --resolution 5 --visible --case-timeout 600
```

What it creates/uses:

- Uses `aspen\ammoniaF.bkp`
- Creates or updates `aspen\data\ammoniaF_inputs.csv`
- Creates or updates `aspen\data\ammoniaF_outputs.csv`
- Writes `aspen\data\ammoniaF_case_grid.csv`
- Writes `aspen\data\ammoniaF_results_live.csv`

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

### 2. `aspen/run_ureaF_grid.py`

Purpose: build a full-factorial Aspen case grid for the urea flowsheet and run all cases, writing live results to CSV.

Typical commands:

```powershell
python .\aspen\run_ureaF_grid.py --init-only
python .\aspen\run_ureaF_grid.py --generate-only --resolution 5
python .\aspen\run_ureaF_grid.py --resolution 5 --visible --case-timeout 600
```

What it creates/uses:

- Uses `aspen\ureaF.bkp`
- Creates or updates `aspen\data\ureaF_inputs.csv`
- Creates or updates `aspen\data\ureaF_outputs.csv`
- Writes `aspen\data\ureaF_case_grid.csv`
- Writes `aspen\data\ureaF_results_live.csv`

Important options:

- Same CLI pattern as `run_ammoniaF_grid.py`

Notes:

- The script auto-computes some urea component-flow and product-output values.
- It currently leaves urea heat duties in the Aspen-reported basis when `UREA_HEAT_DUTIES_ALREADY_MMKCAL_PER_HR=True`.

### 3. `surrogates/surrogate_train_and_plot.py`

Purpose: train or reuse the current ReLU ANN surrogate models based on the per-output settings in `UNIT_TRAINING_OVERRIDES`, then generate the visualization assets.

Typical commands:

```powershell
python .\surrogates\surrogate_train_and_plot.py
python .\surrogates\surrogate_train_and_plot.py --hide-other-points
```

What it creates/uses:

- Reads `aspen\data\ammoniaF_results_live.csv`
- Reads `aspen\data\ureaF_results_live.csv`
- Reads `aspen\data\ammoniaF_case_grid.csv` when available
- Writes trained bundles into `surrogates\trained\trained_unit_surrogates\`
- Writes plot assets under `surrogates\trained\surrogate_visualizations\` and/or `surrogates\trained\trained_unit_surrogates\...`

When to use it:

- Use this as the main training entry point for the current Keras/ReLU surrogate workflow.
- Adjust `UNIT_TRAINING_OVERRIDES` inside the script if you want some outputs retrained and others reused from saved bundles.

### 4. `surrogates/plot_surrogate_graphviz_surface.py`

Purpose: render Graphviz network diagrams and data-vs-surface views for one surrogate unit or one specific output slice.

Typical commands:

```powershell
python .\surrogates\plot_surrogate_graphviz_surface.py --unit ammoniaF_unit
python .\surrogates\plot_surrogate_graphviz_surface.py --unit ammoniaF_unit --output ammonia_kgph --x Ft --y Fh2
python .\surrogates\plot_surrogate_graphviz_surface.py --unit ureaF_unit --output pure_urea_kgph 
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

### 5. `optimization/ipps_green_urea_fixed_operating_point_relu_omlt.py`

Purpose: solve the green urea IPPS planning model using the trained ReLU surrogates embedded with OMLT.

Typical command:

```powershell
python .\optimization\ipps_green_urea_fixed_operating_point_relu_omlt.py
```

What it uses:

- Trained surrogate bundles in `surrogates\trained\trained_unit_surrogates\`
- Wind data in `optimization\uk_hornsea2_wind_availability_2024.csv`

What it writes:

- A planning-results CSV such as `optimization\solutions\ipps_solution_smallhorizon_free_grid.csv`

Notes:

- This script is configured by constants near the top of the file, including horizon settings, grid mode, solver name, and warm-start behavior.
- Edit the script if you want to change `GRID_MODE`, `USE_FULL_YEAR`, `TIME_LIMIT_SEC`, or the output filename.

### 6. `figures/plot_electricity_balance.ipynb`

Purpose: inspect the IPPS output visually, especially electricity production, consumption, hot utility, and battery behavior.

How to use it:

- Open the notebook in Jupyter or VS Code (its default working directory is `figures\`)
- Point it at the desired IPPS solution CSV under `..\optimization\solutions\` if needed
- Run the cells to generate the plots

Best time to use it:

- After `ipps_green_urea_fixed_operating_point_relu_omlt.py` has written a solution CSV

## Other Runnable Scripts

These can be run directly, but they are usually secondary tools rather than the main day-to-day entry points.

### `surrogates/surrogate_functions.py`

Purpose: the core library for the current ReLU surrogate workflow. It can also be run directly to train both unit surrogates from the available results CSVs.

Typical command:

```powershell
python .\surrogates\surrogate_functions.py
```

Use this when:

- You are training the current ReLU/Keras workflow

If you are not sure which training script to use, prefer `surrogates/surrogate_train_and_plot.py` for the current workflow.

## Support Modules You Usually Do Not Run Directly

### `aspen/aspen_grid_runner.py`

Shared engine used by the Aspen runner scripts to:

- create template CSVs
- generate full-factorial case grids
- run Aspen case batches
- resume and validate live results CSVs

Most users should not call this file directly; use `aspen/run_ammoniaF_grid.py` or `aspen/run_ureaF_grid.py`.

### `surrogates/plotting_compat.py`

Small compatibility/helper module used by plotting code to normalize column names and compute fallback electric-load terms.

Most users should import it indirectly through plotting workflows rather than run it directly.

## Quick Start

If you just want the shortest practical path:

```powershell
python .\aspen\run_ammoniaF_grid.py --resolution 5 --visible
python .\aspen\run_ureaF_grid.py --resolution 5 --visible
python .\surrogates\surrogate_train_and_plot.py
python .\optimization\ipps_green_urea_fixed_operating_point_relu_omlt.py
```

Then open:

- `figures\plot_electricity_balance.ipynb`
- or the files under `surrogates\trained\surrogate_visualizations\`

## Important Data Files

You will see these files appear repeatedly across the workflow:

- `aspen\data\ammoniaF_inputs.csv`
- `aspen\data\ammoniaF_outputs.csv`
- `aspen\data\ammoniaF_case_grid.csv`
- `aspen\data\ammoniaF_results_live.csv`
- `aspen\data\ureaF_inputs.csv`
- `aspen\data\ureaF_outputs.csv`
- `aspen\data\ureaF_case_grid.csv`
- `aspen\data\ureaF_results_live.csv`
- `surrogates\trained\trained_unit_surrogates\ammoniaF_unit.joblib`
- `surrogates\trained\trained_unit_surrogates\ureaF_unit.joblib`
- `optimization\uk_hornsea2_wind_availability_2024.csv`
- `optimization\solutions\ipps_solution_smallhorizon_free_grid.csv`

## Which Script Should I Use?

- Need Aspen data for ammonia: `aspen/run_ammoniaF_grid.py`
- Need Aspen data for urea: `aspen/run_ureaF_grid.py`
- Need the current surrogate workflow: `surrogates/surrogate_train_and_plot.py`
- Need one focused surrogate visualization: `surrogates/plot_surrogate_graphviz_surface.py`
- Need the optimization/planning solve: `optimization/ipps_green_urea_fixed_operating_point_relu_omlt.py`
- Need post-processing plots: `figures/plot_electricity_balance.ipynb`
- Need legacy HyperplaneTree experiments: `surrogates/HT_train_surrogates.py`

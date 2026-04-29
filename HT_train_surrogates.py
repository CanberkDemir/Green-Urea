"""
unit_surrogates.py

Two unit-level surrogate backends plus a deterministic stripper function:
    nitrate_unit(...)
    ammonia_stripper_unit(...)
    urea_unit(...)

Workflow
--------
1) Generate synthetic profiles from assumed engineering relationships.
2) Fit one HyperplaneTree per surrogate output.
3) Save/load the trained bundles.
4) Use the same callable functions in the optimization script.

Dependencies
------------
install Graphviz properly via internet

pip install numpy pandas torch scikit-learn joblib
pip install git+https://github.com/LLNL/systems2atoms

or:
pip install "hyperplanetree @ git+https://git@github.com/LLNL/systems2atoms#subdirectory=systems2atoms/hyperplanetree"

if you get error for not recent omlt and lineartree:
pip install --upgrade pip
pip install --upgrade linear-tree pyomo
pip install git+https://github.com/cog-imperial/OMLT
pip install git+https://github.com/LLNL/systems2atoms
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import json
import math
import os

import joblib
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from systems2atoms.hyperplanetree import LinearTreeRegressor, plot_surrogate_2d

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_MODEL_DIR = Path("trained_unit_surrogates")
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_GALLERY_OUT_DIR = Path("surrogate_visualizations_ht")

AMMONIAF_RESULTS_CSV = Path("ammoniaF_results_live.csv")
AMMONIAF_CASE_GRID_CSV = Path("ammoniaF_case_grid.csv")
AMMONIAF_INPUTS_CSV = Path("ammoniaF_inputs.csv")
SHOW_INTERACTIVE_3D_PLOTS = False
HT_TEST_FRACTION = 0.15
HT_RANDOM_SEED = 15
HT_ENABLE_HYPERPARAMETER_EXPLORATION = True
HT_DEFAULT_HYPERPARAMETERS = {
    "criterion": "mae",
    "max_depth": 5,
    "max_bins": 16,
    "min_impurity_decrease": -.1,
    "num_terms": 2,
    "max_weight": 1,
    "do_symmetrize": True,
    "do_scaling": True,
    "torch_device": "cpu",
    "disable_tqdm": False, # training bar animation on when False
    "ridge": 1e-5,
}
HT_MIN_SAMPLES_LEAF_SEARCH = np.logspace(-6, -1, 11).tolist()
HT_MAX_WEIGHT_SEARCH = [3, 6]

UREAF_RESULTS_CSV = Path("ureaF_results_live.csv")
UREAF_TRAIN_ALL_HEAT_DUTIES = True

AMMONIAF_RESULT_META_COLUMNS = {
    "case_id",
    "run_ok",
    "error_message",
    "elapsed_sec",
    "completed_at",
}
AMMONIAF_DUTY_COLUMNS = ["Qh1", "Qc1", "Qr1", "Qcomp"]
AMMONIAF_TOTAL_ABS_DUTY_COLUMN = "Q_total_abs"

UREAF_RESULT_META_COLUMNS = AMMONIAF_RESULT_META_COLUMNS
UREAF_EXCLUDED_HEAT_DUTY_COLUMNS = {"QB10_hot", "QB10_cold"}
UREAF_HEAT_DUTY_COLUMNS = [
    "QB3",
    "QB6",
    "QB27",
    "QB7_reb",
    "QB7_cond",
    "QB25_reb",
    "QB28_reb",
    "QB28_cond",
    "QR01",
]
UREAF_TOTAL_ABS_DUTY_COLUMN = "Q_total_abs"
MMKCAL_PER_HR_TO_KWHPH = 1_000_000.0 / 860.4206500956024

AMMONIA_COMPONENT_OUTPUT_COLUMNS = ["ammonia_kgph", "water_kgph", "electric_kwhph"]
UREA_DIRECT_OUTPUT_COLUMNS = ["pure_urea_kgph", "product_urea_wtfrac", "electric_kwhph"]
AMMONIAF_MODEL_INPUT_COLUMNS = ["Ft", "Fh2"]
AMMONIAF_FIXED_GEOMETRY = {"Kl": 10.0, "Kd": 1.0}
UREAF_RAW_INPUT_COLUMNS = ["Fnh3", "Fco2"]
UREAF_MODEL_INPUT_COLUMNS = ["Fnh3", "Fco2"]
UREAF_FIXED_GEOMETRY = {"Kl": 20.0, "Kd": 2.5}
AMMONIAF_OPERATING_INPUT_BOUNDS = {
    "Ft": (8500.0, 9080.0),
    "Fh2": (5.0, 15.0),
}
UREAF_OPERATING_INPUT_BOUNDS = {
    "Fnh3": (5.0, 12.0),
    "Fco2": (1.0, 50.0),
}
UNIT_INPUT_BOUND_OVERRIDES = {
    "ammoniaF_unit": AMMONIAF_OPERATING_INPUT_BOUNDS,
    "ureaF_unit": UREAF_OPERATING_INPUT_BOUNDS,
}

# Small module-level cache so the optimization script can just import and call.
_MODEL_CACHE: Dict[str, "UnitSurrogateBundle"] = {}


# =============================================================================
# HYPERPLANETREE IMPORT
# =============================================================================

def _import_hyperplanetree():
    try:
        from systems2atoms.hyperplanetree import HyperplaneTreeRegressor
        return HyperplaneTreeRegressor
    except Exception:
        from hyperplanetree import HyperplaneTreeRegressor
        return HyperplaneTreeRegressor

def _import_plot_surrogate_2d():
    tried = []

    candidates = [
        ("systems2atoms.hyperplanetree", "plot_surrogate_2d"),
        ("hyperplanetree", "plot_surrogate_2d"),
    ]

    for mod_name, attr in candidates:
        try:
            module = __import__(mod_name, fromlist=[attr])
            return getattr(module, attr)
        except Exception as e:
            tried.append(f"{mod_name}.{attr}: {e}")

    raise ImportError(
        "Could not import plot_surrogate_2d.\n"
        "Tried:\n- " + "\n- ".join(tried)
    )
# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class UnitSurrogateBundle:
    unit_name: str
    feature_names: List[str]
    output_names: List[str]
    models: Dict[str, object]
    input_bounds: Dict[str, Tuple[float, float]]
    output_bounds: Dict[str, Tuple[float, float]] | None = None
    evaluation_summary: Dict[str, object] | None = None

    def predict(self, x: Dict[str, float]) -> Dict[str, float]:
        x_vec = np.array([[float(x[name]) for name in self.feature_names]], dtype=np.float32)
        x_t = torch.tensor(x_vec, dtype=torch.float32)

        out: Dict[str, float] = {}
        for y_name, model in self.models.items():
            y_pred = model.predict(x_t).detach().cpu().numpy().reshape(-1)[0]
            out[y_name] = float(y_pred)
        return out


# =============================================================================
# GENERIC HELPERS
# =============================================================================
if True: # SMALL HELPERS

    def _clip(a, lo, hi):
        return np.minimum(np.maximum(a, lo), hi)

    def _rng(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    def _sample_uniform(rng: np.random.Generator, low: float, high: float, n: int) -> np.ndarray:
        return rng.uniform(low, high, size=n)

    def _capacity_scale(capacity: np.ndarray, ref: float, exponent: float = 0.7) -> np.ndarray:
        """Monotonic scale-efficiency term in [0,1)."""
        return 1.0 - np.exp(-np.power(np.maximum(capacity, 1e-6) / ref, exponent))

    def _load_or_raise(path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing trained surrogate file: {path}\n"
                f"Run train_and_save_all_surrogates(...) first."
            )
        return joblib.load(path)

def _save_bundle(bundle: UnitSurrogateBundle, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / f"{bundle.unit_name}.joblib")

    # Optional JSON export for each output tree
    json_dir = model_dir / f"{bundle.unit_name}_json"
    json_dir.mkdir(parents=True, exist_ok=True)
    for y_name, model in bundle.models.items():
        try:
            model.write_to_json(str(json_dir / f"{y_name}.json"))
        except Exception:
            pass


def _load_bundle(unit_name: str, model_dir: Path) -> UnitSurrogateBundle:
    return _load_or_raise(model_dir / f"{unit_name}.joblib")


def _get_bundle(unit_name: str, model_dir: Path = DEFAULT_MODEL_DIR) -> UnitSurrogateBundle:
    key = f"{model_dir.resolve()}::{unit_name}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _load_bundle(unit_name, model_dir)
    return _MODEL_CACHE[key]


def _parse_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text == "":
        return default
    return text in {"1", "true", "yes", "y", "t"}


def _prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": float(
            np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0
        ),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def _render_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    output_name: str,
    metrics: Dict[str, float],
) -> Path:
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(y_true, y_pred, s=12, alpha=0.45, color="#2563eb", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{output_name}: parity plot")
    ax.text(
        0.03,
        0.97,
        (
            f"MAE  = {metrics['mae']:.4g}\n"
            f"RMSE = {metrics['rmse']:.4g}\n"
            f"MAPE = {metrics['mape_pct']:.2f}%\n"
            f"R2   = {metrics['r2']:.4f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _resolve_ht_hyperparameter_candidates(n_samples: int) -> List[Dict[str, object]]:
    default_min_leaf = max(0.01, 1.0 / max(n_samples, 1))
    default_min_split = max(2, int(math.ceil(0.06 * n_samples)))

    if not HT_ENABLE_HYPERPARAMETER_EXPLORATION:
        return [{
            **HT_DEFAULT_HYPERPARAMETERS,
            "min_samples_leaf": default_min_leaf,
            "min_samples_split": default_min_split,
        }]

    candidates: List[Dict[str, object]] = []
    seen: set[tuple[tuple[str, object], ...]] = set()
    for min_samples_leaf in HT_MIN_SAMPLES_LEAF_SEARCH:
        for max_weight in HT_MAX_WEIGHT_SEARCH:
            candidate = {
                **HT_DEFAULT_HYPERPARAMETERS,
                "min_samples_leaf": float(min_samples_leaf),
                "min_samples_split": default_min_split,
                "max_weight": max_weight,
            }
            key = tuple(sorted(candidate.items()))
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def _torch_frame(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32)


def _predict_metric_summary(model, X_df: pd.DataFrame, y_df: pd.DataFrame) -> Dict[str, float]:
    y_true = y_df.to_numpy(dtype=float).reshape(-1)
    y_pred = _to_numpy_array(model.predict(_torch_frame(X_df))).reshape(-1)
    return _prediction_metrics(y_true, y_pred)


def _build_gallery_summary_record(
    *,
    gallery_unit_name: str,
    output_name: str,
    x_name: str,
    y_name: str,
    feature_names: List[str],
    metrics: Dict[str, float],
    fixed_inputs: Dict[str, float],
    n_leaves: int,
    surface_plot_path: Path,
    parity_plot_path: Path,
    surface_data: Dict[str, object] | None,
) -> Dict[str, object]:
    return {
        "unit": gallery_unit_name,
        "output": output_name,
        "x_feature": x_name,
        "y_feature": y_name,
        "feature_names": list(feature_names),
        "metrics": metrics,
        "training_metrics": {
            "model_kind": "hyperplane_tree",
            "n_leaves": int(n_leaves),
            "fixed_inputs": fixed_inputs,
        },
        "network_graph_png": "",
        "surface_plot_png": str(surface_plot_path.resolve()),
        "parity_plot_png": str(parity_plot_path.resolve()),
        "training_loss_png": "",
        "surface_data": surface_data,
    }


def reactor_capacity_from_geometry(Kl, Kd=1.0):
    """Convert reactor geometry to a capacity-like feature."""
    return (
        math.pi
        * np.maximum(np.asarray(Kd, dtype=float), 1e-6) ** 2
        * np.asarray(Kl, dtype=float)
        / 4.0
    )


def _ammoniaF_model_feature_names(raw_feature_names: List[str]) -> List[str]:
    excluded = {"Kl", "Kd", "K_N", "hydrogenation_capacity_kgph"}
    return [name for name in raw_feature_names if name not in excluded]


def _ureaF_model_feature_names(raw_feature_names: List[str]) -> List[str]:
    excluded = {"Kl", "Kd", "K_U", "urea_capacity_kgph"}
    return [name for name in raw_feature_names if name not in excluded]


def _filter_to_fixed_geometry(
    df: pd.DataFrame,
    fixed_geometry: Dict[str, float],
    label: str,
) -> pd.DataFrame:
    working_df = df.copy()
    applied_filters: dict[str, float] = {}

    for column_name, target_value in fixed_geometry.items():
        if column_name not in working_df.columns:
            continue

        values = pd.to_numeric(working_df[column_name], errors="coerce").to_numpy(dtype=float)
        tolerance = max(1e-8, 1e-6 * max(abs(float(target_value)), 1.0))
        mask = np.isfinite(values) & np.isclose(
            values,
            float(target_value),
            atol=tolerance,
            rtol=0.0,
        )
        if not np.any(mask):
            raise ValueError(
                f"{label} data does not contain any rows with {column_name}={target_value}."
            )
        working_df = working_df.loc[mask].copy()
        applied_filters[column_name] = float(target_value)

    if applied_filters:
        print(
            f"Filtering {label} training data to fixed geometry {applied_filters}; "
            f"{len(working_df)} rows remain."
        )

    return working_df


def _filter_to_operating_window(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    label: str,
) -> pd.DataFrame:
    working_df = df.copy()
    applied_filters: dict[str, tuple[float, float]] = {}

    for column_name, (lower, upper) in bounds.items():
        if column_name not in working_df.columns:
            raise ValueError(
                f"{label} data is missing required input column `{column_name}` "
                "for operating-window filtering."
            )

        values = pd.to_numeric(working_df[column_name], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values) & (values >= float(lower)) & (values <= float(upper))
        if not np.any(mask):
            raise ValueError(
                f"{label} data does not contain any rows with "
                f"{column_name} between {lower} and {upper}."
            )
        working_df = working_df.loc[mask].copy()
        applied_filters[column_name] = (float(lower), float(upper))

    if applied_filters:
        print(
            f"Filtering {label} training data to operating window {applied_filters}; "
            f"{len(working_df)} rows remain."
        )

    return working_df


def _infer_energy_target(
    df: pd.DataFrame,
    duty_columns: List[str],
    total_abs_column: str,
) -> pd.Series:
    # The training CSV duty columns are expected on an MMkcal/hr basis here.
    # The surrogate target electric_kwhph is therefore an electricity-equivalent
    # kWh/h value, which is numerically the same as kW in the Pyomo power balance.
    if "electric_kwhph" in df.columns:
        return pd.to_numeric(df["electric_kwhph"], errors="coerce")

    if total_abs_column in df.columns:
        total_abs_duty = pd.to_numeric(df[total_abs_column], errors="coerce")
        return total_abs_duty * MMKCAL_PER_HR_TO_KWHPH

    available_duty_columns = [col for col in duty_columns if col in df.columns]
    if not available_duty_columns:
        raise ValueError(
            "Could not infer electric_kwhph because the dataset contains neither an "
            "electric_kwhph column nor the duty columns needed to build one."
        )

    total_abs_duty = (
        df[available_duty_columns]
        .apply(pd.to_numeric, errors="coerce")
        .abs()
        .sum(axis=1)
    )
    return total_abs_duty * MMKCAL_PER_HR_TO_KWHPH


def _ensure_nitrate_component_targets(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    if not {"ammonia_kgph", "water_kgph"}.issubset(working_df.columns):
        if {"hydrous_ammonia_kgph", "nh3_wtfrac_out"}.issubset(working_df.columns):
            total_flow = pd.to_numeric(working_df["hydrous_ammonia_kgph"], errors="coerce")
            nh3_wtfrac = pd.to_numeric(working_df["nh3_wtfrac_out"], errors="coerce")
        elif {"Mt", "Wnh3"}.issubset(working_df.columns):
            total_flow = pd.to_numeric(working_df["Mt"], errors="coerce")
            nh3_wtfrac = pd.to_numeric(working_df["Wnh3"], errors="coerce")
        else:
            raise ValueError(
                "Could not infer ammonia_kgph and water_kgph. Expected either the direct "
                "component-flow targets or one of the legacy total-flow/composition pairs."
            )

        nh3_wtfrac = pd.Series(_clip(nh3_wtfrac.to_numpy(dtype=float), 0.0, 1.0), index=working_df.index)
        working_df["ammonia_kgph"] = total_flow * nh3_wtfrac
        working_df["water_kgph"] = total_flow * (1.0 - nh3_wtfrac)

    if "electric_kwhph" not in working_df.columns:
        working_df["electric_kwhph"] = _infer_energy_target(
            working_df,
            duty_columns=AMMONIAF_DUTY_COLUMNS,
            total_abs_column=AMMONIAF_TOTAL_ABS_DUTY_COLUMN,
        )

    for col in AMMONIA_COMPONENT_OUTPUT_COLUMNS:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    working_df["ammonia_kgph"] = np.maximum(0.0, working_df["ammonia_kgph"].to_numpy(dtype=float))
    working_df["water_kgph"] = np.maximum(0.0, working_df["water_kgph"].to_numpy(dtype=float))
    working_df["electric_kwhph"] = np.maximum(0.0, working_df["electric_kwhph"].to_numpy(dtype=float))

    return working_df


def _ensure_urea_direct_targets(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    if "product_urea_wtfrac" not in working_df.columns:
        if "Wurea" in working_df.columns:
            working_df["product_urea_wtfrac"] = pd.to_numeric(working_df["Wurea"], errors="coerce")
        else:
            raise ValueError(
                "Could not infer product_urea_wtfrac. Expected either product_urea_wtfrac "
                "or the legacy Wurea column."
            )

    if "pure_urea_kgph" not in working_df.columns:
        if "urea_kgph" in working_df.columns:
            working_df["pure_urea_kgph"] = pd.to_numeric(working_df["urea_kgph"], errors="coerce")
        elif "Ft_UREA-OUT" in working_df.columns:
            total_product_flow = pd.to_numeric(working_df["Ft_UREA-OUT"], errors="coerce")
            urea_wtfrac = pd.to_numeric(working_df["product_urea_wtfrac"], errors="coerce")
            urea_wtfrac = pd.Series(_clip(urea_wtfrac.to_numpy(dtype=float), 0.0, 1.0), index=working_df.index)
            working_df["pure_urea_kgph"] = total_product_flow * urea_wtfrac
        else:
            raise ValueError(
                "Could not infer pure_urea_kgph. Expected either pure_urea_kgph, urea_kgph, "
                "or the legacy Ft_UREA-OUT/Wurea pair."
            )

    if "electric_kwhph" not in working_df.columns:
        working_df["electric_kwhph"] = _infer_energy_target(
            working_df,
            duty_columns=UREAF_HEAT_DUTY_COLUMNS,
            total_abs_column=UREAF_TOTAL_ABS_DUTY_COLUMN,
        )

    for col in UREA_DIRECT_OUTPUT_COLUMNS:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    working_df["pure_urea_kgph"] = np.maximum(0.0, working_df["pure_urea_kgph"].to_numpy(dtype=float))
    working_df["product_urea_wtfrac"] = _clip(
        working_df["product_urea_wtfrac"].to_numpy(dtype=float),
        0.0,
        1.0,
    )
    working_df["electric_kwhph"] = np.maximum(0.0, working_df["electric_kwhph"].to_numpy(dtype=float))

    return working_df


def _load_ammoniaf_column_sets(
    case_grid_csv: Path | str | None = None,
    inputs_csv: Path | str | None = AMMONIAF_INPUTS_CSV,
) -> Tuple[List[str], List[str]]:
    input_like_columns: List[str] = []

    feature_names: List[str] = []
    if inputs_csv is not None and Path(inputs_csv).exists():
        inputs_df = pd.read_csv(inputs_csv)
        inputs_df.columns = [str(col).strip().lower() for col in inputs_df.columns]
        if "name" in inputs_df.columns:
            input_like_columns = [
                str(name).strip()
                for name in inputs_df["name"]
                if str(name).strip()
            ]
            if "active" in inputs_df.columns:
                active_mask = inputs_df["active"].apply(lambda value: _parse_bool(value, default=True))
            else:
                active_mask = pd.Series(True, index=inputs_df.index)
            feature_names = [
                str(name).strip()
                for name in inputs_df.loc[active_mask, "name"]
                if str(name).strip()
            ]

    if not feature_names:
        preferred = list(AMMONIAF_MODEL_INPUT_COLUMNS)
        if case_grid_csv is not None and Path(case_grid_csv).exists():
            case_grid_df = pd.read_csv(case_grid_csv, nrows=1)
            input_like_columns = [col for col in case_grid_df.columns if col != "case_id"]
            feature_names = [col for col in preferred if col in input_like_columns]
            if not feature_names:
                varying_columns = [
                    col for col in input_like_columns
                    if not str(col).endswith("_Flow")
                ]
                feature_names = varying_columns or input_like_columns
        else:
            feature_names = preferred

    if not input_like_columns:
        input_like_columns = list(feature_names)

    return input_like_columns, feature_names


def load_ammoniaF_training_data(
    results_csv: Path | str = AMMONIAF_RESULTS_CSV,
    case_grid_csv: Path | str | None = None,
    inputs_csv: Path | str | None = AMMONIAF_INPUTS_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_csv = Path(results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing ammoniaF results CSV: {results_csv}")

    input_like_columns, raw_feature_names = _load_ammoniaf_column_sets(
        case_grid_csv=case_grid_csv,
        inputs_csv=inputs_csv,
    )

    results_raw = pd.read_csv(results_csv)
    missing_input_columns = [
        col for col in raw_feature_names
        if col not in results_raw.columns
    ]
    if missing_input_columns:
        print(
            "Ignoring active ammoniaF input columns not present in the results CSV: "
            f"{missing_input_columns}"
        )
        raw_feature_names = [
            col for col in raw_feature_names
            if col in results_raw.columns
        ]

    if not raw_feature_names:
        raise ValueError("No ammoniaF input feature columns were found in the results CSV.")

    working_df = results_raw.copy()

    if "run_ok" in working_df.columns:
        run_ok = pd.to_numeric(working_df["run_ok"], errors="coerce").fillna(0)
        working_df = working_df.loc[run_ok == 1].copy()

    numeric_columns = [
        col
        for col in working_df.columns
        if col not in AMMONIAF_RESULT_META_COLUMNS
    ]
    for col in numeric_columns:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    working_df = _filter_to_fixed_geometry(
        working_df,
        AMMONIAF_FIXED_GEOMETRY,
        label="ammoniaF",
    )
    working_df = _filter_to_operating_window(
        working_df,
        AMMONIAF_OPERATING_INPUT_BOUNDS,
        label="ammoniaF",
    )

    if all(col in working_df.columns for col in AMMONIAF_DUTY_COLUMNS):
        working_df[AMMONIAF_TOTAL_ABS_DUTY_COLUMN] = (
            working_df[AMMONIAF_DUTY_COLUMNS].abs().sum(axis=1)
        )

    working_df = _ensure_nitrate_component_targets(working_df)

    feature_names = _ammoniaF_model_feature_names(raw_feature_names)
    required_columns = feature_names + AMMONIA_COMPONENT_OUTPUT_COLUMNS
    working_df = working_df.dropna(subset=required_columns).reset_index(drop=True)

    if working_df.empty:
        raise ValueError("No successful ammoniaF rows remained after filtering and target conversion.")

    X = working_df[feature_names].copy()
    Y = working_df[AMMONIA_COMPONENT_OUTPUT_COLUMNS].copy()
    return X, Y, working_df


def train_and_save_ammoniaF_surrogate(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    results_csv: Path | str = AMMONIAF_RESULTS_CSV,
    case_grid_csv: Path | str | None = None,
    inputs_csv: Path | str | None = AMMONIAF_INPUTS_CSV,
    seed: int = HT_RANDOM_SEED,
    test_fraction: float = HT_TEST_FRACTION,
) -> UnitSurrogateBundle:
    X, Y, _ = load_ammoniaF_training_data(
        results_csv=results_csv,
        case_grid_csv=case_grid_csv,
        inputs_csv=inputs_csv,
    )
    bundle = _fit_bundle("ammoniaF_unit", X, Y, seed=seed, test_fraction=test_fraction)
    _save_bundle(bundle, Path(model_dir))
    _MODEL_CACHE.pop(f"{Path(model_dir).resolve()}::ammoniaF_unit", None)
    _print_unit_validation_report("ammoniaF_unit", bundle, X, Y)
    return bundle


def ammoniaF_unit(
    Ft: float,
    Fh2: float,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
) -> Dict[str, float]:
    bundle = _get_bundle("ammoniaF_unit", Path(model_dir))
    return bundle.predict({
        "Ft": Ft,
        "Fh2": Fh2,
    })


def load_ureaF_training_data(
    results_csv: Path | str = UREAF_RESULTS_CSV,
    include_all_heat_duties: bool = UREAF_TRAIN_ALL_HEAT_DUTIES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_csv = Path(results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing ureaF results CSV: {results_csv}")

    working_df = pd.read_csv(results_csv)

    if "run_ok" in working_df.columns:
        run_ok = pd.to_numeric(working_df["run_ok"], errors="coerce").fillna(0)
        working_df = working_df.loc[run_ok == 1].copy()

    missing_inputs = [col for col in UREAF_RAW_INPUT_COLUMNS if col not in working_df.columns]
    if missing_inputs:
        raise ValueError(f"Missing ureaF input columns in results CSV: {missing_inputs}")

    heat_duty_columns = [
        col for col in UREAF_HEAT_DUTY_COLUMNS
        if col in working_df.columns and col not in UREAF_EXCLUDED_HEAT_DUTY_COLUMNS
    ]
    missing_heat_duties = [
        col for col in UREAF_HEAT_DUTY_COLUMNS
        if col not in working_df.columns
    ]
    if missing_heat_duties:
        raise ValueError(f"Missing ureaF heat-duty columns in results CSV: {missing_heat_duties}")

    numeric_columns = list(
        dict.fromkeys(
            UREAF_RAW_INPUT_COLUMNS
            + [col for col in UREAF_FIXED_GEOMETRY if col in working_df.columns]
            + heat_duty_columns
            + ["Ft_UREA-OUT", "Wurea"]
        )
    )
    for col in numeric_columns:
        if col in working_df.columns:
            working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    working_df[UREAF_TOTAL_ABS_DUTY_COLUMN] = (
        working_df[heat_duty_columns].abs().sum(axis=1)
    )

    working_df = _filter_to_fixed_geometry(
        working_df,
        UREAF_FIXED_GEOMETRY,
        label="ureaF",
    )
    working_df = _filter_to_operating_window(
        working_df,
        UREAF_OPERATING_INPUT_BOUNDS,
        label="ureaF",
    )

    working_df = _ensure_urea_direct_targets(working_df)

    feature_names = list(UREAF_MODEL_INPUT_COLUMNS)
    required_columns = feature_names + UREA_DIRECT_OUTPUT_COLUMNS
    working_df = working_df.dropna(subset=required_columns).reset_index(drop=True)

    if working_df.empty:
        raise ValueError("No successful ureaF rows remained after filtering and numeric coercion.")

    X = working_df[feature_names].copy()
    Y = working_df[UREA_DIRECT_OUTPUT_COLUMNS].copy()
    return X, Y, working_df


def train_and_save_ureaF_surrogate(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    results_csv: Path | str = UREAF_RESULTS_CSV,
    include_all_heat_duties: bool = UREAF_TRAIN_ALL_HEAT_DUTIES,
    seed: int = HT_RANDOM_SEED,
    test_fraction: float = HT_TEST_FRACTION,
) -> UnitSurrogateBundle:
    X, Y, _ = load_ureaF_training_data(
        results_csv=results_csv,
        include_all_heat_duties=include_all_heat_duties,
    )
    bundle = _fit_bundle("ureaF_unit", X, Y, seed=seed, test_fraction=test_fraction)
    _save_bundle(bundle, Path(model_dir))
    _MODEL_CACHE.pop(f"{Path(model_dir).resolve()}::ureaF_unit", None)
    _print_unit_validation_report("ureaF_unit", bundle, X, Y)
    return bundle


def ureaF_unit(
    Fnh3: float,
    Fco2: float,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
) -> Dict[str, float]:
    bundle = _get_bundle("ureaF_unit", Path(model_dir))
    return bundle.predict({
        "Fnh3": Fnh3,
        "Fco2": Fco2,
    })


def _fit_bundle(
    unit_name: str,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    seed: int = HT_RANDOM_SEED,
    test_fraction: float = HT_TEST_FRACTION,
) -> UnitSurrogateBundle:
    HyperplaneTreeRegressor = _import_hyperplanetree()
    if len(X) < 2:
        raise ValueError(f"At least two rows are required to fit {unit_name}.")

    effective_test_fraction = float(test_fraction)
    if len(X) < 5:
        effective_test_fraction = min(max(effective_test_fraction, 0.5), (len(X) - 1) / len(X))
    elif not (0.0 < effective_test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction!r}")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=effective_test_fraction,
        random_state=seed,
        shuffle=True,
    )

    X_train_t = _torch_frame(X_train)

    models = {}
    evaluation_outputs: Dict[str, Dict[str, object]] = {}
    n_train_samples = len(X_train)
    candidate_configs = _resolve_ht_hyperparameter_candidates(n_train_samples)

    for y_name in Y.columns:
        y_train_t = _torch_frame(Y_train[[y_name]])
        y_test_df = Y_test[[y_name]]
        y_train_df = Y_train[[y_name]]

        best_model = None
        best_summary = None
        best_score = float("inf")
        exploration_rows: list[Dict[str, object]] = []

        for candidate_idx, candidate in enumerate(candidate_configs, start=1):
            model = HyperplaneTreeRegressor(**candidate)
            model.fit(X_train_t, y_train_t)

            train_metrics = _predict_metric_summary(model, X_train, y_train_df)
            test_metrics = _predict_metric_summary(model, X_test, y_test_df)
            n_leaves = len(getattr(model, "_leaves", []))
            candidate_summary = {
                "candidate_index": candidate_idx,
                "params": dict(candidate),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "n_leaves": int(n_leaves),
            }
            exploration_rows.append(candidate_summary)

            if test_metrics["mae"] < best_score:
                best_score = float(test_metrics["mae"])
                best_model = model
                best_summary = candidate_summary

        if best_model is None or best_summary is None:
            raise RuntimeError(f"Could not fit any HyperplaneTree candidate for {unit_name}.{y_name}.")

        models[y_name] = best_model
        evaluation_outputs[y_name] = {
            "selected_params": best_summary["params"],
            "n_leaves": best_summary["n_leaves"],
            "train_metrics": best_summary["train_metrics"],
            "test_metrics": best_summary["test_metrics"],
            "exploration": exploration_rows if HT_ENABLE_HYPERPARAMETER_EXPLORATION else [],
        }

    requested_bounds = UNIT_INPUT_BOUND_OVERRIDES.get(unit_name, {})
    input_bounds = {
        col: (
            tuple(float(value) for value in requested_bounds[col])
            if col in requested_bounds
            else (float(X[col].min()), float(X[col].max()))
        )
        for col in X.columns
    }
    output_bounds = {
        col: (float(Y[col].min()), float(Y[col].max()))
        for col in Y.columns
    }

    return UnitSurrogateBundle(
        unit_name=unit_name,
        feature_names=list(X.columns),
        output_names=list(Y.columns),
        models=models,
        input_bounds=input_bounds,
        output_bounds=output_bounds,
        evaluation_summary={
            "seed": int(seed),
            "test_fraction": float(effective_test_fraction),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "hyperparameter_exploration_enabled": bool(HT_ENABLE_HYPERPARAMETER_EXPLORATION),
            "outputs": evaluation_outputs,
        },
    )


def _print_unit_validation_report(
    unit_name: str,
    bundle: UnitSurrogateBundle,
    X: pd.DataFrame,
    Y: pd.DataFrame,
) -> None:
    print(f"\nValidation report for {unit_name}:")
    print(f"  inputs:  {list(X.columns)}")
    print(f"  outputs: {list(Y.columns)}")
    evaluation_summary = getattr(bundle, "evaluation_summary", {}) or {}
    if evaluation_summary:
        print(
            "  split:   "
            f"{evaluation_summary.get('train_rows', 'n/a')} train / "
            f"{evaluation_summary.get('test_rows', 'n/a')} test "
            f"(seed={evaluation_summary.get('seed', 'n/a')}, "
            f"test_fraction={evaluation_summary.get('test_fraction', 'n/a')})"
        )
        print(
            "  search:  "
            f"{'enabled' if evaluation_summary.get('hyperparameter_exploration_enabled') else 'disabled'}"
        )
    print("  output ranges:")
    for output_name in Y.columns:
        if getattr(bundle, "output_bounds", None) and output_name in bundle.output_bounds:
            lo, hi = bundle.output_bounds[output_name]
        else:
            lo = float(Y[output_name].min())
            hi = float(Y[output_name].max())
        print(f"    {output_name}: [{lo:.6g}, {hi:.6g}]")
        output_eval = evaluation_summary.get("outputs", {}).get(output_name, {})
        if output_eval:
            train_metrics = output_eval.get("train_metrics", {})
            test_metrics = output_eval.get("test_metrics", {})
            print(
                "      train metrics: "
                f"MAE={train_metrics.get('mae', float('nan')):.6g}, "
                f"RMSE={train_metrics.get('rmse', float('nan')):.6g}, "
                f"R2={train_metrics.get('r2', float('nan')):.6g}"
            )
            print(
                "      test metrics:  "
                f"MAE={test_metrics.get('mae', float('nan')):.6g}, "
                f"RMSE={test_metrics.get('rmse', float('nan')):.6g}, "
                f"R2={test_metrics.get('r2', float('nan')):.6g}"
            )
            print(
                f"      selected HT params: {output_eval.get('selected_params', {})} | "
                f"leaves={output_eval.get('n_leaves', 'n/a')}"
            )

    if set(AMMONIA_COMPONENT_OUTPUT_COLUMNS).issubset(Y.columns):
        print(
            "  confirmation: ammonia_kgph and water_kgph are direct component flows; "
            "the optimizer no longer needs a total-flow x composition reconstruction."
        )
    if set(UREA_DIRECT_OUTPUT_COLUMNS).issubset(Y.columns):
        print(
            "  confirmation: pure_urea_kgph is a direct pure-urea mass-flow target, "
            "while product_urea_wtfrac remains a separate quality output."
        )


# =============================================================================
# TRAIN / SAVE / LOAD ALL
# =============================================================================

def train_and_save_all_surrogates(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    n_samples: int = 2000,
    seed: int = HT_RANDOM_SEED,
    data_dir: Path | str | None = None,
    test_fraction: float = HT_TEST_FRACTION,
) -> Dict[str, UnitSurrogateBundle]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is not None:
        data_dir = Path(data_dir)
        X1 = pd.read_csv(data_dir / "nitrate_unit_X.csv")
        Y1 = pd.read_csv(data_dir / "nitrate_unit_Y.csv")
        X3 = pd.read_csv(data_dir / "urea_unit_X.csv")
        Y3 = pd.read_csv(data_dir / "urea_unit_Y.csv")
        Y1 = _ensure_nitrate_component_targets(Y1)[AMMONIA_COMPONENT_OUTPUT_COLUMNS]
        if {"Fnh3", "Fco2"}.issubset(X3.columns):
            X3 = X3[["Fnh3", "Fco2"]]
        else:
            legacy_map = {
                "nh3_feed_kgph": "Fnh3",
                "co2_feed_kgph": "Fco2",
            }
            available_legacy = [col for col in legacy_map if col in X3.columns]
            if len(available_legacy) != 2:
                raise ValueError(
                    "Could not identify the fixed-geometry urea surrogate inputs in the "
                    f"provided data_dir frame. Columns found: {list(X3.columns)}"
                )
            X3 = X3[available_legacy].rename(columns=legacy_map)[["Fnh3", "Fco2"]]
        Y3 = _ensure_urea_direct_targets(Y3)[UREA_DIRECT_OUTPUT_COLUMNS]
        print(f"Loaded training data from {data_dir}")
    

    # 1) Nitrate -> direct ammonia/water flows
    nitrate_bundle = _fit_bundle("nitrate_unit", X1, Y1, seed=seed, test_fraction=test_fraction)
    _save_bundle(nitrate_bundle, model_dir)
    _print_unit_validation_report("nitrate_unit", nitrate_bundle, X1, Y1)

    # 2) Urea
    urea_bundle = _fit_bundle("urea_unit", X3, Y3, seed=seed, test_fraction=test_fraction)
    _save_bundle(urea_bundle, model_dir)
    _print_unit_validation_report("urea_unit", urea_bundle, X3, Y3)

    return {
        "nitrate_unit": nitrate_bundle,
        "urea_unit": urea_bundle,
    }


# =============================================================================
# STABLE IMPORTABLE FUNCTIONS FOR THE OPTIMIZER
# =============================================================================


def _plot_1d_sweeps(
    unit_name: str,
    bundle: UnitSurrogateBundle,
    baseline: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    direct_fn: Callable[..., Dict[str, float]],
    surrogate_fn: Callable[..., Dict[str, float]],
    model_dir: Path,
    n_points: int = 200,
) -> None:
    out_dir = model_dir / "plots" / unit_name / "response_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    for x_name in bundle.feature_names:
        lo, hi = bounds[x_name]
        xs = np.linspace(lo, hi, n_points)

        # collect curves for all outputs
        true_curves = {y: [] for y in bundle.output_names}
        pred_curves = {y: [] for y in bundle.output_names}

        for x_val in xs:
            x_dict = dict(baseline)
            x_dict[x_name] = float(x_val)

            y_true = direct_fn(**x_dict)
            y_pred = surrogate_fn(model_dir=model_dir, **x_dict)

            for y_name in bundle.output_names:
                true_curves[y_name].append(float(y_true[y_name]))
                pred_curves[y_name].append(float(y_pred[y_name]))

        for y_name in bundle.output_names:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(xs, true_curves[y_name], label="Assumed nonlinear profile")
            ax.plot(xs, pred_curves[y_name], linestyle="--", label="HyperplaneTree fit")
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(f"{unit_name}: {y_name} vs {x_name}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{y_name}_vs_{x_name}.png", dpi=200)
            plt.close(fig)


def _to_numpy_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _make_labeled_surrogate_plots(
    model,
    features,
    unit_name: str,
    x_name: str,
    y_name: str,
    z_name: str,
    out_prefix: Path | None = None,
    cmap: str = "rainbow",
    save: bool = True,
    grid_points: int = 80,
    plot_bounds: Dict[str, Tuple[float, float]] | None = None,
    model_features=None,
    grid_feature_builder=None,
    point_values=None,
    fixed_inputs: Dict[str, float] | None = None,
    point_label: str = "Model predictions at training points",
    show_3d_plot: bool = False,
):
    if model_features is None:
        model_features = features

    leaf = _to_numpy_array(model.apply(model_features)).reshape(-1)
    y_pred = _to_numpy_array(model.predict(model_features)).flatten()
    if point_values is None:
        point_values_np = y_pred
    else:
        point_values_np = np.asarray(point_values, dtype=float).reshape(-1)
    feat_np = features.detach().cpu().numpy()

    saved_files = []
    cmap_obj = plt.get_cmap(cmap)

    if plot_bounds and x_name in plot_bounds:
        x_min, x_max = (float(plot_bounds[x_name][0]), float(plot_bounds[x_name][1]))
    else:
        x_min, x_max = float(feat_np[:, 0].min()), float(feat_np[:, 0].max())
    if plot_bounds and y_name in plot_bounds:
        y_min, y_max = (float(plot_bounds[y_name][0]), float(plot_bounds[y_name][1]))
    else:
        y_min, y_max = float(feat_np[:, 1].min()), float(feat_np[:, 1].max())
    x_grid = np.linspace(x_min, x_max, grid_points)
    y_grid = np.linspace(y_min, y_max, grid_points)
    xx, yy = np.meshgrid(x_grid, y_grid)

    if grid_feature_builder is None:
        grid_feature_array = np.column_stack([xx.ravel(), yy.ravel()])
    else:
        grid_feature_array = grid_feature_builder(xx.ravel(), yy.ravel())

    grid_features = torch.tensor(
        grid_feature_array.astype(np.float32),
        dtype=torch.float32,
    )
    zz = _to_numpy_array(model.predict(grid_features)).reshape(xx.shape)
    leaf_grid = _to_numpy_array(model.apply(grid_features)).reshape(xx.shape)

    unique_leaves = np.unique(leaf_grid)
    leaf_to_index = {leaf_id: idx for idx, leaf_id in enumerate(unique_leaves)}
    leaf_index_grid = np.vectorize(leaf_to_index.get)(leaf_grid)
    facecolors = cmap_obj(
        leaf_index_grid / max(len(unique_leaves) - 1, 1)
    )

    # 2D leaf partition plot across the whole feature rectangle
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    mesh = ax1.pcolormesh(
        xx,
        yy,
        leaf_index_grid,
        shading="auto",
        cmap=cmap_obj,
        alpha=0.85,
    )
    ax1.contour(
        xx,
        yy,
        leaf_index_grid,
        levels=np.arange(len(unique_leaves) + 1) - 0.5,
        colors="black",
        linewidths=0.45,
        alpha=0.45,
    )
    ax1.scatter(
        feat_np[:, 0],
        feat_np[:, 1],
        c="white",
        edgecolors="black",
        linewidths=0.25,
        s=10,
        alpha=0.8,
    )
    ax1.set_xlabel(x_name, fontsize=12)
    ax1.set_ylabel(y_name, fontsize=12)
    ax1.set_title(f"{unit_name}: {z_name} (leaf partition)")
    cbar = fig1.colorbar(mesh, ax=ax1, pad=0.02)
    cbar.set_label("Leaf section", fontsize=10)
    fig1.tight_layout()
    if save and out_prefix is not None:
        f1 = Path(f"{out_prefix}_view1.png")
        fig1.savefig(f1, dpi=300, bbox_inches="tight")
        saved_files.append(f1)
    plt.close(fig1)

    # 3D piecewise-planar surrogate over the whole domain
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(
        xx,
        yy,
        zz,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.88,
    )
    ax2.plot_wireframe(
        xx,
        yy,
        zz,
        rstride=max(grid_points // 20, 1),
        cstride=max(grid_points // 20, 1),
        color="black",
        linewidth=0.25,
        alpha=0.25,
    )
    ax2.scatter(
        feat_np[:, 0],
        feat_np[:, 1],
        point_values_np,
        marker='o',
        s=8,
        c="black",
        alpha=0.35,
        depthshade=False,
        label=point_label,
    )
    z_floor = float(min(zz.min(), point_values_np.min()))
    ax2.contour(
        xx,
        yy,
        leaf_index_grid,
        zdir="z",
        offset=z_floor,
        levels=np.arange(len(unique_leaves) + 1) - 0.5,
        cmap=cmap_obj,
        linewidths=0.8,
    )
    ax2.set_zlim(z_floor, float(max(zz.max(), point_values_np.max())))
    ax2.set_xlabel(x_name)
    ax2.set_ylabel(y_name)
    ax2.set_zlabel(z_name)
    ax2.set_title(f"{unit_name}: {z_name} (piecewise planes)")
    ax2.view_init(elev=24, azim=-135)
    ax2.legend(loc="upper left", fontsize=8)
    fig2.tight_layout()
    if save and out_prefix is not None:
        f2 = Path(f"{out_prefix}_view2.png")
        fig2.savefig(f2, dpi=300, bbox_inches="tight")
        saved_files.append(f2)
    if show_3d_plot:
        plt.show(block=True)
    else:
        plt.close(fig2)

    surface_data = {
        "x_name": x_name,
        "y_name": y_name,
        "z_name": z_name,
        "x_grid": x_grid.tolist(),
        "y_grid": y_grid.tolist(),
        "z_grid": zz.tolist(),
        "sampled_points": {
            "x": feat_np[:, 0].astype(float).tolist(),
            "y": feat_np[:, 1].astype(float).tolist(),
            "z": point_values_np.astype(float).tolist(),
        },
        "slice_points": {
            "x": feat_np[:, 0].astype(float).tolist(),
            "y": feat_np[:, 1].astype(float).tolist(),
            "z": point_values_np.astype(float).tolist(),
        },
        "other_points": {
            "x": [],
            "y": [],
            "z": [],
        },
        "fixed_inputs": dict(fixed_inputs or {}),
        "show_other_points": False,
        "leaf_index_grid": np.asarray(leaf_index_grid, dtype=float).tolist(),
    }

    return {
        "saved_files": saved_files,
        "surface_data": surface_data,
    }


def save_ammoniaF_tree_plots(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    results_csv: Path | str = AMMONIAF_RESULTS_CSV,
    case_grid_csv: Path | str | None = None,
    inputs_csv: Path | str | None = AMMONIAF_INPUTS_CSV,
    show_3d_plots: bool = SHOW_INTERACTIVE_3D_PLOTS,
    gallery_out_dir: Path | str = DEFAULT_GALLERY_OUT_DIR,
) -> None:
    model_dir = Path(model_dir)
    plot_root = model_dir / "plots_phase" / "ammoniaF_unit"
    plot_root.mkdir(parents=True, exist_ok=True)
    gallery_root = Path(gallery_out_dir) / "ammoniaF_unit_ht"
    gallery_root.mkdir(parents=True, exist_ok=True)

    X, Y, _ = load_ammoniaF_training_data(
        results_csv=results_csv,
        case_grid_csv=case_grid_csv,
        inputs_csv=inputs_csv,
    )

    if len(X.columns) < 2:
        raise ValueError("At least two ammoniaF inputs are required to generate tree plots.")

    try:
        bundle = _get_bundle("ammoniaF_unit", model_dir)
    except Exception as exc:
        print(f"Could not load existing ammoniaF_unit.joblib ({exc}); retraining from CSV.")
        bundle = train_and_save_ammoniaF_surrogate(
            model_dir=model_dir,
            results_csv=results_csv,
            case_grid_csv=case_grid_csv,
            inputs_csv=inputs_csv,
        )

    if list(bundle.feature_names) != list(X.columns):
        print(
            "Existing ammoniaF_unit.joblib uses feature names "
            f"{bundle.feature_names}; retraining with {list(X.columns)}."
        )
        bundle = train_and_save_ammoniaF_surrogate(
            model_dir=model_dir,
            results_csv=results_csv,
            case_grid_csv=case_grid_csv,
            inputs_csv=inputs_csv,
        )
        _MODEL_CACHE.pop(f"{model_dir.resolve()}::ammoniaF_unit", None)
        bundle = _get_bundle("ammoniaF_unit", model_dir)

    missing_features = [name for name in bundle.feature_names if name not in X.columns]
    if missing_features:
        raise ValueError(
            "The trained ammoniaF model expects input columns that are not available "
            f"in the ammoniaF data: {missing_features}"
        )

    full_feature_df = X[bundle.feature_names].copy()
    full_features = torch.tensor(
        full_feature_df.to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    slice_baseline = {
        name: float(full_feature_df[name].median())
        for name in bundle.feature_names
    }

    all_metrics = []
    gallery_summaries: list[Dict[str, object]] = []

    for x_name, y_name in combinations(bundle.feature_names, 2):
        pair_dir = plot_root / f"{x_name}__{y_name}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        plot_features = torch.tensor(
            full_feature_df[[x_name, y_name]].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        fixed_inputs = {
            name: value
            for name, value in slice_baseline.items()
            if name not in {x_name, y_name}
        }

        def build_full_slice_features(x_values, y_values, x_col=x_name, y_col=y_name):
            grid_df = pd.DataFrame({
                name: np.full(len(x_values), value, dtype=float)
                for name, value in slice_baseline.items()
            })
            grid_df[x_col] = x_values
            grid_df[y_col] = y_values
            return grid_df[bundle.feature_names].to_numpy(dtype=np.float32)

        for output_name in bundle.output_names:
            if output_name not in Y.columns:
                continue

            model_4d = bundle.models[output_name]
            y_pred = _to_numpy_array(model_4d.predict(full_features)).reshape(-1)
            y_true = Y[output_name].to_numpy(dtype=float).reshape(-1)

            metrics = _prediction_metrics(y_true, y_pred)
            n_leaves = len(getattr(model_4d, "_leaves", []))

            print(
                f"ammoniaF_unit | {x_name} vs {y_name} | {output_name} | "
                f"Full reduced-model slice | "
                f"Leaves: {n_leaves} | "
                f"MAE: {metrics['mae']:.4f} | "
                f"MAPE: {metrics['mape_pct']:.2f}% | "
                f"Fixed inputs: {fixed_inputs}"
            )

            plot_payload = _make_labeled_surrogate_plots(
                model=model_4d,
                features=plot_features,
                unit_name="ammoniaF_unit",
                x_name=x_name,
                y_name=y_name,
                z_name=output_name,
                out_prefix=pair_dir / output_name,
                cmap="rainbow",
                save=True,
                plot_bounds=AMMONIAF_OPERATING_INPUT_BOUNDS,
                model_features=full_features,
                grid_feature_builder=build_full_slice_features,
                point_values=y_true,
                fixed_inputs=fixed_inputs,
                point_label="Raw training data at Aspen cases",
                show_3d_plot=show_3d_plots,
            )
            saved_plot_paths = plot_payload["saved_files"]
            surface_plot_path = saved_plot_paths[-1] if saved_plot_paths else pair_dir / f"{output_name}_view2.png"

            gallery_pair_dir = gallery_root / output_name / f"{x_name}__{y_name}"
            gallery_pair_dir.mkdir(parents=True, exist_ok=True)
            parity_plot_path = _render_parity_plot(
                y_true=y_true,
                y_pred=y_pred,
                out_path=gallery_pair_dir / "parity_plot.png",
                output_name=output_name,
                metrics=metrics,
            )
            summary = _build_gallery_summary_record(
                gallery_unit_name="ammoniaF_unit_ht",
                output_name=output_name,
                x_name=x_name,
                y_name=y_name,
                feature_names=bundle.feature_names,
                metrics=metrics,
                fixed_inputs=fixed_inputs,
                n_leaves=n_leaves,
                surface_plot_path=surface_plot_path,
                parity_plot_path=parity_plot_path,
                surface_data=plot_payload["surface_data"],
            )
            with (gallery_pair_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            gallery_summaries.append(summary)

            all_metrics.append({
                "unit": "ammoniaF_unit",
                "x_feature": x_name,
                "y_feature": y_name,
                "output": output_name,
                "slice_type": "full_reduced_model",
                "fixed_inputs": json.dumps(fixed_inputs),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
                "mape": float(metrics["mape_pct"]),
                "n_leaves": int(n_leaves),
            })

    pd.DataFrame(all_metrics).to_csv(plot_root / "phase_plot_metrics.csv", index=False)
    with (gallery_root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(gallery_summaries, f, indent=2)


def save_ureaF_tree_plots(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    results_csv: Path | str = UREAF_RESULTS_CSV,
    include_all_heat_duties: bool = UREAF_TRAIN_ALL_HEAT_DUTIES,
    show_3d_plots: bool = SHOW_INTERACTIVE_3D_PLOTS,
    gallery_out_dir: Path | str = DEFAULT_GALLERY_OUT_DIR,
) -> None:
    model_dir = Path(model_dir)
    plot_root = model_dir / "plots_phase" / "ureaF_unit"
    plot_root.mkdir(parents=True, exist_ok=True)
    gallery_root = Path(gallery_out_dir) / "ureaF_unit_ht"
    gallery_root.mkdir(parents=True, exist_ok=True)

    X, Y, _ = load_ureaF_training_data(
        results_csv=results_csv,
        include_all_heat_duties=include_all_heat_duties,
    )

    if len(X.columns) < 2:
        raise ValueError("At least two ureaF inputs are required to generate tree plots.")

    try:
        bundle = _get_bundle("ureaF_unit", model_dir)
    except Exception as exc:
        print(f"Could not load existing ureaF_unit.joblib ({exc}); retraining from CSV.")
        bundle = train_and_save_ureaF_surrogate(
            model_dir=model_dir,
            results_csv=results_csv,
            include_all_heat_duties=include_all_heat_duties,
        )

    if list(bundle.feature_names) != list(X.columns) or list(bundle.output_names) != list(Y.columns):
        print(
            "Existing ureaF_unit.joblib does not match requested feature/output set; "
            f"retraining with features {list(X.columns)} and outputs {list(Y.columns)}."
        )
        bundle = train_and_save_ureaF_surrogate(
            model_dir=model_dir,
            results_csv=results_csv,
            include_all_heat_duties=include_all_heat_duties,
        )
        bundle = _get_bundle("ureaF_unit", model_dir)

    full_feature_df = X[bundle.feature_names].copy()
    full_features = torch.tensor(
        full_feature_df.to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    slice_baseline = {
        name: float(full_feature_df[name].median())
        for name in bundle.feature_names
    }

    all_metrics = []
    gallery_summaries: list[Dict[str, object]] = []

    for x_name, y_name in combinations(bundle.feature_names, 2):
        pair_dir = plot_root / f"{x_name}__{y_name}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        plot_features = torch.tensor(
            full_feature_df[[x_name, y_name]].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        fixed_inputs = {
            name: value
            for name, value in slice_baseline.items()
            if name not in {x_name, y_name}
        }

        def build_full_slice_features(x_values, y_values, x_col=x_name, y_col=y_name):
            grid_df = pd.DataFrame({
                name: np.full(len(x_values), value, dtype=float)
                for name, value in slice_baseline.items()
            })
            grid_df[x_col] = x_values
            grid_df[y_col] = y_values
            return grid_df[bundle.feature_names].to_numpy(dtype=np.float32)

        for output_name in bundle.output_names:
            if output_name not in Y.columns:
                continue

            model = bundle.models[output_name]
            y_pred = _to_numpy_array(model.predict(full_features)).reshape(-1)
            y_true = Y[output_name].to_numpy(dtype=float).reshape(-1)

            metrics = _prediction_metrics(y_true, y_pred)
            n_leaves = len(getattr(model, "_leaves", []))

            print(
                f"ureaF_unit | {x_name} vs {y_name} | {output_name} | "
                f"Full model slice | "
                f"Leaves: {n_leaves} | "
                f"MAE: {metrics['mae']:.6f} | "
                f"MAPE: {metrics['mape_pct']:.2f}% | "
                f"Fixed inputs: {fixed_inputs}"
            )

            plot_payload = _make_labeled_surrogate_plots(
                model=model,
                features=plot_features,
                unit_name="ureaF_unit",
                x_name=x_name,
                y_name=y_name,
                z_name=output_name,
                out_prefix=pair_dir / output_name,
                cmap="rainbow",
                save=True,
                plot_bounds=UREAF_OPERATING_INPUT_BOUNDS,
                model_features=full_features,
                grid_feature_builder=build_full_slice_features,
                point_values=y_true,
                fixed_inputs=fixed_inputs,
                point_label="Raw training data at Aspen cases",
                show_3d_plot=show_3d_plots,
            )
            saved_plot_paths = plot_payload["saved_files"]
            surface_plot_path = saved_plot_paths[-1] if saved_plot_paths else pair_dir / f"{output_name}_view2.png"

            gallery_pair_dir = gallery_root / output_name / f"{x_name}__{y_name}"
            gallery_pair_dir.mkdir(parents=True, exist_ok=True)
            parity_plot_path = _render_parity_plot(
                y_true=y_true,
                y_pred=y_pred,
                out_path=gallery_pair_dir / "parity_plot.png",
                output_name=output_name,
                metrics=metrics,
            )
            summary = _build_gallery_summary_record(
                gallery_unit_name="ureaF_unit_ht",
                output_name=output_name,
                x_name=x_name,
                y_name=y_name,
                feature_names=bundle.feature_names,
                metrics=metrics,
                fixed_inputs=fixed_inputs,
                n_leaves=n_leaves,
                surface_plot_path=surface_plot_path,
                parity_plot_path=parity_plot_path,
                surface_data=plot_payload["surface_data"],
            )
            with (gallery_pair_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            gallery_summaries.append(summary)

            all_metrics.append({
                "unit": "ureaF_unit",
                "x_feature": x_name,
                "y_feature": y_name,
                "output": output_name,
                "slice_type": "full_model",
                "fixed_inputs": json.dumps(fixed_inputs),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
                "mape": float(metrics["mape_pct"]),
                "n_leaves": int(n_leaves),
                "include_all_heat_duties": bool(include_all_heat_duties),
            })

    pd.DataFrame(all_metrics).to_csv(plot_root / "phase_plot_metrics.csv", index=False)
    with (gallery_root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(gallery_summaries, f, indent=2)

# =============================================================================
# EXAMPLE MAIN
# =============================================================================

if __name__ == "__main__":
    model_dir = DEFAULT_MODEL_DIR
    data_dir = DEFAULT_MODEL_DIR / "training_data"
    use_ammoniaf_data = AMMONIAF_RESULTS_CSV.exists()
    use_ureaf_data = UREAF_RESULTS_CSV.exists()

    if use_ammoniaf_data or use_ureaf_data:
        if use_ammoniaf_data:
            print(f"Training ammoniaF surrogate from: {AMMONIAF_RESULTS_CSV.resolve()}")
            ammonia_bundle = train_and_save_ammoniaF_surrogate(
                model_dir=model_dir,
                results_csv=AMMONIAF_RESULTS_CSV,
                case_grid_csv=None,
                inputs_csv=AMMONIAF_INPUTS_CSV if AMMONIAF_INPUTS_CSV.exists() else None,
            )
            print(
                f"Loaded ammoniaF training data with "
                f"{len(ammonia_bundle.feature_names)} inputs and "
                f"{len(ammonia_bundle.output_names)} outputs."
            )
            save_ammoniaF_tree_plots(
                model_dir=model_dir,
                results_csv=AMMONIAF_RESULTS_CSV,
                case_grid_csv=None,
                inputs_csv=AMMONIAF_INPUTS_CSV if AMMONIAF_INPUTS_CSV.exists() else None,
                show_3d_plots=SHOW_INTERACTIVE_3D_PLOTS,
                gallery_out_dir=DEFAULT_GALLERY_OUT_DIR,
            )
            print(f"\nAmmoniaF plots saved under: {model_dir / 'plots_phase' / 'ammoniaF_unit'}")
            print(f"AmmoniaF gallery summary saved under: {DEFAULT_GALLERY_OUT_DIR / 'ammoniaF_unit_ht' / 'run_summary.json'}")

        if use_ureaf_data:
            print(f"Training ureaF surrogate from: {UREAF_RESULTS_CSV.resolve()}")
            ureaF_bundle = train_and_save_ureaF_surrogate(
                model_dir=model_dir,
                results_csv=UREAF_RESULTS_CSV,
                include_all_heat_duties=UREAF_TRAIN_ALL_HEAT_DUTIES,
            )
            print(
                f"Loaded ureaF training data with "
                f"{len(ureaF_bundle.feature_names)} inputs and "
                f"{len(ureaF_bundle.output_names)} outputs."
            )
            save_ureaF_tree_plots(
                model_dir=model_dir,
                results_csv=UREAF_RESULTS_CSV,
                include_all_heat_duties=UREAF_TRAIN_ALL_HEAT_DUTIES,
                show_3d_plots=SHOW_INTERACTIVE_3D_PLOTS,
                gallery_out_dir=DEFAULT_GALLERY_OUT_DIR,
            )
            print(f"\nUreaF plots saved under: {model_dir / 'plots_phase' / 'ureaF_unit'}")
            print(f"UreaF gallery summary saved under: {DEFAULT_GALLERY_OUT_DIR / 'ureaF_unit_ht' / 'run_summary.json'}")
    


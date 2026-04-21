"""
Fixed-operating-point green urea planning model.

The optimizer chooses:
1. Fixed design capacities for storage, wind, heat recovery, and electrolysis.
2. One time-invariant operating point for each process unit at fixed reactor geometry.
3. Hourly on/off decisions and storage dispatch around those fixed points.

The trained Keras ReLU surrogate bundles are embedded once at the operating
point through OMLT neural-network formulations.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.core import TransformationFactory
from pyomo.repn.standard_repn import generate_standard_repn

try:
    from omlt import OmltBlock
except Exception as exc:
    raise ImportError("Could not import OmltBlock from OMLT.") from exc

try:
    from omlt.io.keras import load_keras_sequential
    from omlt.neuralnet import NetworkDefinition, ReluBigMFormulation
except Exception as exc:
    raise ImportError("Could not import OMLT neural-network classes.") from exc


def patch_omlt_linear_tree_bug() -> None:
    import numpy as _np
    import pyomo.environ as _pe
    from itertools import product
    from pyomo.gdp import Disjunct
    import omlt.linear_tree.lt_formulation as ltf

    def _patched_add_gdp_formulation_to_block(
        block,
        model_definition,
        input_vars,
        output_vars,
        transformation,
        epsilon,
        include_leaf_equalities,
    ):
        leaves = model_definition.leaves
        scaled_input_bounds = model_definition.scaled_input_bounds
        unscaled_input_bounds = model_definition.unscaled_input_bounds
        n_inputs = model_definition.n_inputs
        n_outputs = model_definition.n_outputs

        tree_ids = list(leaves.keys())
        tree_leaf_pairs = [
            (tree_id, leaf_id)
            for tree_id in tree_ids
            for leaf_id in leaves[tree_id]
        ]
        features = _np.arange(0, n_inputs)
        output_indices = _np.arange(0, n_outputs)
        intermediate_index = list(product(tree_ids, output_indices))

        scaled_output_bounds = ltf._build_output_bounds(
            model_definition,
            scaled_input_bounds,
        )
        if unscaled_input_bounds is not None:
            unscaled_output_bounds = ltf._build_output_bounds(
                model_definition,
                unscaled_input_bounds,
            )
        else:
            unscaled_output_bounds = scaled_output_bounds.copy()

        for output_idx in output_indices:
            block.outputs[output_idx].setlb(unscaled_output_bounds[output_idx, 0])
            block.outputs[output_idx].setub(unscaled_output_bounds[output_idx, 1])
            block.scaled_outputs[output_idx].setlb(scaled_output_bounds[output_idx, 0])
            block.scaled_outputs[output_idx].setub(scaled_output_bounds[output_idx, 1])

        if model_definition.is_scaled is True:
            block.intermediate_output = _pe.Var(
                intermediate_index,
                bounds=(
                    float(_np.min(scaled_output_bounds[:, 0])),
                    float(_np.max(scaled_output_bounds[:, 1])),
                ),
            )
        else:
            block.intermediate_output = _pe.Var(
                intermediate_index,
                bounds=(
                    float(_np.min(unscaled_output_bounds[:, 0])),
                    float(_np.max(unscaled_output_bounds[:, 1])),
                ),
            )

        def disjuncts_rule(dsj, tree_id, leaf_id):
            dsj.lb_constraint = _pe.Constraint(
                features,
                rule=lambda _dsj, feat_idx: (
                    input_vars[feat_idx]
                    >= leaves[tree_id][leaf_id]["bounds"][feat_idx][0] + epsilon
                ),
            )
            dsj.ub_constraint = _pe.Constraint(
                features,
                rule=lambda _dsj, feat_idx: (
                    input_vars[feat_idx]
                    <= leaves[tree_id][leaf_id]["bounds"][feat_idx][1]
                ),
            )

            if include_leaf_equalities:
                slope = _np.array(leaves[tree_id][leaf_id]["slope"].T)
                intercept = leaves[tree_id][leaf_id]["intercept"]
                if slope.ndim == 1:
                    slope = slope.reshape(-1, 1)
                if isinstance(intercept, (int, float, _np.number)):
                    intercept = _np.array([intercept])

                dsj.linear_exp = _pe.ConstraintList()
                for output_idx in output_indices:
                    dsj.linear_exp.add(
                        sum(
                            slope[k][output_idx] * input_vars[k]
                            for k in features
                        )
                        + intercept[output_idx]
                        == block.intermediate_output[tree_id, output_idx]
                    )

        block.disjunct = Disjunct(tree_leaf_pairs, rule=disjuncts_rule)

        @block.Disjunction(tree_ids)
        def disjunction_rule(_block, tree_id):
            return [
                block.disjunct[tree_id, leaf_id]
                for leaf_id in leaves[tree_id]
            ]

        block.total_output = _pe.ConstraintList()
        for output_idx in output_indices:
            block.total_output.add(
                output_vars[output_idx]
                == sum(
                    block.intermediate_output[tree_id, output_idx]
                    for tree_id in tree_ids
                )
            )

        if transformation != "custom":
            _pe.TransformationFactory("gdp." + transformation).apply_to(block)

    ltf._add_gdp_formulation_to_block = _patched_add_gdp_formulation_to_block


patch_omlt_linear_tree_bug()


THIS_DIR = Path(__file__).resolve().parent
SURROGATE_FILE = THIS_DIR / "surrogate_functions.py"
MODEL_DIR = THIS_DIR / "trained_unit_surrogates"
WIND_CSV = THIS_DIR / "uk_hornsea2_wind_availability_2024.csv"
RESULTS_CSV = THIS_DIR / "ipps_solution_smallhorizon_free_grid.csv"

USE_FULL_YEAR = False
DAYS = 365
N_PERIODS = None if USE_FULL_YEAR else 24 * DAYS
HORIZON_MODE = "random_3day_to_1day"  # "calendar", "random_3day_to_1day"
RANDOM_DAY_REDUCTION_FACTOR = 10
RANDOM_DAY_REDUCTION_SEED = 20260415
SOLVER_NAME = "gurobi"
TIME_LIMIT_SEC = 10 * 60 

GRID_MODE = "free_grid"  # "free_grid", "grid_5pct", "grid_10pct", "wind_only"
ENABLE_OXYGEN_REVENUE = True
FORCE_AMMONIA_ALWAYS_ON = False
FORCE_UREA_ALWAYS_ON = True
WRITE_LP_DEBUG = False
AUTO_WRITE_IIS = True
USE_MANUAL_WARMSTART = True

CRF = 0.05
c_B = 104.0  # GBP per kWh battery capacity
c_wind = 864_545.45
c_el = 4_130.4  # GBP per kWh electrolyzer capacity (applied to E_cap_incremental)
c_H = 645.89  # GBP per kg-H2 hydrogen storage
c_C = 0.42  # GBP per kg-LCO2 liquid CO2 storage
c_HX = 2.07  # GBP per kW heat-exchanger capacity
c_NH3 = 1.89  # GBP per kg-LNH3 ammonia storage
c_Nplant_base = 191.6
c_Nplant_scale = 0.001067
c_Uplant_base = 95.81
c_Uplant_scale = 0.000356
c_grid_default = 0.18
c_CO2 = 0.02
c_H2O_el = 0.00  # GBP per kg-H2O electrolyzer make-up water (placeholder)
c_Ft_feed = 0.00
c_O2_sale = 0.03
eta_el = 0.98
eta_eheater = 1.0
eta_H_ch = 0.98
eta_H_dis = 0.98
eta_B_ch = 0.95
eta_B_dis = 0.95
eta_N_ch = 0.995
eta_N_dis = 0.995
#eta_rec_A = 0.30
#eta_rec_U = 0.60
eta_rec_A = 0.6
eta_rec_U = 0.6
stripper_duty_kwh_per_kg_ammonia = 7.1
FT_SUPPLY_DEFAULT = None  # if None, use the ammonia surrogate upper input bound
Fbar_C = 200.0
UREA_PRODUCT_WTFRAC_MIN = 0.325
D_ann = 125.0 * 365.0
alpha_H_ch = 2.0 # Linear scaling for required energy to store hydrogen
alpha_H_dis = 0.1 # Linear scaling for required energy to discharge hydrogen
alpha_C = 0.02 # Linear scaling for CO2 storage
BATTERY_POWER_MAX = 50_000.0
MMKCAL_PER_HR_TO_KW = 1_000_000.0 / 860.4206500956024
KW_PER_KWHPH = 1.0  # 1 kWh/h is 1 kW, so surrogate electric_kwhph outputs are kW-equivalent.
TON_PER_YEAR_PER_KGPH = 8000.0 / 1000.0
MILLION = 1_000_000.0
AMMONIA_FIXED_LENGTH_M = 10.0
AMMONIA_FIXED_DIAMETER_M = 1.0
UREA_FIXED_LENGTH_M = 20.0
UREA_FIXED_DIAMETER_M = 2.5
AMMONIA_FIXED_VOLUME_M3 = math.pi * AMMONIA_FIXED_DIAMETER_M**2 * AMMONIA_FIXED_LENGTH_M / 4.0
UREA_FIXED_VOLUME_M3 = math.pi * UREA_FIXED_DIAMETER_M**2 * UREA_FIXED_LENGTH_M / 4.0
AMMONIAF_MODEL_INPUT_BOUND_OVERRIDES = {
    "Ft_op": (8500.0, 9080.0),
    "Fh2_op": (5.0, 15.0),
}
UREAF_MODEL_INPUT_BOUND_OVERRIDES = {
    "Fnh3_op": (5.0, 12.0),
    "Fco2_op": (0.0, 50.0),
}
UNIT_MODEL_INPUT_BOUND_OVERRIDES = {
    "ammoniaF_unit": AMMONIAF_MODEL_INPUT_BOUND_OVERRIDES,
    "ureaF_unit": UREAF_MODEL_INPUT_BOUND_OVERRIDES,
}

MANUAL_WARMSTART_VALUES = {
    "W_cap": 525.476298,
    "E_cap": 454.845634,
    "E_cap_incremental": 443.472920,
    "E_cap_included_in_nh3": 11.372715,
    "B_cap": 535.384596,
    "H_cap": 0.0,
    "C_cap": 0.000000,
    "NH3_cap": 37.274962,
    "HX_cap": 79.131350,
    "Ft_op": 8500.000000,
    "Fh2_op": 8.914974,
    "Fnh3_op": 9.272147,
    "Fco2_op": 49.386426,
    "F_NH3_A_op": 11.145261,
    "F_H2O_A_op": 2285.211766,
    "E_A_op": 374.631923,
    "F_U_op": 5.208333,
    "w_U_op": 0.339085,
    "E_U_op": 161.038706,
}

AMMONIAF_DUTY_COLUMNS = ("Qh1", "Qc1", "Qr1", "Qcomp")
UREAF_DUTY_COLUMNS = (
    "QB3",
    "QB6",
    "QB27",
    "QB7_reb",
    "QB7_cond",
    "QB25_reb",
    "QB28_reb",
    "QB28_cond",
    "QR01",
)
AMMONIAF_RESULTS_LIVE_CSV = THIS_DIR / "ammoniaF_results_live.csv"
UREAF_RESULTS_LIVE_CSV = THIS_DIR / "ureaF_results_live.csv"

EXPECTED_OUTPUTS = {
    "ammoniaF_unit": ("ammonia_kgph", "water_kgph", "electric_kwhph"),
    "ureaF_unit": ("pure_urea_kgph", "product_urea_wtfrac", "electric_kwhph"),
}

FEATURE_ALIASES = {
    "ammoniaF_unit": {
        "Ft": "Ft_op",
        "Ft_op": "Ft_op",
        "Fh2": "Fh2_op",
        "Fh2_op": "Fh2_op",
    },
    "ureaF_unit": {
        "nh3_feed_kgph": "Fnh3_op",
        "Fnh3": "Fnh3_op",
        "Fnh3_op": "Fnh3_op",
        "co2_feed_kgph": "Fco2_op",
        "Fco2": "Fco2_op",
        "Fco2_op": "Fco2_op",
    },
}

EXPECTED_MODEL_INPUT_KEYS = {
    "ammoniaF_unit": ("Ft_op", "Fh2_op"),
    "ureaF_unit": ("Fnh3_op", "Fco2_op"),
}


@dataclass(frozen=True)
class LoadedSurrogate:
    unit_name: str
    bundle: object
    feature_to_model_key: dict[str, str]
    model_input_bounds: dict[str, tuple[float, float]]
    input_bounds: dict[str, tuple[float, float]]
    output_bounds: dict[str, tuple[float, float]]
    definitions: dict[str, NetworkDefinition]


def load_surrogate_module():
    if not SURROGATE_FILE.exists():
        raise FileNotFoundError(f"surrogate_functions.py not found at {SURROGATE_FILE}")

    spec = importlib.util.spec_from_file_location("surrogate_functions", SURROGATE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {SURROGATE_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Some saved joblib files were pickled while surrogate_functions.py was run
    # as __main__, so expose the bundle class there before joblib.load(...).
    sys.modules["__main__"].UnitSurrogateBundle = module.UnitSurrogateBundle
    return module


def _coerce_bounds_dict(
    raw_bounds: object,
    names: list[str],
    label: str,
    unit_name: str,
) -> dict[str, tuple[float, float]]:
    if not isinstance(raw_bounds, Mapping):
        raise ValueError(
            f"{unit_name} bundle is missing `{label}` metadata. "
            "The operating-point MILP needs explicit surrogate bounds."
        )

    missing = [name for name in names if name not in raw_bounds]
    if missing:
        raise ValueError(
            f"{unit_name} bundle `{label}` metadata is incomplete. Missing: {missing}"
        )

    clean_bounds: dict[str, tuple[float, float]] = {}
    for name in names:
        lb_raw, ub_raw = raw_bounds[name]
        lb = float(lb_raw)
        ub = float(ub_raw)
        if not np.isfinite(lb) or not np.isfinite(ub) or lb >= ub:
            raise ValueError(
                f"{unit_name} bundle has invalid `{label}` for {name}: {(lb_raw, ub_raw)}"
            )
        clean_bounds[name] = (lb, ub)
    return clean_bounds


def _override_unit_input_bounds(
    unit_name: str,
    feature_to_model_key: dict[str, str],
    input_bounds: dict[str, tuple[float, float]],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    requested_bounds = UNIT_MODEL_INPUT_BOUND_OVERRIDES.get(unit_name, {})
    updated_input_bounds: dict[str, tuple[float, float]] = {}
    model_input_bounds: dict[str, tuple[float, float]] = {}

    for feature_name, existing_bounds in input_bounds.items():
        model_key = feature_to_model_key[feature_name]
        bounds = requested_bounds.get(model_key, existing_bounds)
        lower = float(bounds[0])
        upper = float(bounds[1])
        if lower >= upper:
            raise ValueError(
                f"{unit_name} has invalid constrained input bounds for {model_key}: {bounds}"
            )

        updated_input_bounds[feature_name] = (lower, upper)
        prior_bounds = model_input_bounds.get(model_key)
        if prior_bounds is not None and prior_bounds != updated_input_bounds[feature_name]:
            raise ValueError(
                f"{unit_name} maps multiple features onto {model_key} with inconsistent "
                f"constrained bounds: {prior_bounds} versus {updated_input_bounds[feature_name]}"
            )
        model_input_bounds[model_key] = updated_input_bounds[feature_name]

    return updated_input_bounds, model_input_bounds


def validate_and_load_bundle(module, unit_name: str) -> LoadedSurrogate:
    bundle_path = MODEL_DIR / f"{unit_name}.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing trained surrogate bundle: {bundle_path}\n"
            "Run the surrogate-functions training workflow first."
        )

    bundle = joblib.load(bundle_path)
    model_kind = str(getattr(bundle, "model_kind", "")).strip().lower()
    if model_kind and model_kind != "relu_ann":
        raise ValueError(
            f"{unit_name} bundle has model_kind={model_kind!r}, but this script expects "
            "the Keras ReLU surrogate bundles produced by surrogate_functions.py."
        )
    expected_outputs = list(EXPECTED_OUTPUTS[unit_name])
    bundle_outputs = list(getattr(bundle, "output_names", []))
    if bundle_outputs != expected_outputs:
        raise ValueError(
            f"{unit_name} bundle outputs do not match the new direct-output "
            f"formulation.\nExpected: {expected_outputs}\nFound:    {bundle_outputs}\n"
            "The saved surrogate still looks like the old formulation."
        )

    if unit_name == "ammoniaF_unit":
        module_outputs = list(
            getattr(module, "AMMONIA_COMPONENT_OUTPUT_COLUMNS", expected_outputs)
        )
    else:
        module_outputs = list(
            getattr(module, "UREA_DIRECT_OUTPUT_COLUMNS", expected_outputs)
        )
    if module_outputs != expected_outputs:
        raise ValueError(
            f"surrogate_functions.py exposes {module_outputs} for {unit_name}, but "
            f"the optimizer expects {expected_outputs}. Retrain and save the updated "
            "surrogates before solving."
        )

    feature_aliases = FEATURE_ALIASES[unit_name]
    bundle_features = list(getattr(bundle, "feature_names", []))
    if not bundle_features:
        raise ValueError(f"{unit_name} bundle does not expose any feature names.")

    feature_to_model_key: dict[str, str] = {}
    for feature_name in bundle_features:
        if feature_name not in feature_aliases:
            raise ValueError(
                f"{unit_name} bundle uses unsupported input feature `{feature_name}`.\n"
                f"Supported names: {sorted(feature_aliases)}"
            )
        feature_to_model_key[feature_name] = feature_aliases[feature_name]

    mapped_keys = list(feature_to_model_key.values())
    expected_model_keys = list(EXPECTED_MODEL_INPUT_KEYS[unit_name])
    if sorted(mapped_keys) != sorted(expected_model_keys):
        raise ValueError(
            f"{unit_name} bundle inputs are not compatible with the fixed operating "
            f"point formulation.\nExpected conceptual inputs: {expected_model_keys}\n"
            f"Mapped bundle inputs: {mapped_keys}"
        )

    input_bounds = _coerce_bounds_dict(
        getattr(bundle, "input_bounds", None),
        bundle_features,
        "input_bounds",
        unit_name,
    )
    output_bounds = _coerce_bounds_dict(
        getattr(bundle, "output_bounds", None),
        expected_outputs,
        "output_bounds",
        unit_name,
    )

    input_bounds, model_input_bounds = _override_unit_input_bounds(
        unit_name,
        feature_to_model_key,
        input_bounds,
    )

    scaled_input_bounds = {
        input_idx: input_bounds[feature_name]
        for input_idx, feature_name in enumerate(bundle_features)
    }
    definitions = {
        output_name: load_keras_sequential(
            bundle.models[output_name],
            scaling_object=None,
            scaled_input_bounds=scaled_input_bounds,
        )
        for output_name in expected_outputs
    }

    return LoadedSurrogate(
        unit_name=unit_name,
        bundle=bundle,
        feature_to_model_key=feature_to_model_key,
        model_input_bounds=model_input_bounds,
        input_bounds=input_bounds,
        output_bounds=output_bounds,
        definitions=definitions,
    )


def load_surrogates() -> dict[str, LoadedSurrogate]:
    module = load_surrogate_module()
    return {
        "ammoniaF_unit": validate_and_load_bundle(module, "ammoniaF_unit"),
        "ureaF_unit": validate_and_load_bundle(module, "ureaF_unit"),
    }


def _truncate_to_whole_days(df: pd.DataFrame) -> pd.DataFrame:
    whole_days = len(df) // 24
    if whole_days <= 0:
        raise ValueError("The wind availability file does not contain a full day of data.")
    return df.iloc[: 24 * whole_days].copy()


def _sample_random_days_by_block(
    df: pd.DataFrame,
    reduction_factor: int,
    seed: int,
) -> pd.DataFrame:
    if reduction_factor <= 1:
        return df.reset_index(drop=True)

    n_days = len(df) // 24
    if n_days <= 0:
        raise ValueError("The sampled horizon requires at least one full day of wind data.")

    rng = np.random.default_rng(seed)
    sampled_day_frames: list[pd.DataFrame] = []
    for block_idx, block_start_day in enumerate(range(0, n_days, reduction_factor)):
        block_end_day = min(block_start_day + reduction_factor, n_days)
        chosen_day = int(rng.integers(block_start_day, block_end_day))
        day_slice = df.iloc[24 * chosen_day : 24 * (chosen_day + 1)].copy()
        day_slice["source_day_index"] = chosen_day
        day_slice["source_hour_index"] = np.arange(24 * chosen_day, 24 * (chosen_day + 1))
        day_slice["sample_block_index"] = block_idx
        sampled_day_frames.append(day_slice)

    sampled_df = pd.concat(sampled_day_frames, ignore_index=True)
    sampled_df.attrs["sampling_seed"] = seed
    sampled_df.attrs["sampling_factor"] = reduction_factor
    return sampled_df


def read_wind_data(csv_path: Path, n_periods: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "cf_wind" not in df.columns:
        raise KeyError(
            f"'cf_wind' column is missing from {csv_path}. Found columns: {list(df.columns)}"
        )
    if "time_utc" not in df.columns:
        df["time_utc"] = np.arange(len(df))
    if n_periods is not None:
        df = df.iloc[:n_periods].copy()
    df = _truncate_to_whole_days(df)
    requested_days = len(df) // 24

    if HORIZON_MODE == "calendar":
        df = df.reset_index(drop=True)
    elif HORIZON_MODE == "random_3day_to_1day":
        df = _sample_random_days_by_block(
            df,
            reduction_factor=RANDOM_DAY_REDUCTION_FACTOR,
            seed=RANDOM_DAY_REDUCTION_SEED,
        )
    else:
        raise ValueError(
            f"Unsupported HORIZON_MODE={HORIZON_MODE!r}. "
            "Use 'calendar' or 'random_3day_to_1day'."
        )

    df.attrs["requested_days"] = requested_days
    df.attrs["effective_days"] = len(df) // 24
    df.attrs["horizon_mode"] = HORIZON_MODE
    if df.empty:
        raise ValueError("The wind availability file produced an empty planning horizon.")
    return df


def add_fixed_operating_point_link(
    model: pyo.ConcreteModel,
    actual_var,
    op_var,
    on_var,
    lb: float,
    ub: float,
) -> None:
    model.fixed_point_links.add(actual_var <= ub * on_var)
    model.fixed_point_links.add(actual_var >= lb * on_var)
    model.fixed_point_links.add(actual_var <= op_var - lb * (1 - on_var))
    model.fixed_point_links.add(actual_var >= op_var - ub * (1 - on_var))


def initialize_var_within_bounds(var, lb: float, ub: float) -> None:
    current_value = var.value
    if current_value is None or current_value < lb or current_value > ub:
        var.set_value(lb)


def configure_surrogate_block_domain(
    domain_constraints: pyo.ConstraintList,
    block: OmltBlock,
    feature_names: list[str] | tuple[str, ...],
    input_bounds: Mapping[str, tuple[float, float]],
    output_name: str,
    output_bounds: Mapping[str, tuple[float, float]],
) -> None:
    for input_idx, feature_name in enumerate(feature_names):
        lb, ub = input_bounds[feature_name]
        initialize_var_within_bounds(block.inputs[input_idx], lb, ub)
        block.inputs[input_idx].setlb(lb)
        block.inputs[input_idx].setub(ub)
        domain_constraints.add(block.inputs[input_idx] >= lb)
        domain_constraints.add(block.inputs[input_idx] <= ub)
        if hasattr(block, "scaled_inputs") and input_idx in block.scaled_inputs:
            initialize_var_within_bounds(block.scaled_inputs[input_idx], lb, ub)
            block.scaled_inputs[input_idx].setlb(lb)
            block.scaled_inputs[input_idx].setub(ub)

    out_lb, out_ub = output_bounds[output_name]
    initialize_var_within_bounds(block.outputs[0], out_lb, out_ub)
    block.outputs[0].setlb(out_lb)
    block.outputs[0].setub(out_ub)
    domain_constraints.add(block.outputs[0] >= out_lb)
    domain_constraints.add(block.outputs[0] <= out_ub)
    if hasattr(block, "scaled_outputs") and 0 in block.scaled_outputs:
        initialize_var_within_bounds(block.scaled_outputs[0], out_lb, out_ub)
        block.scaled_outputs[0].setlb(out_lb)
        block.scaled_outputs[0].setub(out_ub)


def build_model(
    surrogates: dict[str, LoadedSurrogate],
    wind_df: pd.DataFrame,
) -> pyo.ConcreteModel:
    ammonia = surrogates["ammoniaF_unit"]
    urea = surrogates["ureaF_unit"]

    t_list = list(range(len(wind_df)))
    first_t = t_list[0]
    last_t = t_list[-1]
    ft_supply_default = (
        ammonia.model_input_bounds["Ft_op"][1]
        if FT_SUPPLY_DEFAULT is None
        else FT_SUPPLY_DEFAULT
    )

    m = pyo.ConcreteModel("green_urea_fixed_operating_point")
    m.ammonia_feature_order = tuple(ammonia.bundle.feature_names)
    m.urea_feature_order = tuple(urea.bundle.feature_names)
    m._ammonia_network_definitions = dict(ammonia.definitions)
    m._urea_network_definitions = dict(urea.definitions)
    m.T = pyo.Set(initialize=t_list, ordered=True)
    m.NY_A = pyo.Set(initialize=list(EXPECTED_OUTPUTS["ammoniaF_unit"]), ordered=True)
    m.NY_U = pyo.Set(initialize=list(EXPECTED_OUTPUTS["ureaF_unit"]), ordered=True)

    m.dt = pyo.Param(initialize=1.0, mutable=True)
    m.CF = pyo.Param(
        m.T,
        initialize={t: float(wind_df.loc[t, "cf_wind"]) for t in t_list},
        within=pyo.NonNegativeReals,
    )
    m.c_grid = pyo.Param(
        m.T,
        initialize={t: c_grid_default for t in t_list},
        mutable=True,
    )
    m.Fbar_Ft = pyo.Param(
        m.T,
        initialize={t: ft_supply_default for t in t_list},
        mutable=True,
    )
    m.Fbar_C = pyo.Param(
        m.T,
        initialize={t: Fbar_C for t in t_list},
        mutable=True,
    )

    # Compute upper bounds before creating the capacity vars
    n_hours = len(t_list)

    # Tight/safe from existing model structure
    e_cap_ub = 100.0 * ammonia.model_input_bounds["Fh2_op"][1] / eta_el
    e_cap_included_ub = ammonia.output_bounds["ammonia_kgph"][1] / eta_el
    e_cap_incremental_ub = e_cap_ub
    hx_cap_ub = stripper_duty_kwh_per_kg_ammonia * ammonia.output_bounds["ammonia_kgph"][1]
    c_cap_ub = sum(float(pyo.value(m.Fbar_C[t])) for t in m.T)
    nh3_cap_ub = n_hours * ammonia.output_bounds["ammonia_kgph"][1]
    hydrogen_charge_flow_ub = 50.0 * ammonia.model_input_bounds["Fh2_op"][1]
    hydrogen_discharge_flow_ub = ammonia.model_input_bounds["Fh2_op"][1]
    ammonia_charge_flow_ub = ammonia.output_bounds["ammonia_kgph"][1]
    ammonia_discharge_flow_ub = urea.model_input_bounds["Fnh3_op"][1]

    # These 3 need engineering caps or a conservative design envelope
    w_cap_ub = 1500.0
    b_cap_ub = 10000.0
    h_cap_ub = n_hours * ammonia.model_input_bounds["Fh2_op"][1]

    m.W_cap = pyo.Var(bounds=(0.0, w_cap_ub))
    m.E_cap = pyo.Var(bounds=(0.0, e_cap_ub))
    m.E_cap_incremental = pyo.Var(bounds=(0.0, e_cap_incremental_ub))
    m.E_cap_included_in_nh3 = pyo.Var(bounds=(0.0, e_cap_included_ub))
    m.B_cap = pyo.Var(bounds=(0.0, b_cap_ub))
    m.H_cap = pyo.Var(bounds=(0.0, h_cap_ub))
    m.C_cap = pyo.Var(bounds=(0.0, c_cap_ub))
    m.NH3_cap = pyo.Var(bounds=(0.0, nh3_cap_ub))
    m.HX_cap = pyo.Var(bounds=(0.0, hx_cap_ub))

    m.ammonia_reactor_length_m = pyo.Param(initialize=AMMONIA_FIXED_LENGTH_M)
    m.ammonia_reactor_diameter_m = pyo.Param(initialize=AMMONIA_FIXED_DIAMETER_M)
    m.urea_reactor_length_m = pyo.Param(initialize=UREA_FIXED_LENGTH_M)
    m.urea_reactor_diameter_m = pyo.Param(initialize=UREA_FIXED_DIAMETER_M)
    m.Ft_op = pyo.Var(bounds=ammonia.model_input_bounds["Ft_op"])
    m.Fh2_op = pyo.Var(bounds=ammonia.model_input_bounds["Fh2_op"])
    m.Fnh3_op = pyo.Var(bounds=urea.model_input_bounds["Fnh3_op"])
    m.Fco2_op = pyo.Var(bounds=urea.model_input_bounds["Fco2_op"])

    m.F_NH3_A_op = pyo.Var(bounds=ammonia.output_bounds["ammonia_kgph"])
    m.F_H2O_A_op = pyo.Var(bounds=ammonia.output_bounds["water_kgph"])
    m.E_A_op = pyo.Var(bounds=ammonia.output_bounds["electric_kwhph"])
    m.F_U_op = pyo.Var(bounds=urea.output_bounds["pure_urea_kgph"])
    m.w_U_op = pyo.Var(bounds=urea.output_bounds["product_urea_wtfrac"])
    m.E_U_op = pyo.Var(bounds=urea.output_bounds["electric_kwhph"])

    m.y_A_on = pyo.Var(m.T, within=pyo.Binary)
    m.y_U_on = pyo.Var(m.T, within=pyo.Binary)
    if FORCE_AMMONIA_ALWAYS_ON:
        for t in t_list:
            m.y_A_on[t].fix(1)
    if FORCE_UREA_ALWAYS_ON:
        for t in t_list:
            m.y_U_on[t].fix(1)

    m.P_wind = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.P_grid = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.P_el = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.M_H2_prod = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.I_B = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.ch_B = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.dis_B = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.bat_charge_mode = pyo.Var(m.T, within=pyo.Binary)
    m.hydrogen_charge_mode = pyo.Var(m.T, within=pyo.Binary)
    m.ammonia_charge_mode = pyo.Var(m.T, within=pyo.Binary)
    m.P_wind_to_el = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.P_wind_to_heat = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.P_wind_to_batt = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.P_batt_to_el = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.P_batt_to_heat = pyo.Var(m.T, within=pyo.NonNegativeReals)

    m.Ft = pyo.Var(m.T, bounds=(0.0, ammonia.model_input_bounds["Ft_op"][1]))
    m.Fh2 = pyo.Var(m.T, bounds=(0.0, ammonia.model_input_bounds["Fh2_op"][1]))
    m.Fnh3 = pyo.Var(m.T, bounds=(0.0, urea.model_input_bounds["Fnh3_op"][1]))
    m.Fco2 = pyo.Var(m.T, bounds=(0.0, urea.model_input_bounds["Fco2_op"][1]))
    m.F_NH3_A = pyo.Var(m.T, bounds=(0.0, ammonia.output_bounds["ammonia_kgph"][1]))
    m.F_H2O_A = pyo.Var(m.T, bounds=(0.0, ammonia.output_bounds["water_kgph"][1]))
    m.E_A = pyo.Var(m.T, bounds=(0.0, ammonia.output_bounds["electric_kwhph"][1]))
    m.F_U = pyo.Var(m.T, bounds=(0.0, urea.output_bounds["pure_urea_kgph"][1]))
    m.w_U = pyo.Var(m.T, bounds=(0.0, urea.output_bounds["product_urea_wtfrac"][1]))
    m.E_U = pyo.Var(m.T, bounds=(0.0, urea.output_bounds["electric_kwhph"][1]))

    m.F_NH3_strip = pyo.Var(m.T, bounds=(0.0, ammonia.output_bounds["ammonia_kgph"][1]))
    max_qs = stripper_duty_kwh_per_kg_ammonia * ammonia.output_bounds["ammonia_kgph"][1]
    m.Q_S = pyo.Var(m.T, bounds=(0.0, max_qs))

    m.I_H = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.ch_H = pyo.Var(m.T, bounds=(0.0, hydrogen_charge_flow_ub))
    m.dis_H = pyo.Var(m.T, bounds=(0.0, hydrogen_discharge_flow_ub))
    m.I_C = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.Ft_spill = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.Fco2_spill = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.I_NH3 = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.ch_NH3 = pyo.Var(m.T, bounds=(0.0, ammonia_charge_flow_ub))
    m.dis_NH3 = pyo.Var(m.T, bounds=(0.0, ammonia_discharge_flow_ub))
    m.Q_rec = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.Q_HU = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.P_HU_el = pyo.Var(m.T, within=pyo.NonNegativeReals)

    m.P_H_storage = pyo.Expression(
        m.T,
        rule=lambda mdl, t: alpha_H_ch * mdl.ch_H[t] + alpha_H_dis * mdl.dis_H[t],
    )
    m.P_CO2_feed = pyo.Expression(
        m.T,
        rule=lambda mdl, t: alpha_C * mdl.Fco2[t],
    )
    m.M_O2_prod = pyo.Expression(
        m.T,
        rule=lambda mdl, t: 8.0 * mdl.M_H2_prod[t],
    )
    m.M_H2O_el = pyo.Expression(
        m.T,
        # Electrolysis consumes 1 mol H2O per mol H2, i.e. 9 kg-H2O per kg-H2.
        rule=lambda mdl, t: 9.0 * mdl.M_H2_prod[t],
    )
    m.E_A_kw = pyo.Expression(
        m.T,
        rule=lambda mdl, t: KW_PER_KWHPH * mdl.E_A[t],
    )
    m.E_U_kw = pyo.Expression(
        m.T,
        rule=lambda mdl, t: KW_PER_KWHPH * mdl.E_U[t],
    )
    m.total_electric_load = pyo.Expression(
        m.T,
        rule=lambda mdl, t: (
            mdl.P_el[t]
            + mdl.E_A_kw[t]
            + mdl.E_U_kw[t]
            + mdl.P_HU_el[t]
            + mdl.P_H_storage[t]
            + mdl.P_CO2_feed[t]
            + mdl.ch_B[t]
        ),
    )
    m.ammonia_reactor_volume_m3 = pyo.Expression(expr=AMMONIA_FIXED_VOLUME_M3)
    m.urea_reactor_volume_m3 = pyo.Expression(expr=UREA_FIXED_VOLUME_M3)
    # Plant CAPEX correlations are defined on annual product capacity (t/y),
    # while only the electrolyzer overbuild above the ammonia design H2 load is costed separately.
    m.K_nh3_plant_tpy = pyo.Expression(
        expr=TON_PER_YEAR_PER_KGPH * m.F_NH3_A_op,
    )
    m.K_urea_plant_tpy = pyo.Expression(
        expr=TON_PER_YEAR_PER_KGPH * m.F_U_op,
    )

    # Keep E_cap as a design variable so warm starts can seed it directly,
    # then tie it to the ammonia design H2 load with an equality constraint.
    m.electrolyzer_design_capacity_basis = pyo.Constraint(
        expr=m.E_cap == m.Fh2_op*50 / eta_el,
    )
    m.electrolyzer_included_capex_basis = pyo.Constraint(
        expr=m.E_cap_included_in_nh3 == m.F_NH3_A_op / eta_el,
    )

    m.nh3_plant_capex = pyo.Expression(
        expr=MILLION * (c_Nplant_base + c_Nplant_scale * m.K_nh3_plant_tpy),
    )
    m.urea_plant_capex = pyo.Expression(
        expr=MILLION * (c_Uplant_base + c_Uplant_scale * m.K_urea_plant_tpy),
    )

    m.fixed_point_links = pyo.ConstraintList()
    for t in t_list:
        add_fixed_operating_point_link(
            m, m.Ft[t], m.Ft_op, m.y_A_on[t], *ammonia.model_input_bounds["Ft_op"]
        )
        add_fixed_operating_point_link(
            m, m.Fh2[t], m.Fh2_op, m.y_A_on[t], *ammonia.model_input_bounds["Fh2_op"]
        )
        add_fixed_operating_point_link(
            m, m.F_NH3_A[t], m.F_NH3_A_op, m.y_A_on[t], *ammonia.output_bounds["ammonia_kgph"]
        )
        add_fixed_operating_point_link(
            m, m.F_H2O_A[t], m.F_H2O_A_op, m.y_A_on[t], *ammonia.output_bounds["water_kgph"]
        )
        add_fixed_operating_point_link(
            m, m.E_A[t], m.E_A_op, m.y_A_on[t], *ammonia.output_bounds["electric_kwhph"]
        )
        add_fixed_operating_point_link(
            m, m.Fnh3[t], m.Fnh3_op, m.y_U_on[t], *urea.model_input_bounds["Fnh3_op"]
        )
        add_fixed_operating_point_link(
            m, m.Fco2[t], m.Fco2_op, m.y_U_on[t], *urea.model_input_bounds["Fco2_op"]
        )
        add_fixed_operating_point_link(
            m, m.F_U[t], m.F_U_op, m.y_U_on[t], *urea.output_bounds["pure_urea_kgph"]
        )
        add_fixed_operating_point_link(
            m, m.w_U[t], m.w_U_op, m.y_U_on[t], *urea.output_bounds["product_urea_wtfrac"]
        )
        add_fixed_operating_point_link(
            m, m.E_U[t], m.E_U_op, m.y_U_on[t], *urea.output_bounds["electric_kwhph"]
        )

    m.ammonia_tree = OmltBlock(m.NY_A)
    m.urea_tree = OmltBlock(m.NY_U)
    m.ammonia_tree_links = pyo.ConstraintList()
    m.urea_tree_links = pyo.ConstraintList()
    m.ammonia_tree_domain = pyo.ConstraintList()
    m.urea_tree_domain = pyo.ConstraintList()

    ammonia_model_inputs = {
        "Ft_op": m.Ft_op,
        "Fh2_op": m.Fh2_op,
    }
    urea_model_inputs = {
        "Fnh3_op": m.Fnh3_op,
        "Fco2_op": m.Fco2_op,
    }
    ammonia_output_vars = {
        "ammonia_kgph": m.F_NH3_A_op,
        "water_kgph": m.F_H2O_A_op,
        "electric_kwhph": m.E_A_op,
    }
    urea_output_vars = {
        "pure_urea_kgph": m.F_U_op,
        "product_urea_wtfrac": m.w_U_op,
        "electric_kwhph": m.E_U_op,
    }

    for output_name in m.NY_A:
        blk = m.ammonia_tree[output_name]
        blk.build_formulation(
            ReluBigMFormulation(ammonia.definitions[str(output_name)])
        )
        configure_surrogate_block_domain(
            m.ammonia_tree_domain,
            blk,
            ammonia.bundle.feature_names,
            ammonia.input_bounds,
            str(output_name),
            ammonia.output_bounds,
        )
        for input_idx, feature_name in enumerate(ammonia.bundle.feature_names):
            model_key = ammonia.feature_to_model_key[feature_name]
            m.ammonia_tree_links.add(blk.inputs[input_idx] == ammonia_model_inputs[model_key])
        m.ammonia_tree_links.add(blk.outputs[0] == ammonia_output_vars[str(output_name)])

    for output_name in m.NY_U:
        blk = m.urea_tree[output_name]
        blk.build_formulation(
            ReluBigMFormulation(urea.definitions[str(output_name)])
        )
        configure_surrogate_block_domain(
            m.urea_tree_domain,
            blk,
            urea.bundle.feature_names,
            urea.input_bounds,
            str(output_name),
            urea.output_bounds,
        )
        for input_idx, feature_name in enumerate(urea.bundle.feature_names):
            model_key = urea.feature_to_model_key[feature_name]
            m.urea_tree_links.add(blk.inputs[input_idx] == urea_model_inputs[model_key])
        m.urea_tree_links.add(blk.outputs[0] == urea_output_vars[str(output_name)])

    def battery_storage_rule(mdl, t):
        if t == first_t:
            return mdl.I_B[t] == mdl.I_B[last_t] + eta_B_ch * mdl.ch_B[t] - mdl.dis_B[t] / eta_B_dis
        return mdl.I_B[t] == mdl.I_B[t - 1] + eta_B_ch * mdl.ch_B[t] - mdl.dis_B[t] / eta_B_dis

    m.battery_storage_bal = pyo.Constraint(m.T, rule=battery_storage_rule)
    m.battery_storage_cap = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.I_B[t] <= mdl.B_cap)
    m.battery_charge_cap = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.ch_B[t] <= mdl.B_cap)
    m.battery_discharge_cap = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.dis_B[t] <= mdl.B_cap)
    m.battery_charge_only = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.ch_B[t] <= BATTERY_POWER_MAX * mdl.bat_charge_mode[t],
    )
    m.battery_discharge_only = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.dis_B[t] <= BATTERY_POWER_MAX * (1 - mdl.bat_charge_mode[t]),
    )
    m.battery_charge_from_wind = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.ch_B[t] == mdl.P_wind_to_batt[t],
    )

    m.wind_availability = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.P_wind[t] <= mdl.CF[t] * mdl.W_cap,
    )
    m.electrolyzer_cap = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.P_el[t] <= mdl.E_cap,
    )
    m.electrolyzer_incremental_capex_basis = pyo.Constraint(
        expr=m.E_cap_incremental >= m.E_cap - m.E_cap_included_in_nh3,
    )
    m.hydrogen_production = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.M_H2_prod[t] == eta_el * mdl.P_el[t],
    )
    m.electricity_balance = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.total_electric_load[t] <= mdl.P_wind[t] + mdl.P_grid[t] + mdl.dis_B[t],
    )
    m.clean_power_to_electrolyzer = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.P_el[t] + mdl.P_H_storage[t] == mdl.P_wind_to_el[t] + mdl.P_batt_to_el[t],
    )
    m.clean_power_to_heat = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.P_HU_el[t] == mdl.P_wind_to_heat[t] + mdl.P_batt_to_heat[t],
    )
    m.wind_allocation_limit = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.P_wind_to_el[t] + mdl.P_wind_to_heat[t] + mdl.P_wind_to_batt[t] <= mdl.P_wind[t],
    )
    m.battery_allocation_limit = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.P_batt_to_el[t] + mdl.P_batt_to_heat[t] <= mdl.dis_B[t],
    )

    m.feed_total_balance = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.Ft[t] + mdl.Ft_spill[t] == mdl.Fbar_Ft[t],
    )

    def co2_storage_rule(mdl, t):
        if t == first_t:
            return mdl.I_C[t] == mdl.I_C[last_t] + mdl.Fbar_C[t] - mdl.Fco2[t] - mdl.Fco2_spill[t]
        return mdl.I_C[t] == mdl.I_C[t - 1] + mdl.Fbar_C[t] - mdl.Fco2[t] - mdl.Fco2_spill[t]

    m.co2_storage_bal = pyo.Constraint(m.T, rule=co2_storage_rule)
    m.co2_storage_cap = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.I_C[t] <= mdl.C_cap)

    m.hydrogen_flow_balance = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.Fh2[t] + mdl.ch_H[t] == mdl.M_H2_prod[t] + mdl.dis_H[t],
    )

    def hydrogen_storage_rule(mdl, t):
        if t == first_t:
            return mdl.I_H[t] == mdl.I_H[last_t] + eta_H_ch * mdl.ch_H[t] - mdl.dis_H[t] / eta_H_dis
        return mdl.I_H[t] == mdl.I_H[t - 1] + eta_H_ch * mdl.ch_H[t] - mdl.dis_H[t] / eta_H_dis

    m.hydrogen_storage_bal = pyo.Constraint(m.T, rule=hydrogen_storage_rule)
    m.hydrogen_storage_cap = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.I_H[t] <= mdl.H_cap)
    m.hydrogen_charge_only = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.ch_H[t] <= hydrogen_charge_flow_ub * mdl.hydrogen_charge_mode[t],
    )
    m.hydrogen_discharge_only = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.dis_H[t] <= hydrogen_discharge_flow_ub * (1 - mdl.hydrogen_charge_mode[t]),
    )

    m.ammonia_flow_balance = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.Fnh3[t] + mdl.ch_NH3[t] == mdl.F_NH3_strip[t] + mdl.dis_NH3[t],
    )

    def ammonia_storage_rule(mdl, t):
        if t == first_t:
            return mdl.I_NH3[t] == mdl.I_NH3[last_t] + eta_N_ch * mdl.ch_NH3[t] - mdl.dis_NH3[t] / eta_N_dis
        return mdl.I_NH3[t] == mdl.I_NH3[t - 1] + eta_N_ch * mdl.ch_NH3[t] - mdl.dis_NH3[t] / eta_N_dis

    m.ammonia_storage_bal = pyo.Constraint(m.T, rule=ammonia_storage_rule)
    m.ammonia_storage_cap = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.I_NH3[t] <= mdl.NH3_cap)
    m.ammonia_charge_only = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.ch_NH3[t] <= ammonia_charge_flow_ub * mdl.ammonia_charge_mode[t],
    )
    m.ammonia_discharge_only = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.dis_NH3[t] <= ammonia_discharge_flow_ub * (1 - mdl.ammonia_charge_mode[t]),
    )

    m.stripper_ammonia = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.F_NH3_strip[t] == mdl.F_NH3_A[t],
    )
    m.stripper_duty = pyo.Constraint(
        m.T,
        rule=lambda mdl, t: mdl.Q_S[t] == stripper_duty_kwh_per_kg_ammonia * mdl.F_NH3_A[t],
    )

    m.Q_avail = pyo.Expression(
        m.T,
        rule=lambda mdl, t: eta_rec_A * mdl.E_A_kw[t] + eta_rec_U * mdl.E_U_kw[t],
    )
    if UREA_PRODUCT_WTFRAC_MIN is not None:
        m.urea_quality_spec = pyo.Constraint(expr=m.w_U_op >= UREA_PRODUCT_WTFRAC_MIN)
    m.heat_recovery_1 = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.Q_rec[t] <= mdl.Q_S[t])
    m.heat_recovery_2 = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.Q_rec[t] <= mdl.HX_cap)
    m.heat_recovery_3 = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.Q_rec[t] <= mdl.Q_avail[t])
    m.hot_utility = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.Q_HU[t] == mdl.Q_S[t] - mdl.Q_rec[t])
    m.electric_hot_utility = pyo.Constraint(m.T, rule=lambda mdl, t: mdl.P_HU_el[t] == mdl.Q_HU[t] / eta_eheater)

    modeled_hours = float(len(t_list))
    target_rhs = D_ann * (modeled_hours / 8760.0)
    m.production_target = pyo.Constraint(expr=sum(m.dt * m.F_U[t] for t in m.T) >= target_rhs)

    if GRID_MODE == "free_grid":
        pass
    elif GRID_MODE == "grid_5pct":
        m.grid_share_limit = pyo.Constraint(
            expr=sum(m.P_grid[t] for t in m.T) <= 0.05 * sum(m.total_electric_load[t] for t in m.T)
        )
    elif GRID_MODE == "grid_10pct":
        m.grid_share_limit = pyo.Constraint(
            expr=sum(m.P_grid[t] for t in m.T) <= 0.10 * sum(m.total_electric_load[t] for t in m.T)
        )
    elif GRID_MODE == "wind_only":
        for t in t_list:
            m.P_grid[t].fix(0.0)
    else:
        raise ValueError(
            f"Unsupported GRID_MODE={GRID_MODE}. Use 'free_grid', 'grid_5pct', or 'wind_only'."
        )

    annualization_factor = 8760.0 / (pyo.value(m.dt) * modeled_hours)
    capex = CRF * (
        c_wind * m.W_cap
        + c_el * m.E_cap_incremental
        + c_B * m.B_cap
        + c_H * m.H_cap
        + c_C * m.C_cap
        + c_NH3 * m.NH3_cap
        + c_HX * m.HX_cap
        + m.nh3_plant_capex
        + m.urea_plant_capex
    )
    opex = annualization_factor * sum(
        m.dt
        * (
            m.c_grid[t] * m.P_grid[t]
            + c_CO2 * m.Fco2[t]
            + c_H2O_el * m.M_H2O_el[t]
            + c_Ft_feed * m.Ft[t]
            - (c_O2_sale * m.M_O2_prod[t] if ENABLE_OXYGEN_REVENUE else 0.0)
        )
        for t in m.T
    )
    m.obj = pyo.Objective(expr=capex + opex, sense=pyo.minimize)

    TransformationFactory("gdp.hull").apply_to(m)
    return m


def solve_model(model: pyo.ConcreteModel):
    solver = pyo.SolverFactory(SOLVER_NAME)
    if solver is None or not solver.available(False):
        raise RuntimeError(f"Solver `{SOLVER_NAME}` is not available in this environment.")

    seeded_warmstart_values = apply_manual_warm_start(model)
    if seeded_warmstart_values:
        print(f"Applied manual warm start values to {seeded_warmstart_values} variables.")

    if SOLVER_NAME in {"gurobi", "gurobi_direct"}:
        solver.options["TimeLimit"] = TIME_LIMIT_SEC
        solver.options["MIPGap"] = 0.01
        solver.options["MIPFocus"] = 1
        solver.options["NumericFocus"] = 2
        solver.options["Cuts"] = 2
        solver.options["Presolve"] = 2

    solve_kwargs = {"tee": True, "load_solutions": False}
    if seeded_warmstart_values and SOLVER_NAME in {
        "gurobi",
        "gurobi_direct",
        "gurobi_persistent",
        "cplex",
        "cplex_direct",
    }:
        solve_kwargs["warmstart"] = True

    results = solver.solve(model, **solve_kwargs)
    if has_feasible_solution(results):
        model.solutions.load_from(results)
    return results


def has_feasible_solution(results) -> bool:
    try:
        return len(results.solution) > 0
    except Exception:
        return False


def _report_process_energy_basis(
    label: str,
    csv_path: Path,
    duty_columns: tuple[str, ...],
    csv_basis_note: str,
) -> None:
    print(f"\n[{label}]")
    print(f"  duty columns in training/results CSV: {', '.join(duty_columns)}")
    print(f"  CSV duty basis: {csv_basis_note}")
    print("  surrogate output electric_kwhph basis: kWh/h (= kW-equivalent)")

    if not csv_path.exists():
        print(f"  data file not found for live unit audit: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "run_ok" in df.columns:
        run_ok = pd.to_numeric(df["run_ok"], errors="coerce").fillna(0.0)
        df = df.loc[run_ok == 1].copy()

    available = [col for col in duty_columns if col in df.columns]
    if not available:
        print(f"  none of the expected duty columns were found in {csv_path.name}")
        return

    total_abs = (
        df[available]
        .apply(pd.to_numeric, errors="coerce")
        .abs()
        .sum(axis=1)
    )
    total_abs = total_abs[np.isfinite(total_abs.to_numpy())]
    if total_abs.empty:
        print(f"  could not compute total absolute duty statistics from {csv_path.name}")
        return

    converted = total_abs * MMKCAL_PER_HR_TO_KW
    print(
        "  total_abs duty range from data: "
        f"{float(total_abs.min()):.6f} to {float(total_abs.max()):.6f} MMkcal/hr"
    )
    print(
        "  implied electric_kwhph range from data: "
        f"{float(converted.min()):.6f} to {float(converted.max()):.6f} kWh/h"
    )


def report_energy_unit_audit(surrogates: dict[str, LoadedSurrogate]) -> None:
    print("\n=== Process Energy Unit Audit ===")
    print("All power-balance terms are summed on a kWh/h basis, which is numerically kW.")
    print(
        "The surrogate outputs E_A and E_U come from surrogate_functions.py as "
        "`electric_kwhph`, so they are already on the same numeric basis as P_el and P_HU_el."
    )
    print(
        "The large gap between ammonia and urea values comes from the underlying duty magnitudes "
        "used for training, not from a factor-of-1000 mismatch in the optimization summation."
    )

    _report_process_energy_basis(
        label="ammoniaF_unit",
        csv_path=AMMONIAF_RESULTS_LIVE_CSV,
        duty_columns=AMMONIAF_DUTY_COLUMNS,
        csv_basis_note=(
            "run_ammoniaF_grid.py converts Aspen heat duties from cal/s to MMkcal/hr "
            "before writing the CSV"
        ),
    )
    _report_process_energy_basis(
        label="ureaF_unit",
        csv_path=UREAF_RESULTS_LIVE_CSV,
        duty_columns=UREAF_DUTY_COLUMNS,
        csv_basis_note=(
            "run_ureaF_grid.py currently writes the Aspen values as-is; "
            "surrogate_functions.py then interprets those CSV duties as MMkcal/hr"
        ),
    )

    print("\nSaved surrogate output bounds:")
    for unit_name, surrogate in surrogates.items():
        lo, hi = surrogate.output_bounds["electric_kwhph"]
        print(f"  {unit_name}.electric_kwhph: {lo:.6f} to {hi:.6f} kWh/h")


def _scalar_bound_value(bound):
    if bound is None:
        return None
    return float(pyo.value(bound))


def _set_warmstart_value(
    var: pyo.Var,
    value: float,
    name: str,
    seeded_var_ids: set[int] | None = None,
) -> None:
    if not np.isfinite(value):
        raise ValueError(f"Warm start value for {name} must be finite, got {value}.")

    lb = _scalar_bound_value(var.lb)
    ub = _scalar_bound_value(var.ub)
    if lb is not None:
        value = max(value, lb)
    if ub is not None:
        value = min(value, ub)

    # Project integer warm starts back onto an integer-feasible point after clipping.
    if var.is_binary():
        value = 1.0 if value >= 0.5 else 0.0
    elif var.is_integer():
        value = float(round(value))
        if lb is not None:
            value = max(value, lb)
        if ub is not None:
            value = min(value, ub)

    var.set_value(value)
    if seeded_var_ids is not None:
        seeded_var_ids.add(id(var))


def _is_fixed_to(var: pyo.Var, target: float, tol: float = 1e-8) -> bool:
    return bool(var.fixed) and abs(float(pyo.value(var)) - target) <= tol


def _infer_unit_commitment_start_state(
    on_vars,
    operating_point_names: tuple[str, ...],
) -> int:
    if all(_is_fixed_to(on_vars[t], 1.0) for t in on_vars.index_set()):
        return 1
    if all(_is_fixed_to(on_vars[t], 0.0) for t in on_vars.index_set()):
        return 0

    has_complete_operating_point_seed = all(
        name in MANUAL_WARMSTART_VALUES for name in operating_point_names
    )
    return 1 if has_complete_operating_point_seed else 0


def _synchronize_design_basis_warmstart(
    model: pyo.ConcreteModel,
    seeded_var_ids: set[int],
) -> None:
    """
    Recompute design-basis variables from seeded operating-point values.

    Reported final solutions are often rounded for printing, which can make a
    copied warm start miss exact equality constraints by a few e-5. These
    variables are definitionally tied to the operating point, so seed them from
    the model equations instead of trusting rounded report values.
    """
    e_cap = float(pyo.value(50.0 * model.Fh2_op / eta_el))
    _set_warmstart_value(model.E_cap, e_cap, "E_cap", seeded_var_ids)

    e_cap_included = float(pyo.value(model.F_NH3_A_op / eta_el))
    _set_warmstart_value(
        model.E_cap_included_in_nh3,
        e_cap_included,
        "E_cap_included_in_nh3",
        seeded_var_ids,
    )

    e_cap_incremental = max(e_cap - e_cap_included, 0.0)
    _set_warmstart_value(
        model.E_cap_incremental,
        e_cap_incremental,
        "E_cap_incremental",
        seeded_var_ids,
    )


def _seed_tree_block(
    block: OmltBlock,
    input_values: list[float],
    output_value: float,
    label: str,
    seeded_var_ids: set[int],
) -> int:
    seeded = 0
    for idx, value in enumerate(input_values):
        if idx in block.inputs:
            _set_warmstart_value(block.inputs[idx], value, f"{label}.inputs[{idx}]", seeded_var_ids)
            seeded += 1
        if hasattr(block, "scaled_inputs") and idx in block.scaled_inputs:
            _set_warmstart_value(
                block.scaled_inputs[idx],
                value,
                f"{label}.scaled_inputs[{idx}]",
                seeded_var_ids,
            )
            seeded += 1

    if 0 in block.outputs:
        _set_warmstart_value(block.outputs[0], output_value, f"{label}.outputs[0]", seeded_var_ids)
        seeded += 1
        if hasattr(block, "scaled_outputs") and 0 in block.scaled_outputs:
            _set_warmstart_value(
                block.scaled_outputs[0],
                output_value,
                f"{label}.scaled_outputs[0]",
                seeded_var_ids,
            )
            seeded += 1

    return seeded


def _flatten_layer_index(index) -> int:
    if isinstance(index, tuple):
        if len(index) != 1:
            raise ValueError(f"Unexpected OMLT layer index shape: {index!r}")
        return int(index[0])
    return int(index)


def _seed_relu_network_internal_state(
    block: OmltBlock,
    definition: NetworkDefinition,
    input_values: list[float],
    label: str,
    seeded_var_ids: set[int],
) -> int:
    seeded = 0
    current_values = np.asarray(input_values, dtype=float)
    block_layer_ids = list(block.layers)
    network_layers = list(definition.layers)
    if len(block_layer_ids) != len(network_layers):
        raise ValueError(
            f"{label} layer-count mismatch between OMLT block and network definition: "
            f"{len(block_layer_ids)} versus {len(network_layers)}"
        )

    for layer_id, layer_def in zip(block_layer_ids, network_layers):
        layer_block = block.layer[layer_id]
        layer_output_indices = [
            _flatten_layer_index(layer_idx) for layer_idx in layer_def.output_indexes
        ]

        if hasattr(layer_block, "zhat"):
            weights = np.asarray(layer_def.weights, dtype=float)
            biases = np.asarray(layer_def.biases, dtype=float).reshape(-1)
            zhat_values = current_values @ weights + biases
            for idx, value in zip(layer_output_indices, zhat_values, strict=True):
                _set_warmstart_value(
                    layer_block.zhat[idx],
                    float(value),
                    f"{label}.layer[{layer_id}].zhat[{idx}]",
                    seeded_var_ids,
                )
                seeded += 1
        else:
            zhat_values = current_values

        activation_name = str(getattr(layer_def, "activation", "linear")).lower()
        if activation_name == "relu":
            z_values = np.maximum(zhat_values, 0.0)
            q_values = np.where(zhat_values >= 0.0, 1.0, 0.0)
        else:
            z_values = zhat_values
            q_values = None

        if hasattr(layer_block, "z"):
            for idx, value in zip(layer_output_indices, z_values, strict=True):
                _set_warmstart_value(
                    layer_block.z[idx],
                    float(value),
                    f"{label}.layer[{layer_id}].z[{idx}]",
                    seeded_var_ids,
                )
                seeded += 1

        if q_values is not None and hasattr(layer_block, "q_relu"):
            for idx, value in zip(layer_output_indices, q_values, strict=True):
                _set_warmstart_value(
                    layer_block.q_relu[idx],
                    float(value),
                    f"{label}.layer[{layer_id}].q_relu[{idx}]",
                    seeded_var_ids,
                )
                seeded += 1

        current_values = np.asarray(z_values, dtype=float)

    return seeded


def _evaluate_network_output(
    definition: NetworkDefinition,
    input_values: list[float],
) -> np.ndarray:
    current_values = np.asarray(input_values, dtype=float)
    for layer_def in definition.layers:
        if hasattr(layer_def, "weights"):
            weights = np.asarray(layer_def.weights, dtype=float)
            biases = np.asarray(layer_def.biases, dtype=float).reshape(-1)
            zhat_values = current_values @ weights + biases
        else:
            zhat_values = current_values

        activation_name = str(getattr(layer_def, "activation", "linear")).lower()
        if activation_name == "relu":
            current_values = np.maximum(zhat_values, 0.0)
        else:
            current_values = zhat_values

    return np.asarray(current_values, dtype=float)


def _load_results_csv_for_warmstart(
    model: pyo.ConcreteModel,
) -> pd.DataFrame | None:
    if not RESULTS_CSV.exists():
        return None

    try:
        warm_df = pd.read_csv(RESULTS_CSV)
    except Exception:
        return None

    expected_t = list(model.T)
    if len(warm_df) != len(expected_t):
        return None
    if "t" in warm_df.columns:
        csv_t = [int(t) for t in warm_df["t"].tolist()]
        if csv_t != expected_t:
            return None

    # Only trust the saved time series if it matches the current warm-start
    # point on the main design and operating-point variables.
    reference_pairs = (
        ("W_cap", float(pyo.value(model.W_cap))),
        ("B_cap", float(pyo.value(model.B_cap))),
        ("H_cap", float(pyo.value(model.H_cap))),
        ("C_cap", float(pyo.value(model.C_cap))),
        ("NH3_cap", float(pyo.value(model.NH3_cap))),
        ("HX_cap", float(pyo.value(model.HX_cap))),
        ("Ft_op", float(pyo.value(model.Ft_op))),
        ("Fh2_op", float(pyo.value(model.Fh2_op))),
        ("Fnh3_op", float(pyo.value(model.Fnh3_op))),
        ("Fco2_op", float(pyo.value(model.Fco2_op))),
    )
    first_row = warm_df.iloc[0]
    for column_name, current_value in reference_pairs:
        if column_name not in warm_df.columns:
            return None
        csv_value = float(first_row[column_name])
        if not math.isclose(csv_value, current_value, rel_tol=1e-6, abs_tol=1e-4):
            return None

    if "cf_wind" in warm_df.columns:
        csv_cf = warm_df["cf_wind"].astype(float).to_numpy()
        model_cf = np.array([float(pyo.value(model.CF[t])) for t in expected_t], dtype=float)
        if csv_cf.shape != model_cf.shape or not np.allclose(csv_cf, model_cf, rtol=1e-9, atol=1e-9):
            return None

    tol = 1e-8
    if {"ch_H", "dis_H"}.issubset(warm_df.columns):
        if np.any(
            (warm_df["ch_H"].astype(float).to_numpy() > tol)
            & (warm_df["dis_H"].astype(float).to_numpy() > tol)
        ):
            return None
    if {"ch_NH3", "dis_NH3"}.issubset(warm_df.columns):
        if np.any(
            (warm_df["ch_NH3"].astype(float).to_numpy() > tol)
            & (warm_df["dis_NH3"].astype(float).to_numpy() > tol)
        ):
            return None

    return warm_df


def _apply_results_csv_warm_start(
    model: pyo.ConcreteModel,
    seeded_var_ids: set[int],
) -> int:
    warm_df = _load_results_csv_for_warmstart(model)
    if warm_df is None:
        return 0

    seeded = 0
    for column_name in warm_df.columns:
        if not hasattr(model, column_name):
            continue
        component = getattr(model, column_name)
        if getattr(component, "ctype", None) is not pyo.Var:
            continue

        if component.is_indexed():
            for row in warm_df.itertuples(index=False):
                t = int(getattr(row, "t"))
                if t not in component:
                    continue
                value = getattr(row, column_name)
                if pd.isna(value):
                    continue
                _set_warmstart_value(
                    component[t],
                    float(value),
                    f"{column_name}[{t}]",
                    seeded_var_ids,
                )
                seeded += 1
        else:
            value = warm_df.iloc[0][column_name]
            if pd.isna(value):
                continue
            _set_warmstart_value(
                component,
                float(value),
                column_name,
                seeded_var_ids,
            )
            seeded += 1

    # The saved CSV does not include the clean-power routing variables, but they
    # can be reconstructed from the exact hourly dispatch.
    for t in model.T:
        p_h_storage = alpha_H_ch * float(pyo.value(model.ch_H[t])) + alpha_H_dis * float(
            pyo.value(model.dis_H[t])
        )
        p_clean_el = float(pyo.value(model.P_el[t])) + p_h_storage
        p_clean_heat = float(pyo.value(model.P_HU_el[t]))
        p_wind = float(pyo.value(model.P_wind[t]))
        p_wind_to_batt = float(pyo.value(model.ch_B[t]))

        wind_remaining = max(p_wind - p_wind_to_batt, 0.0)
        p_wind_to_heat = min(p_clean_heat, wind_remaining)
        wind_remaining -= p_wind_to_heat
        p_wind_to_el = min(p_clean_el, wind_remaining)
        p_batt_to_heat = max(p_clean_heat - p_wind_to_heat, 0.0)
        p_batt_to_el = max(p_clean_el - p_wind_to_el, 0.0)
        bat_charge_mode = 1.0 if float(pyo.value(model.ch_B[t])) > 1e-8 else 0.0
        hydrogen_charge_mode = 1.0 if float(pyo.value(model.ch_H[t])) > 1e-8 else 0.0
        ammonia_charge_mode = 1.0 if float(pyo.value(model.ch_NH3[t])) > 1e-8 else 0.0

        for var_name, value in (
            ("P_wind_to_batt", p_wind_to_batt),
            ("P_wind_to_heat", p_wind_to_heat),
            ("P_wind_to_el", p_wind_to_el),
            ("P_batt_to_heat", p_batt_to_heat),
            ("P_batt_to_el", p_batt_to_el),
            ("bat_charge_mode", bat_charge_mode),
            ("hydrogen_charge_mode", hydrogen_charge_mode),
            ("ammonia_charge_mode", ammonia_charge_mode),
        ):
            _set_warmstart_value(
                getattr(model, var_name)[t],
                value,
                f"{var_name}[{t}]",
                seeded_var_ids,
            )
            seeded += 1

    return seeded


def _propagate_linear_equalities(block, seeded_var_ids: set[int]) -> int:
    """
    Fill hidden continuous warm-start values implied by linear equalities inside an OMLT block.

    This is especially helpful for OMLT blocks that introduce auxiliary variables
    beyond the visible surrogate features.
    """
    seeded = 0
    progress = True
    while progress:
        progress = False
        for con in block.component_data_objects(pyo.Constraint, active=True, descend_into=True):
            if not con.equality:
                continue

            repn = generate_standard_repn(con.body)
            if not repn.is_linear():
                continue

            rhs = float(pyo.value(con.lower))
            known_total = float(repn.constant or 0.0)
            unknowns = []
            for coef, var in zip(repn.linear_coefs, repn.linear_vars):
                is_known = var.fixed or id(var) in seeded_var_ids
                if is_known:
                    known_total += float(coef) * float(var.value)
                else:
                    unknowns.append((float(coef), var))

            if len(unknowns) != 1:
                continue

            coef, unknown_var = unknowns[0]
            if abs(coef) <= 1e-12:
                continue

            implied_value = (rhs - known_total) / coef
            _set_warmstart_value(unknown_var, implied_value, unknown_var.name, seeded_var_ids)
            seeded += 1
            progress = True

    return seeded


def apply_manual_warm_start(model: pyo.ConcreteModel) -> int:
    """
    Apply a partial warm start using the supplied design and operating-point values.

    This seeds the first-stage variables directly and propagates only the hourly values
    that are unambiguously implied by units fixed on/off. Hourly storage dispatch is left
    unset unless it is structurally zero so we do not introduce an inconsistent start.
    Any seeded value outside the active variable bounds is clipped back into bounds.
    """
    if not USE_MANUAL_WARMSTART or not MANUAL_WARMSTART_VALUES:
        return 0

    seeded = 0
    seeded_var_ids: set[int] = set()
    for name, value in MANUAL_WARMSTART_VALUES.items():
        if not hasattr(model, name):
            continue
        _set_warmstart_value(getattr(model, name), float(value), name, seeded_var_ids)
        seeded += 1

    seeded_from_results_csv = _apply_results_csv_warm_start(model, seeded_var_ids)
    seeded += seeded_from_results_csv

    ammonia_tree_inputs = [float(pyo.value(getattr(model, name))) for name in ("Ft_op", "Fh2_op")]
    for output_name, var_name in (
        ("ammonia_kgph", "F_NH3_A_op"),
        ("water_kgph", "F_H2O_A_op"),
        ("electric_kwhph", "E_A_op"),
    ):
        exact_output_value = float(
            _evaluate_network_output(
                model._ammonia_network_definitions[output_name],
                ammonia_tree_inputs,
            )[0]
        )
        _set_warmstart_value(
            getattr(model, var_name),
            exact_output_value,
            var_name,
            seeded_var_ids,
        )
        seeded += _seed_tree_block(
            model.ammonia_tree[output_name],
            ammonia_tree_inputs,
            exact_output_value,
            f"ammonia_tree[{output_name}]",
            seeded_var_ids,
        )
        seeded += _seed_relu_network_internal_state(
            model.ammonia_tree[output_name],
            model._ammonia_network_definitions[str(output_name)],
            ammonia_tree_inputs,
            f"ammonia_tree[{output_name}]",
            seeded_var_ids,
        )
        seeded += _propagate_linear_equalities(model.ammonia_tree[output_name], seeded_var_ids)

    urea_tree_inputs = [float(pyo.value(getattr(model, name))) for name in ("Fnh3_op", "Fco2_op")]
    for output_name, var_name in (
        ("pure_urea_kgph", "F_U_op"),
        ("product_urea_wtfrac", "w_U_op"),
        ("electric_kwhph", "E_U_op"),
    ):
        exact_output_value = float(
            _evaluate_network_output(
                model._urea_network_definitions[output_name],
                urea_tree_inputs,
            )[0]
        )
        _set_warmstart_value(
            getattr(model, var_name),
            exact_output_value,
            var_name,
            seeded_var_ids,
        )
        seeded += _seed_tree_block(
            model.urea_tree[output_name],
            urea_tree_inputs,
            exact_output_value,
            f"urea_tree[{output_name}]",
            seeded_var_ids,
        )
        seeded += _seed_relu_network_internal_state(
            model.urea_tree[output_name],
            model._urea_network_definitions[str(output_name)],
            urea_tree_inputs,
            f"urea_tree[{output_name}]",
            seeded_var_ids,
        )
        seeded += _propagate_linear_equalities(model.urea_tree[output_name], seeded_var_ids)

    _synchronize_design_basis_warmstart(model, seeded_var_ids)

    if seeded_from_results_csv:
        return seeded

    ammonia_start_state = _infer_unit_commitment_start_state(
        model.y_A_on,
        ("Ft_op", "Fh2_op", "F_NH3_A_op", "F_H2O_A_op", "E_A_op"),
    )
    if not all(var.fixed for var in model.y_A_on.values()):
        for t in model.T:
            if model.y_A_on[t].fixed:
                continue
            _set_warmstart_value(
                model.y_A_on[t],
                float(ammonia_start_state),
                f"y_A_on[{t}]",
                seeded_var_ids,
            )
            seeded += 1

    urea_start_state = _infer_unit_commitment_start_state(
        model.y_U_on,
        ("Fnh3_op", "Fco2_op", "F_U_op", "w_U_op", "E_U_op"),
    )
    if not all(var.fixed for var in model.y_U_on.values()):
        for t in model.T:
            if model.y_U_on[t].fixed:
                continue
            _set_warmstart_value(
                model.y_U_on[t],
                float(urea_start_state),
                f"y_U_on[{t}]",
                seeded_var_ids,
            )
            seeded += 1

    battery_zero = abs(MANUAL_WARMSTART_VALUES.get("B_cap", 0.0)) <= 1e-8
    co2_storage_zero = abs(MANUAL_WARMSTART_VALUES.get("C_cap", 0.0)) <= 1e-8

    if battery_zero:
        for t in model.T:
            for var_name, value in (
                ("I_B", 0.0),
                ("ch_B", 0.0),
                ("dis_B", 0.0),
                ("P_wind_to_batt", 0.0),
                ("P_batt_to_el", 0.0),
                ("P_batt_to_heat", 0.0),
                ("bat_charge_mode", 0.0),
            ):
                _set_warmstart_value(getattr(model, var_name)[t], value, f"{var_name}[{t}]", seeded_var_ids)
                seeded += 1

    for t in model.T:
        for var_name in ("hydrogen_charge_mode", "ammonia_charge_mode"):
            _set_warmstart_value(getattr(model, var_name)[t], 0.0, f"{var_name}[{t}]", seeded_var_ids)
            seeded += 1

    if urea_start_state == 1:
        for t in model.T:
            for var_name, op_name in (
                ("Fnh3", "Fnh3_op"),
                ("Fco2", "Fco2_op"),
                ("F_U", "F_U_op"),
                ("w_U", "w_U_op"),
                ("E_U", "E_U_op"),
            ):
                value = float(pyo.value(getattr(model, op_name)))
                _set_warmstart_value(getattr(model, var_name)[t], value, f"{var_name}[{t}]", seeded_var_ids)
                seeded += 1
            if co2_storage_zero:
                fco2 = float(pyo.value(model.Fco2[t]))
                spill = max(float(pyo.value(model.Fbar_C[t])) - fco2, 0.0)
                _set_warmstart_value(model.I_C[t], 0.0, f"I_C[{t}]", seeded_var_ids)
                _set_warmstart_value(model.Fco2_spill[t], spill, f"Fco2_spill[{t}]", seeded_var_ids)
                seeded += 2
    else:
        for t in model.T:
            for var_name in ("Fnh3", "Fco2", "F_U", "w_U", "E_U"):
                _set_warmstart_value(getattr(model, var_name)[t], 0.0, f"{var_name}[{t}]", seeded_var_ids)
                seeded += 1
            if co2_storage_zero:
                _set_warmstart_value(model.I_C[t], 0.0, f"I_C[{t}]", seeded_var_ids)
                _set_warmstart_value(
                    model.Fco2_spill[t],
                    float(pyo.value(model.Fbar_C[t])),
                    f"Fco2_spill[{t}]",
                    seeded_var_ids,
                )
                seeded += 2

    if ammonia_start_state == 1:
        for t in model.T:
            for var_name, op_name in (
                ("Ft", "Ft_op"),
                ("Fh2", "Fh2_op"),
                ("F_NH3_A", "F_NH3_A_op"),
                ("F_H2O_A", "F_H2O_A_op"),
                ("E_A", "E_A_op"),
            ):
                value = float(pyo.value(getattr(model, op_name)))
                _set_warmstart_value(getattr(model, var_name)[t], value, f"{var_name}[{t}]", seeded_var_ids)
                seeded += 1
            q_s = stripper_duty_kwh_per_kg_ammonia * float(pyo.value(model.F_NH3_A[t]))
            q_rec = min(q_s, float(pyo.value(model.HX_cap)))
            q_hu = q_s - q_rec
            p_hu_el = q_hu / eta_eheater
            ft_spill = max(float(pyo.value(model.Fbar_Ft[t])) - float(pyo.value(model.Ft[t])), 0.0)
            _set_warmstart_value(model.F_NH3_strip[t], float(pyo.value(model.F_NH3_A[t])), f"F_NH3_strip[{t}]", seeded_var_ids)
            _set_warmstart_value(model.Q_S[t], q_s, f"Q_S[{t}]", seeded_var_ids)
            _set_warmstart_value(model.Q_rec[t], q_rec, f"Q_rec[{t}]", seeded_var_ids)
            _set_warmstart_value(model.Q_HU[t], q_hu, f"Q_HU[{t}]", seeded_var_ids)
            _set_warmstart_value(model.P_HU_el[t], p_hu_el, f"P_HU_el[{t}]", seeded_var_ids)
            _set_warmstart_value(model.Ft_spill[t], ft_spill, f"Ft_spill[{t}]", seeded_var_ids)
            seeded += 6
    else:
        for t in model.T:
            for var_name in ("Ft", "Fh2", "F_NH3_A", "F_H2O_A", "E_A", "F_NH3_strip", "Q_S", "Q_rec", "Q_HU", "P_HU_el"):
                _set_warmstart_value(getattr(model, var_name)[t], 0.0, f"{var_name}[{t}]", seeded_var_ids)
                seeded += 1
            ft_spill = float(pyo.value(model.Fbar_Ft[t]))
            _set_warmstart_value(model.Ft_spill[t], ft_spill, f"Ft_spill[{t}]", seeded_var_ids)
            seeded += 1

    return seeded


def needs_iis(results) -> bool:
    termination = results.solver.termination_condition
    return termination in {
        pyo.TerminationCondition.infeasible,
        pyo.TerminationCondition.infeasibleOrUnbounded,
    }


def _normalize_iis_solver_name(solver_name: str) -> str:
    normalized = str(solver_name).lower().strip()
    for suffix in ("_persistent", "_direct"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def generate_named_iis(
    model: pyo.ConcreteModel,
    output_dir: Path | str = THIS_DIR,
    basename: str = "infeasible_model",
    solver_name: str = SOLVER_NAME,
) -> dict[str, Path]:
    """
    Generate an IIS and a readable summary using the Pyomo component names
    from model construction.

    For Gurobi, this avoids Pyomo's symbolic-label duplicate-name issue by:
    1. writing a non-symbolic LP,
    2. keeping the writer's symbol map,
    3. reading the LP in gurobipy,
    4. translating IIS members back to Pyomo names in the summary.
    """
    solver_family = _normalize_iis_solver_name(solver_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lp_path = output_dir / f"{basename}_{timestamp}.lp"
    ilp_path = output_dir / f"{basename}_{timestamp}.ilp"
    txt_path = output_dir / f"{basename}_{timestamp}_summary.txt"

    if solver_family != "gurobi":
        from pyomo.contrib.iis import write_iis

        write_iis(model, str(ilp_path), solver=solver_family)
        txt_path.write_text(
            "IIS was generated with Pyomo's generic IIS writer.\n"
            f"Solver family: {solver_family}\n"
            f"IIS file: {ilp_path}\n",
            encoding="utf-8",
        )
        return {"ilp": ilp_path, "summary": txt_path}

    import gurobipy as gp
    from gurobipy import GRB

    if lp_path.exists():
        lp_path.unlink()

    _, smap_id = model.write(str(lp_path))
    symbol_map = model.solutions.symbol_map[smap_id]
    solver_to_pyomo = {
        solver_name: obj
        for solver_name, obj in symbol_map.bySymbol.items()
    }

    grb_model = gp.read(str(lp_path))
    grb_model.Params.OutputFlag = 0
    grb_model.optimize()
    if grb_model.Status == GRB.INF_OR_UNBD:
        grb_model.Params.DualReductions = 0
        grb_model.optimize()
    if grb_model.Status != GRB.INFEASIBLE:
        raise RuntimeError(
            f"Gurobi IIS requires an infeasible model, but LP solve status was {grb_model.Status}."
        )

    grb_model.computeIIS()
    grb_model.write(str(ilp_path))

    iis_constraint_names = [con.ConstrName for con in grb_model.getConstrs() if con.IISConstr]
    iis_var_lb_names = [var.VarName for var in grb_model.getVars() if var.IISLB]
    iis_var_ub_names = [var.VarName for var in grb_model.getVars() if var.IISUB]

    summary_lines = [
        "Named IIS summary",
        f"Generated: {timestamp}",
        f"Solver family: {solver_family}",
        f"LP file: {lp_path}",
        f"IIS file: {ilp_path}",
        "",
        f"Constraints in IIS: {len(iis_constraint_names)}",
    ]

    for raw_name in sorted(iis_constraint_names):
        con = solver_to_pyomo.get(raw_name)
        if con is None:
            summary_lines.append(f"  - {raw_name}")
        else:
            summary_lines.append(f"  - {con.name} [solver: {raw_name}]: {con.expr}")

    summary_lines.append("")
    summary_lines.append(f"Variable lower bounds in IIS: {len(iis_var_lb_names)}")
    for raw_name in sorted(iis_var_lb_names):
        var = solver_to_pyomo.get(raw_name)
        if var is None:
            summary_lines.append(f"  - {raw_name}: lower bound in IIS")
        else:
            summary_lines.append(f"  - {var.name} [solver: {raw_name}]: lb = {var.lb}")

    summary_lines.append("")
    summary_lines.append(f"Variable upper bounds in IIS: {len(iis_var_ub_names)}")
    for raw_name in sorted(iis_var_ub_names):
        var = solver_to_pyomo.get(raw_name)
        if var is None:
            summary_lines.append(f"  - {raw_name}: upper bound in IIS")
        else:
            summary_lines.append(f"  - {var.name} [solver: {raw_name}]: ub = {var.ub}")

    txt_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return {"lp": lp_path, "ilp": ilp_path, "summary": txt_path}


def maybe_generate_iis(model: pyo.ConcreteModel, results) -> None:
    if not AUTO_WRITE_IIS or not needs_iis(results):
        return

    try:
        paths = generate_named_iis(model)
        print("\n=== IIS ===")
        if "lp" in paths:
            print(f"LP written to {paths['lp']}")
        print(f"IIS written to {paths['ilp']}")
        print(f"Readable summary written to {paths['summary']}")
    except Exception as exc:
        print(f"\nIIS generation failed: {type(exc).__name__}: {exc}")


def maybe_write_lp(model: pyo.ConcreteModel) -> None:
    if not WRITE_LP_DEBUG:
        return

    lp_path = THIS_DIR / "model_debug.lp"
    try:
        if lp_path.exists():
            lp_path.unlink()
        model.write(str(lp_path))
        print(f"Successfully wrote {lp_path}")
    except Exception as exc:
        print(f"LP write failed: {type(exc).__name__}: {exc}")


def print_solution_summary(model: pyo.ConcreteModel, results) -> None:
    print("\n=== Solve status ===")
    print(results.solver.status)
    print(results.solver.termination_condition)

    if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        print("Model is infeasible. No solution summary is available.")
        return
    if not has_feasible_solution(results):
        print("Solver returned without a feasible incumbent solution.")
        return

    print("\n=== Design variables ===")
    for name in [
        "W_cap",
        "E_cap",
        "E_cap_incremental",
        "E_cap_included_in_nh3",
        "B_cap",
        "H_cap",
        "C_cap",
        "NH3_cap",
        "HX_cap",
    ]:
        print(f"{name:12s} = {pyo.value(getattr(model, name)):.6f}")

    print("\n=== Fixed Geometry ===")
    print(
        "Ammonia reactor = "
        f"{pyo.value(model.ammonia_reactor_length_m):.3f} m length, "
        f"{pyo.value(model.ammonia_reactor_diameter_m):.3f} m diameter, "
        f"{pyo.value(model.ammonia_reactor_volume_m3):.6f} m^3 volume"
    )
    print(
        "Urea reactor    = "
        f"{pyo.value(model.urea_reactor_length_m):.3f} m length, "
        f"{pyo.value(model.urea_reactor_diameter_m):.3f} m diameter, "
        f"{pyo.value(model.urea_reactor_volume_m3):.6f} m^3 volume"
    )

    print("\n=== Operating points ===")
    for name in [
        "Ft_op",
        "Fh2_op",
        "Fnh3_op",
        "Fco2_op",
        "F_NH3_A_op",
        "F_H2O_A_op",
        "E_A_op",
        "F_U_op",
        "w_U_op",
        "E_U_op",
    ]:
        print(f"{name:12s} = {pyo.value(getattr(model, name)):.6f}")

    print("\n=== Objective ===")
    print(f"NH3 plant capacity (t/y) = {pyo.value(model.K_nh3_plant_tpy):.6f}")
    print(f"Urea plant capacity (t/y) = {pyo.value(model.K_urea_plant_tpy):.6f}")
    print(f"Electrolyzer capacity included in NH3 CAPEX = {pyo.value(model.E_cap_included_in_nh3):.6f}")
    print(f"NH3 plant CAPEX basis = {pyo.value(model.nh3_plant_capex):.6f}")
    print(f"Urea plant CAPEX basis = {pyo.value(model.urea_plant_capex):.6f}")
    print(f"Total annualized cost = {pyo.value(model.obj):.6f}")


def export_results(model: pyo.ConcreteModel, results) -> None:
    if (
        results.solver.termination_condition == pyo.TerminationCondition.infeasible
        or not has_feasible_solution(results)
    ):
        print("Skipping result export because no feasible solution is available.")
        return

    rows = []
    for t in model.T:
        rows.append(
            {
                "t": int(t),
                "cf_wind": pyo.value(model.CF[t]),
                "y_A_on": pyo.value(model.y_A_on[t]),
                "y_U_on": pyo.value(model.y_U_on[t]),
                "P_wind": pyo.value(model.P_wind[t]),
                "P_grid": pyo.value(model.P_grid[t]),
                "P_el": pyo.value(model.P_el[t]),
                "M_H2_prod": pyo.value(model.M_H2_prod[t]),
                "M_O2_prod": pyo.value(model.M_O2_prod[t]),
                "M_H2O_el": pyo.value(model.M_H2O_el[t]),
                "total_electric_load": pyo.value(model.total_electric_load[t]),
                "ch_B": pyo.value(model.ch_B[t]),
                "dis_B": pyo.value(model.dis_B[t]),
                "I_B": pyo.value(model.I_B[t]),
                "hydrogen_charge_mode": pyo.value(model.hydrogen_charge_mode[t]),
                "ch_H": pyo.value(model.ch_H[t]),
                "dis_H": pyo.value(model.dis_H[t]),
                "I_H": pyo.value(model.I_H[t]),
                "I_C": pyo.value(model.I_C[t]),
                "I_NH3": pyo.value(model.I_NH3[t]),
                "Ft": pyo.value(model.Ft[t]),
                "Fh2": pyo.value(model.Fh2[t]),
                "Fnh3": pyo.value(model.Fnh3[t]),
                "Fco2": pyo.value(model.Fco2[t]),
                "F_NH3_A": pyo.value(model.F_NH3_A[t]),
                "F_H2O_A": pyo.value(model.F_H2O_A[t]),
                "F_NH3_strip": pyo.value(model.F_NH3_strip[t]),
                "Q_S": pyo.value(model.Q_S[t]),
                "Q_rec": pyo.value(model.Q_rec[t]),
                "Q_HU": pyo.value(model.Q_HU[t]),
                "P_HU_el": pyo.value(model.P_HU_el[t]),
                "F_U": pyo.value(model.F_U[t]),
                "w_U": pyo.value(model.w_U[t]),
                "E_A": pyo.value(model.E_A[t]),
                "E_U": pyo.value(model.E_U[t]),
                "Ft_spill": pyo.value(model.Ft_spill[t]),
                "Fco2_spill": pyo.value(model.Fco2_spill[t]),
                "ammonia_charge_mode": pyo.value(model.ammonia_charge_mode[t]),
                "ch_NH3": pyo.value(model.ch_NH3[t]),
                "dis_NH3": pyo.value(model.dis_NH3[t]),
                "W_cap": pyo.value(model.W_cap),
                "E_cap": pyo.value(model.E_cap),
                "E_cap_incremental": pyo.value(model.E_cap_incremental),
                "B_cap": pyo.value(model.B_cap),
                "H_cap": pyo.value(model.H_cap),
                "C_cap": pyo.value(model.C_cap),
                "NH3_cap": pyo.value(model.NH3_cap),
                "HX_cap": pyo.value(model.HX_cap),
                "K_nh3_plant_tpy": pyo.value(model.K_nh3_plant_tpy),
                "K_urea_plant_tpy": pyo.value(model.K_urea_plant_tpy),
                "E_cap_included_in_nh3": pyo.value(model.E_cap_included_in_nh3),
                "nh3_plant_capex": pyo.value(model.nh3_plant_capex),
                "urea_plant_capex": pyo.value(model.urea_plant_capex),
                "ammonia_reactor_length_m": pyo.value(model.ammonia_reactor_length_m),
                "ammonia_reactor_diameter_m": pyo.value(model.ammonia_reactor_diameter_m),
                "ammonia_reactor_volume_m3": pyo.value(model.ammonia_reactor_volume_m3),
                "urea_reactor_length_m": pyo.value(model.urea_reactor_length_m),
                "urea_reactor_diameter_m": pyo.value(model.urea_reactor_diameter_m),
                "urea_reactor_volume_m3": pyo.value(model.urea_reactor_volume_m3),
                "Ft_op": pyo.value(model.Ft_op),
                "Fh2_op": pyo.value(model.Fh2_op),
                "Fnh3_op": pyo.value(model.Fnh3_op),
                "Fco2_op": pyo.value(model.Fco2_op),
                "F_NH3_A_op": pyo.value(model.F_NH3_A_op),
                "F_H2O_A_op": pyo.value(model.F_H2O_A_op),
                "E_A_op": pyo.value(model.E_A_op),
                "F_U_op": pyo.value(model.F_U_op),
                "w_U_op": pyo.value(model.w_U_op),
                "E_U_op": pyo.value(model.E_U_op),
            }
        )

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"Saved results to {RESULTS_CSV}")


def main() -> None:
    surrogates = load_surrogates()
    print("\nLoaded trained surrogate bundles:")
    for unit_name, surrogate in surrogates.items():
        print(f"[{unit_name}]")
        print(f"  features : {list(surrogate.bundle.feature_names)}")
        print(f"  outputs  : {list(surrogate.bundle.output_names)}")
    report_energy_unit_audit(surrogates)

    wind_df = read_wind_data(WIND_CSV, N_PERIODS)
    effective_days = int(wind_df.attrs.get("effective_days", len(wind_df) // 24))
    requested_days = int(wind_df.attrs.get("requested_days", effective_days))
    print(f"Planning horizon: {effective_days} day(s)")
    print(f"Horizon mode: {HORIZON_MODE}")
    if HORIZON_MODE == "random_3day_to_1day":
        print(
            f"Requested source days: {requested_days}; "
            f"sampled one day from each {RANDOM_DAY_REDUCTION_FACTOR}-day block "
            f"using seed {RANDOM_DAY_REDUCTION_SEED}"
        )
    print(f"Grid mode: {GRID_MODE}")
    print(f"Oxygen revenue enabled: {ENABLE_OXYGEN_REVENUE}")

    model = build_model(surrogates, wind_df)
    maybe_write_lp(model)
    results = solve_model(model)
    maybe_generate_iis(model, results)
    print_solution_summary(model, results)
    export_results(model, results)


if __name__ == "__main__":
    main()

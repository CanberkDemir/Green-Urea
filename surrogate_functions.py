"""
surrogate_functions.py

Data-driven ammoniaF and ureaF surrogate models with a stable Python API.

This module preserves the training/loading/prediction workflow expected by the
rest of the project, but replaces the previous HyperplaneTree back-end with
small TensorFlow/Keras ReLU neural networks. Each surrogate output is trained
as its own `tf.keras.Sequential` model so it can later be wrapped by reluMIP's
`AnnModel` interface without changing the calling code here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import math
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    tf.keras.utils.disable_interactive_logging()
except Exception:
    pass
# =============================================================================
# CONFIG
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_DIR = THIS_DIR / "trained_unit_surrogates"
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

AMMONIAF_RESULTS_CSV = THIS_DIR / "ammoniaF_results_live.csv"
AMMONIAF_CASE_GRID_CSV = THIS_DIR / "ammoniaF_case_grid.csv"
AMMONIAF_INPUTS_CSV = THIS_DIR / "ammoniaF_inputs.csv"

UREAF_RESULTS_CSV = THIS_DIR / "ureaF_results_live.csv"
UREAF_INPUTS_CSV = THIS_DIR / "ureaF_inputs.csv"
UREAF_TRAIN_ALL_HEAT_DUTIES = True

SHOW_INTERACTIVE_3D_PLOTS = False
ISOMETRIC_ELEVATION_DEG = 35.264389682754654
ISOMETRIC_AZIMUTH_DEG = -135

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
    "Fco2": (3.0, 50.0),
}
UNIT_INPUT_BOUND_OVERRIDES = {
    "ammoniaF_unit": AMMONIAF_OPERATING_INPUT_BOUNDS,
    "ureaF_unit": UREAF_OPERATING_INPUT_BOUNDS,
}

ANN_HIDDEN_LAYER_SIZES = (4,3)
ANN_LEARNING_RATE = 2e-3
ANN_MAX_EPOCHS = 400
ANN_PATIENCE = 10
ANN_VALIDATION_SPLIT = 0.15
ANN_BATCH_SIZE = 256
ANN_MIN_DELTA = 1e-5
ANN_L2_REG = 1e-6
MIN_SCALE = 1e-8

# Edit these per-target entries when you want different layer sizes or epoch
# counts for specific outputs within each unit.
ANN_UNIT_TARGET_TRAINING_CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "ammoniaF_unit": {
        "ammonia_kgph": {
            "hidden_layer_sizes": ANN_HIDDEN_LAYER_SIZES,
            "max_epochs": ANN_MAX_EPOCHS,
        },
        "water_kgph": {
            "hidden_layer_sizes": ANN_HIDDEN_LAYER_SIZES,
            "max_epochs": ANN_MAX_EPOCHS,
        },
        "electric_kwhph": {
            "hidden_layer_sizes": ANN_HIDDEN_LAYER_SIZES,
            "max_epochs": ANN_MAX_EPOCHS,
        },
    },
    "ureaF_unit": {
        "pure_urea_kgph": {
            "hidden_layer_sizes": ANN_HIDDEN_LAYER_SIZES,
            "max_epochs": ANN_MAX_EPOCHS,
        },
        "product_urea_wtfrac": {
            "hidden_layer_sizes": ANN_HIDDEN_LAYER_SIZES,
            "max_epochs": ANN_MAX_EPOCHS,
        },
        "electric_kwhph": {
            "hidden_layer_sizes": ANN_HIDDEN_LAYER_SIZES,
            "max_epochs": ANN_MAX_EPOCHS,
        },
    },
}
UNIT_NAME_ALIASES = {
    "nitrate_unit": "ammoniaF_unit",
    "urea_unit": "ureaF_unit",
}

_MODEL_CACHE: Dict[str, "UnitSurrogateBundle"] = {}

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class UnitSurrogateBundle:
    unit_name: str
    feature_names: List[str]
    output_names: List[str]
    models: Dict[str, tf.keras.Sequential]
    input_bounds: Dict[str, Tuple[float, float]]
    output_bounds: Dict[str, Tuple[float, float]]
    model_metrics: Dict[str, Dict[str, Any]] | None = None
    training_config: Dict[str, Any] | None = None
    model_kind: str = "relu_ann"

    def predict(self, x: Dict[str, float]) -> Dict[str, float]:
        x_vec = np.array(
            [[float(x[name]) for name in self.feature_names]],
            dtype=np.float32,
        )
        out: Dict[str, float] = {}
        for y_name, model in self.models.items():
            y_pred = _keras_predict(model, x_vec).reshape(-1)[0]
            out[y_name] = float(y_pred)
        return out

    def make_relumip_model(
        self,
        output_name: str,
        modeling_language: str = "PYOMO",
        name: str | None = None,
    ):
        from relumip import AnnModel

        if output_name not in self.models:
            raise KeyError(
                f"{output_name!r} is not an output of {self.unit_name}. "
                f"Available outputs: {list(self.models)}"
            )
        return AnnModel(
            self.models[output_name],
            modeling_language=str(modeling_language).upper(),
            name=name or f"{self.unit_name}_{output_name}",
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        serialized_models: Dict[str, Any] = {}
        for output_name, model in self.models.items():
            if isinstance(model, tf.keras.Sequential):
                serialized_models[output_name] = {
                    "config_json": model.to_json(),
                    "weights": [np.asarray(w, dtype=np.float32) for w in model.get_weights()],
                }
            else:
                serialized_models[output_name] = model
        state["models"] = serialized_models
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        raw_models = state.get("models", {})
        rebuilt_models: Dict[str, tf.keras.Sequential] = {}
        for output_name, payload in raw_models.items():
            if (
                isinstance(payload, dict)
                and "config_json" in payload
                and "weights" in payload
            ):
                model = tf.keras.models.model_from_json(payload["config_json"])
                model.set_weights(payload["weights"])
                rebuilt_models[output_name] = model
            else:
                rebuilt_models[output_name] = payload
        state["models"] = rebuilt_models
        self.__dict__.update(state)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def _clip(values, lower, upper):
    return np.minimum(np.maximum(values, lower), upper)


def _parse_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "t"}


def _to_numpy_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    if isinstance(values, pd.DataFrame):
        values = values.to_numpy()
    elif isinstance(values, pd.Series):
        values = values.to_numpy()
    return np.asarray(values)


def _keras_predict(model: tf.keras.Sequential, values) -> np.ndarray:
    inputs = _to_numpy_array(values).astype(np.float32, copy=False)
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)
    return np.asarray(model.predict(inputs, verbose=0), dtype=np.float32)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_or_raise(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing trained surrogate file: {path}\n"
            "Run train_and_save_all_surrogates(...) first."
        )
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "UnitSurrogateBundle"):
        setattr(main_module, "UnitSurrogateBundle", UnitSurrogateBundle)
    return joblib.load(path)


def _nondegenerate_bounds(values) -> Tuple[float, float]:
    values_np = _to_numpy_array(values).astype(float)
    lb = float(np.nanmin(values_np))
    ub = float(np.nanmax(values_np))
    if not np.isfinite(lb) or not np.isfinite(ub):
        raise ValueError("Encountered non-finite values while computing bounds.")
    if lb >= ub:
        width = max(abs(lb), 1.0) * 1e-6
        lb -= width
        ub += width
    return lb, ub


def _standardize_matrix(values) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_np = _to_numpy_array(values).astype(np.float32)
    mean = values_np.mean(axis=0, keepdims=False).astype(np.float32)
    std = values_np.std(axis=0, keepdims=False).astype(np.float32)
    std = np.where(np.abs(std) < MIN_SCALE, 1.0, std).astype(np.float32)
    scaled = ((values_np - mean) / std).astype(np.float32)
    return scaled, mean, std


def _canonical_unit_name(unit_name: str) -> str:
    return UNIT_NAME_ALIASES.get(unit_name, unit_name)


def _normalize_hidden_layer_sizes(hidden_layer_sizes: Any) -> Tuple[int, ...]:
    if isinstance(hidden_layer_sizes, (int, np.integer)):
        sizes = (int(hidden_layer_sizes),)
    else:
        sizes = tuple(int(size) for size in hidden_layer_sizes)
    if not sizes:
        raise ValueError("`hidden_layer_sizes` must contain at least one positive layer width.")
    if any(size <= 0 for size in sizes):
        raise ValueError("Every hidden layer width must be a positive integer.")
    return sizes


def _normalize_output_training_config(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config = dict(config or {})
    config.pop("mode", None)
    hidden_layer_sizes = config.pop(
        "hidden_layer_sizes",
        config.pop("layers", ANN_HIDDEN_LAYER_SIZES),
    )
    max_epochs = config.pop(
        "max_epochs",
        config.pop("epochs", ANN_MAX_EPOCHS),
    )
    learning_rate = config.pop("learning_rate", ANN_LEARNING_RATE)
    batch_size = config.pop("batch_size", ANN_BATCH_SIZE)
    patience = config.pop("patience", ANN_PATIENCE)
    min_delta = config.pop("min_delta", ANN_MIN_DELTA)
    validation_split = config.pop("validation_split", ANN_VALIDATION_SPLIT)
    l2_reg = config.pop("l2_reg", ANN_L2_REG)
    seed = config.pop("seed", None)

    normalized = dict(config)
    normalized["hidden_layer_sizes"] = _normalize_hidden_layer_sizes(hidden_layer_sizes)
    normalized["max_epochs"] = int(max_epochs)
    if normalized["max_epochs"] <= 0:
        raise ValueError("`max_epochs` must be a positive integer.")
    normalized["learning_rate"] = float(learning_rate)
    if normalized["learning_rate"] <= 0.0:
        raise ValueError("`learning_rate` must be positive.")
    normalized["batch_size"] = int(batch_size)
    if normalized["batch_size"] <= 0:
        raise ValueError("`batch_size` must be a positive integer.")
    normalized["patience"] = int(patience)
    if normalized["patience"] < 0:
        raise ValueError("`patience` must be non-negative.")
    normalized["min_delta"] = float(min_delta)
    if normalized["min_delta"] < 0.0:
        raise ValueError("`min_delta` must be non-negative.")
    normalized["validation_split"] = float(validation_split)
    if not 0.0 <= normalized["validation_split"] < 1.0:
        raise ValueError("`validation_split` must be in [0, 1).")
    normalized["l2_reg"] = float(l2_reg)
    if normalized["l2_reg"] < 0.0:
        raise ValueError("`l2_reg` must be non-negative.")
    if seed is not None:
        normalized["seed"] = int(seed)
    return normalized


def _canonicalize_output_training_config_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    canonical = dict(config)
    if "hidden_layer_sizes" not in canonical and "layers" in canonical:
        canonical["hidden_layer_sizes"] = canonical.pop("layers")
    else:
        canonical.pop("layers", None)
    if "max_epochs" not in canonical and "epochs" in canonical:
        canonical["max_epochs"] = canonical.pop("epochs")
    else:
        canonical.pop("epochs", None)
    return canonical


def _merge_output_training_config(*configs: Dict[str, Any] | None) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for config in configs:
        if config:
            merged.update(_canonicalize_output_training_config_keys(config))
    return _normalize_output_training_config(merged)


def _resolve_output_training_configs(
    unit_name: str,
    output_names: List[str],
    output_training_overrides: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    canonical_unit_name = _canonical_unit_name(unit_name)
    base_configs = ANN_UNIT_TARGET_TRAINING_CONFIGS.get(canonical_unit_name, {})
    overrides = output_training_overrides or {}
    unknown_outputs = sorted(set(overrides) - set(output_names))
    if unknown_outputs:
        raise KeyError(
            f"Unknown training-config outputs for {canonical_unit_name}: {unknown_outputs}. "
            f"Available outputs: {list(output_names)}"
        )

    resolved: Dict[str, Dict[str, Any]] = {}
    for output_name in output_names:
        resolved[output_name] = _merge_output_training_config(
            base_configs.get(output_name),
            overrides.get(output_name),
        )
    return resolved


def _resolve_unit_training_overrides(
    unit_name: str,
    unit_training_overrides: Dict[str, Dict[str, Dict[str, Any]]] | None,
) -> Dict[str, Dict[str, Any]] | None:
    if not unit_training_overrides:
        return None

    canonical_unit_name = _canonical_unit_name(unit_name)
    merged: Dict[str, Dict[str, Any]] = {}

    for candidate_name, candidate_overrides in unit_training_overrides.items():
        if _canonical_unit_name(candidate_name) == canonical_unit_name:
            merged.update(candidate_overrides)

    return merged or None


def _serialize_output_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    serialized = dict(config)
    serialized["hidden_layer_sizes"] = list(config["hidden_layer_sizes"])
    serialized["max_epochs"] = int(config["max_epochs"])
    serialized["learning_rate"] = float(config["learning_rate"])
    serialized["batch_size"] = int(config["batch_size"])
    serialized["patience"] = int(config["patience"])
    serialized["min_delta"] = float(config["min_delta"])
    serialized["validation_split"] = float(config["validation_split"])
    serialized["l2_reg"] = float(config["l2_reg"])
    if "seed" in config:
        serialized["seed"] = int(config["seed"])
    return serialized


def _bundle_output_training_config(
    bundle: UnitSurrogateBundle,
    output_name: str,
) -> Dict[str, Any]:
    training_config = bundle.training_config or {}
    per_output_config = training_config.get("per_output", {})
    if isinstance(per_output_config, dict) and output_name in per_output_config:
        return _normalize_output_training_config(per_output_config[output_name])
    return _normalize_output_training_config(training_config)


def _build_relu_network(
    input_dim: int,
    hidden_layer_sizes: Tuple[int, ...] = ANN_HIDDEN_LAYER_SIZES,
    learning_rate: float = ANN_LEARNING_RATE,
    l2_reg: float = ANN_L2_REG,
) -> tf.keras.Sequential:
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    layers: List[tf.keras.layers.Layer] = [
        tf.keras.layers.Input(shape=(input_dim,), dtype=tf.float32, name="input"),
    ]
    for idx, units in enumerate(hidden_layer_sizes, start=1):
        layers.append(
            tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                kernel_regularizer=regularizer,
                name=f"hidden_{idx}",
            )
        )
    layers.append(tf.keras.layers.Dense(1, activation="linear", name="output"))
    model = tf.keras.Sequential(layers, name="relu_surrogate")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def _fold_standardization_into_model(
    model: tf.keras.Sequential,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tf.keras.Sequential:
    folded = tf.keras.models.clone_model(model)
    folded.set_weights([np.array(w, copy=True) for w in model.get_weights()])
    weights = [np.array(w, copy=True) for w in folded.get_weights()]

    first_kernel = weights[0]
    first_bias = weights[1]
    x_mean_vec = np.asarray(x_mean, dtype=np.float32).reshape(-1)
    x_std_vec = np.asarray(x_std, dtype=np.float32).reshape(-1)

    weights[0] = (first_kernel / x_std_vec[:, None]).astype(np.float32)
    weights[1] = (
        first_bias - np.matmul((x_mean_vec / x_std_vec), first_kernel)
    ).astype(np.float32)

    last_kernel = weights[-2]
    last_bias = weights[-1]
    y_mean_scalar = float(np.asarray(y_mean, dtype=np.float32).reshape(-1)[0])
    y_std_scalar = float(np.asarray(y_std, dtype=np.float32).reshape(-1)[0])

    weights[-2] = (last_kernel * y_std_scalar).astype(np.float32)
    weights[-1] = (last_bias * y_std_scalar + y_mean_scalar).astype(np.float32)

    folded.set_weights(weights)
    return folded


def _train_single_output_model(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    training_config: Dict[str, Any] | None = None,
) -> Tuple[tf.keras.Sequential, Dict[str, Any]]:
    x_values = X.to_numpy(dtype=np.float32)
    y_values = y.to_numpy(dtype=np.float32).reshape(-1, 1)

    x_scaled, x_mean, x_std = _standardize_matrix(x_values)
    y_scaled, y_mean, y_std = _standardize_matrix(y_values)
    resolved_training_config = _normalize_output_training_config(training_config)
    model_seed = int(resolved_training_config.get("seed", seed))

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(model_seed)

    model = _build_relu_network(
        input_dim=x_values.shape[1],
        hidden_layer_sizes=resolved_training_config["hidden_layer_sizes"],
        learning_rate=resolved_training_config["learning_rate"],
        l2_reg=resolved_training_config["l2_reg"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=resolved_training_config["patience"],
            min_delta=resolved_training_config["min_delta"],
            restore_best_weights=True,
        )
    ]

    fit_kwargs: Dict[str, Any] = {
        "x": x_scaled,
        "y": y_scaled,
        "epochs": resolved_training_config["max_epochs"],
        "batch_size": int(max(1, min(resolved_training_config["batch_size"], len(x_scaled)))),
        "shuffle": True,
        "verbose": 2,
        "callbacks": callbacks,
    }
    if len(x_scaled) >= 32 and resolved_training_config["validation_split"] > 0:
        fit_kwargs["validation_split"] = resolved_training_config["validation_split"]

    history = model.fit(**fit_kwargs)
    folded_model = _fold_standardization_into_model(model, x_mean, x_std, y_mean, y_std)

    y_pred = _keras_predict(folded_model, x_values).reshape(-1)
    y_true = y_values.reshape(-1)

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": float(
            np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0
        ),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "epochs_ran": float(len(history.history.get("loss", []))),
        "best_train_loss": float(np.min(history.history.get("loss", [np.nan]))),
        "train_loss_history": [
            float(loss_value) for loss_value in history.history.get("loss", [])
        ],
        "configured_max_epochs": float(resolved_training_config["max_epochs"]),
        "hidden_layer_sizes": list(resolved_training_config["hidden_layer_sizes"]),
        "learning_rate": float(resolved_training_config["learning_rate"]),
        "batch_size": float(resolved_training_config["batch_size"]),
        "patience": float(resolved_training_config["patience"]),
        "min_delta": float(resolved_training_config["min_delta"]),
        "validation_split": float(resolved_training_config["validation_split"]),
        "l2_reg": float(resolved_training_config["l2_reg"]),
        "seed": float(model_seed),
    }
    if "val_loss" in history.history and history.history["val_loss"]:
        metrics["best_val_loss"] = float(np.min(history.history["val_loss"]))
        metrics["val_loss_history"] = [
            float(loss_value) for loss_value in history.history.get("val_loss", [])
        ]

    return folded_model, metrics


def _fit_bundle(
    unit_name: str,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    seed: int = 42,
    output_training_overrides: Dict[str, Dict[str, Any]] | None = None,
) -> UnitSurrogateBundle:
    models: Dict[str, tf.keras.Sequential] = {}
    model_metrics: Dict[str, Dict[str, Any]] = {}
    resolved_output_configs = _resolve_output_training_configs(
        unit_name,
        list(Y.columns),
        output_training_overrides=output_training_overrides,
    )

    for output_idx, output_name in enumerate(Y.columns):
        model_seed = seed + output_idx + 1
        model, metrics = _train_single_output_model(
            X,
            Y[output_name],
            seed=model_seed,
            training_config=resolved_output_configs[output_name],
        )
        models[output_name] = model
        model_metrics[output_name] = metrics

    input_bounds = _resolve_input_bounds(unit_name, X)
    output_bounds = {col: _nondegenerate_bounds(Y[col]) for col in Y.columns}

    return UnitSurrogateBundle(
        unit_name=unit_name,
        feature_names=list(X.columns),
        output_names=list(Y.columns),
        models=models,
        input_bounds=input_bounds,
        output_bounds=output_bounds,
        model_metrics=model_metrics,
        training_config={
            "hidden_layer_sizes": list(ANN_HIDDEN_LAYER_SIZES),
            "learning_rate": ANN_LEARNING_RATE,
            "max_epochs": ANN_MAX_EPOCHS,
            "patience": ANN_PATIENCE,
            "min_delta": ANN_MIN_DELTA,
            "validation_split": ANN_VALIDATION_SPLIT,
            "batch_size": ANN_BATCH_SIZE,
            "l2_reg": ANN_L2_REG,
            "seed": seed,
            "defaults": {
                "hidden_layer_sizes": list(ANN_HIDDEN_LAYER_SIZES),
                "learning_rate": ANN_LEARNING_RATE,
                "max_epochs": ANN_MAX_EPOCHS,
                "patience": ANN_PATIENCE,
                "min_delta": ANN_MIN_DELTA,
                "validation_split": ANN_VALIDATION_SPLIT,
                "batch_size": ANN_BATCH_SIZE,
                "l2_reg": ANN_L2_REG,
            },
            "per_output": {
                output_name: _serialize_output_training_config(config)
                for output_name, config in resolved_output_configs.items()
            },
        },
        model_kind="relu_ann",
    )


def _save_bundle(bundle: UnitSurrogateBundle, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / f"{bundle.unit_name}.joblib"
    joblib.dump(bundle, bundle_path)

    keras_dir = model_dir / f"{bundle.unit_name}_keras"
    keras_dir.mkdir(parents=True, exist_ok=True)
    for output_name, model in bundle.models.items():
        model.save(keras_dir / f"{output_name}.keras")

    metadata = {
        "unit_name": bundle.unit_name,
        "feature_names": bundle.feature_names,
        "output_names": bundle.output_names,
        "input_bounds": bundle.input_bounds,
        "output_bounds": bundle.output_bounds,
        "model_metrics": bundle.model_metrics,
        "training_config": bundle.training_config,
        "model_kind": bundle.model_kind,
    }
    with (model_dir / f"{bundle.unit_name}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_json_default)


def _load_bundle(unit_name: str, model_dir: Path) -> UnitSurrogateBundle:
    return _load_or_raise(model_dir / f"{unit_name}.joblib")


def _get_bundle(unit_name: str, model_dir: Path = DEFAULT_MODEL_DIR) -> UnitSurrogateBundle:
    key = f"{model_dir.resolve()}::{unit_name}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _load_bundle(unit_name, model_dir)
    return _MODEL_CACHE[key]


def reactor_capacity_from_geometry(Kl, Kd=1.0):
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


def _filter_xy_to_fixed_geometry(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    fixed_geometry: Dict[str, float],
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    working_X = X.copy()
    working_Y = Y.copy()
    applied_filters: dict[str, float] = {}
    mask = np.ones(len(working_X), dtype=bool)

    for column_name, target_value in fixed_geometry.items():
        if column_name not in working_X.columns:
            continue

        values = pd.to_numeric(working_X[column_name], errors="coerce").to_numpy(dtype=float)
        tolerance = max(1e-8, 1e-6 * max(abs(float(target_value)), 1.0))
        mask &= np.isfinite(values) & np.isclose(
            values,
            float(target_value),
            atol=tolerance,
            rtol=0.0,
        )
        applied_filters[column_name] = float(target_value)

    if applied_filters:
        if not np.any(mask):
            raise ValueError(
                f"{label} feature data does not contain any rows with fixed geometry "
                f"{applied_filters}."
            )
        working_X = working_X.loc[mask].reset_index(drop=True)
        working_Y = working_Y.loc[mask].reset_index(drop=True)

    return working_X, working_Y


def _infer_energy_target(
    df: pd.DataFrame,
    duty_columns: List[str],
    total_abs_column: str,
) -> pd.Series:
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
        elif {"Mt", "NH3_out_kgph"}.issubset(working_df.columns):
            total_flow = pd.to_numeric(working_df["Mt"], errors="coerce")
            ammonia_flow = pd.to_numeric(working_df["NH3_out_kgph"], errors="coerce")
            nh3_wtfrac = ammonia_flow / np.maximum(total_flow, 1e-6)
        else:
            raise ValueError(
                "Could not infer ammonia_kgph and water_kgph. Expected either the direct "
                "component-flow targets or one of the legacy total-flow/composition pairs."
            )

        nh3_wtfrac = pd.Series(
            _clip(_to_numpy_array(nh3_wtfrac).astype(float), 0.0, 1.0),
            index=working_df.index,
        )
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

    working_df["ammonia_kgph"] = np.maximum(
        0.0, working_df["ammonia_kgph"].to_numpy(dtype=float)
    )
    working_df["water_kgph"] = np.maximum(
        0.0, working_df["water_kgph"].to_numpy(dtype=float)
    )
    working_df["electric_kwhph"] = np.maximum(
        0.0, working_df["electric_kwhph"].to_numpy(dtype=float)
    )
    return working_df


def _ensure_urea_direct_targets(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    if "product_urea_wtfrac" not in working_df.columns:
        if "Wurea" in working_df.columns:
            working_df["product_urea_wtfrac"] = pd.to_numeric(
                working_df["Wurea"], errors="coerce"
            )
        else:
            raise ValueError(
                "Could not infer product_urea_wtfrac. Expected either "
                "product_urea_wtfrac or the legacy Wurea column."
            )

    if "pure_urea_kgph" not in working_df.columns:
        if "urea_kgph" in working_df.columns:
            working_df["pure_urea_kgph"] = pd.to_numeric(
                working_df["urea_kgph"], errors="coerce"
            )
        elif "Ft_UREA-OUT" in working_df.columns:
            total_product_flow = pd.to_numeric(
                working_df["Ft_UREA-OUT"], errors="coerce"
            )
            urea_wtfrac = pd.to_numeric(
                working_df["product_urea_wtfrac"], errors="coerce"
            )
            urea_wtfrac = pd.Series(
                _clip(_to_numpy_array(urea_wtfrac).astype(float), 0.0, 1.0),
                index=working_df.index,
            )
            working_df["pure_urea_kgph"] = total_product_flow * urea_wtfrac
        else:
            raise ValueError(
                "Could not infer pure_urea_kgph. Expected either pure_urea_kgph, "
                "urea_kgph, or the legacy Ft_UREA-OUT/Wurea pair."
            )

    if "electric_kwhph" not in working_df.columns:
        working_df["electric_kwhph"] = _infer_energy_target(
            working_df,
            duty_columns=UREAF_HEAT_DUTY_COLUMNS,
            total_abs_column=UREAF_TOTAL_ABS_DUTY_COLUMN,
        )

    for col in UREA_DIRECT_OUTPUT_COLUMNS:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    working_df["pure_urea_kgph"] = np.maximum(
        0.0, working_df["pure_urea_kgph"].to_numpy(dtype=float)
    )
    working_df["product_urea_wtfrac"] = _clip(
        working_df["product_urea_wtfrac"].to_numpy(dtype=float),
        0.0,
        1.0,
    )
    working_df["electric_kwhph"] = np.maximum(
        0.0, working_df["electric_kwhph"].to_numpy(dtype=float)
    )
    return working_df


def _resolve_input_bounds(
    unit_name: str,
    X: pd.DataFrame,
) -> Dict[str, Tuple[float, float]]:
    requested_bounds = UNIT_INPUT_BOUND_OVERRIDES.get(
        _canonical_unit_name(unit_name),
        {},
    )
    input_bounds: Dict[str, Tuple[float, float]] = {}

    for column_name in X.columns:
        if column_name in requested_bounds:
            lower, upper = requested_bounds[column_name]
            if float(lower) >= float(upper):
                raise ValueError(
                    f"Configured input bounds for {unit_name}.{column_name} are invalid: "
                    f"{(lower, upper)}"
                )
            input_bounds[column_name] = (float(lower), float(upper))
        else:
            input_bounds[column_name] = _nondegenerate_bounds(X[column_name])

    return input_bounds


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
                active_mask = inputs_df["active"].apply(
                    lambda value: _parse_bool(value, default=True)
                )
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
                    col
                    for col in input_like_columns
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

    _, raw_feature_names = _load_ammoniaf_column_sets(
        case_grid_csv=case_grid_csv,
        inputs_csv=inputs_csv,
    )

    results_raw = pd.read_csv(results_csv)
    missing_input_columns = [col for col in raw_feature_names if col not in results_raw.columns]
    if missing_input_columns:
        print(
            "Ignoring active ammoniaF input columns not present in the results CSV: "
            f"{missing_input_columns}"
        )
        raw_feature_names = [col for col in raw_feature_names if col in results_raw.columns]

    if not raw_feature_names:
        raise ValueError("No ammoniaF input feature columns were found in the results CSV.")

    working_df = results_raw.copy()
    if "run_ok" in working_df.columns:
        run_ok = pd.to_numeric(working_df["run_ok"], errors="coerce").fillna(0)
        working_df = working_df.loc[run_ok == 1].copy()

    numeric_columns = [col for col in working_df.columns if col not in AMMONIAF_RESULT_META_COLUMNS]
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

    available_duties = [col for col in AMMONIAF_DUTY_COLUMNS if col in working_df.columns]
    if available_duties:
        working_df[AMMONIAF_TOTAL_ABS_DUTY_COLUMN] = (
            working_df[available_duties].abs().sum(axis=1)
        )

    working_df = _ensure_nitrate_component_targets(working_df)

    feature_names = _ammoniaF_model_feature_names(raw_feature_names)
    required_columns = feature_names + AMMONIA_COMPONENT_OUTPUT_COLUMNS
    working_df = working_df.dropna(subset=required_columns).reset_index(drop=True)

    if working_df.empty:
        raise ValueError(
            "No successful ammoniaF rows remained after filtering and target conversion."
        )

    X = working_df[feature_names].copy()
    Y = working_df[AMMONIA_COMPONENT_OUTPUT_COLUMNS].copy()
    return X, Y, working_df


def load_ureaF_training_data(
    results_csv: Path | str = UREAF_RESULTS_CSV,
    include_all_heat_duties: bool = UREAF_TRAIN_ALL_HEAT_DUTIES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del include_all_heat_duties

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
        col
        for col in UREAF_HEAT_DUTY_COLUMNS
        if col in working_df.columns and col not in UREAF_EXCLUDED_HEAT_DUTY_COLUMNS
    ]
    if not heat_duty_columns:
        raise ValueError("No usable ureaF heat-duty columns were found in the results CSV.")

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


def train_and_save_ammoniaF_surrogate(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    results_csv: Path | str = AMMONIAF_RESULTS_CSV,
    case_grid_csv: Path | str | None = None,
    inputs_csv: Path | str | None = AMMONIAF_INPUTS_CSV,
    seed: int = 42,
    output_training_overrides: Dict[str, Dict[str, Any]] | None = None,
    **_unused,
) -> UnitSurrogateBundle:
    X, Y, _ = load_ammoniaF_training_data(
        results_csv=results_csv,
        case_grid_csv=case_grid_csv,
        inputs_csv=inputs_csv,
    )
    bundle = _fit_bundle(
        "ammoniaF_unit",
        X,
        Y,
        seed=seed,
        output_training_overrides=output_training_overrides,
    )
    _save_bundle(bundle, Path(model_dir))
    _MODEL_CACHE.pop(f"{Path(model_dir).resolve()}::ammoniaF_unit", None)
    _print_unit_validation_report("ammoniaF_unit", bundle, X, Y)
    return bundle


def train_and_save_ureaF_surrogate(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    results_csv: Path | str = UREAF_RESULTS_CSV,
    include_all_heat_duties: bool = UREAF_TRAIN_ALL_HEAT_DUTIES,
    seed: int = 52,
    output_training_overrides: Dict[str, Dict[str, Any]] | None = None,
    **_unused,
) -> UnitSurrogateBundle:
    X, Y, _ = load_ureaF_training_data(
        results_csv=results_csv,
        include_all_heat_duties=include_all_heat_duties,
    )
    bundle = _fit_bundle(
        "ureaF_unit",
        X,
        Y,
        seed=seed,
        output_training_overrides=output_training_overrides,
    )
    _save_bundle(bundle, Path(model_dir))
    _MODEL_CACHE.pop(f"{Path(model_dir).resolve()}::ureaF_unit", None)
    _print_unit_validation_report("ureaF_unit", bundle, X, Y)
    return bundle


def ammoniaF_unit(
    Ft: float,
    Fh2: float,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
) -> Dict[str, float]:
    bundle = _get_bundle("ammoniaF_unit", Path(model_dir))
    return bundle.predict(
        {
            "Ft": Ft,
            "Fh2": Fh2,
        }
    )


def ureaF_unit(
    Fnh3: float,
    Fco2: float,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
) -> Dict[str, float]:
    bundle = _get_bundle("ureaF_unit", Path(model_dir))
    return bundle.predict(
        {
            "Fnh3": Fnh3,
            "Fco2": Fco2,
        }
    )


nitrate_unit = ammoniaF_unit
urea_unit = ureaF_unit


def _load_frame_from_candidates(data_dir: Path, candidates: List[str]) -> pd.DataFrame | None:
    for candidate in candidates:
        candidate_path = data_dir / candidate
        if candidate_path.exists():
            return pd.read_csv(candidate_path)
    return None


def save_training_data_to_csv(
    data_dir: Path | str,
    n_samples: int = 2000,
    seed: int = 42,
    ammonia_results_csv: Path | str = AMMONIAF_RESULTS_CSV,
    urea_results_csv: Path | str = UREAF_RESULTS_CSV,
    **_unused,
) -> Dict[str, Path]:
    del n_samples, seed

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    ammonia_case_grid = AMMONIAF_CASE_GRID_CSV if AMMONIAF_CASE_GRID_CSV.exists() else None
    ammonia_inputs = AMMONIAF_INPUTS_CSV if AMMONIAF_INPUTS_CSV.exists() else None
    X_a, Y_a, _ = load_ammoniaF_training_data(
        results_csv=ammonia_results_csv,
        case_grid_csv=ammonia_case_grid,
        inputs_csv=ammonia_inputs,
    )
    X_u, Y_u, _ = load_ureaF_training_data(results_csv=urea_results_csv)

    outputs = {
        "ammoniaF_unit_X.csv": X_a,
        "ammoniaF_unit_Y.csv": Y_a,
        "nitrate_unit_X.csv": X_a,
        "nitrate_unit_Y.csv": Y_a,
        "ureaF_unit_X.csv": X_u,
        "ureaF_unit_Y.csv": Y_u,
        "urea_unit_X.csv": X_u,
        "urea_unit_Y.csv": Y_u,
    }

    saved_paths: Dict[str, Path] = {}
    for filename, frame in outputs.items():
        output_path = data_dir / filename
        frame.to_csv(output_path, index=False)
        saved_paths[filename] = output_path

    return saved_paths


def train_and_save_all_surrogates(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    n_samples: int = 2000,
    seed: int = 42,
    data_dir: Path | str | None = None,
    unit_training_overrides: Dict[str, Dict[str, Dict[str, Any]]] | None = None,
    **_unused,
) -> Dict[str, UnitSurrogateBundle]:
    del n_samples

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is not None:
        data_dir = Path(data_dir)
        X_a = _load_frame_from_candidates(data_dir, ["ammoniaF_unit_X.csv", "nitrate_unit_X.csv"])
        Y_a = _load_frame_from_candidates(data_dir, ["ammoniaF_unit_Y.csv", "nitrate_unit_Y.csv"])
        X_u = _load_frame_from_candidates(data_dir, ["ureaF_unit_X.csv", "urea_unit_X.csv"])
        Y_u = _load_frame_from_candidates(data_dir, ["ureaF_unit_Y.csv", "urea_unit_Y.csv"])
    else:
        X_a = Y_a = X_u = Y_u = None

    if X_a is None or Y_a is None:
        ammonia_case_grid = AMMONIAF_CASE_GRID_CSV if AMMONIAF_CASE_GRID_CSV.exists() else None
        ammonia_inputs = AMMONIAF_INPUTS_CSV if AMMONIAF_INPUTS_CSV.exists() else None
        X_a, Y_a, _ = load_ammoniaF_training_data(
            results_csv=AMMONIAF_RESULTS_CSV,
            case_grid_csv=ammonia_case_grid,
            inputs_csv=ammonia_inputs,
        )
    else:
        X_a, Y_a = _filter_xy_to_fixed_geometry(
            X_a,
            Y_a,
            AMMONIAF_FIXED_GEOMETRY,
            label="ammoniaF",
        )
        X_a = X_a[_ammoniaF_model_feature_names(list(X_a.columns))]
        Y_a = _ensure_nitrate_component_targets(Y_a)[AMMONIA_COMPONENT_OUTPUT_COLUMNS]

    if X_u is None or Y_u is None:
        X_u, Y_u, _ = load_ureaF_training_data(
            results_csv=UREAF_RESULTS_CSV,
            include_all_heat_duties=UREAF_TRAIN_ALL_HEAT_DUTIES,
        )
    else:
        X_u, Y_u = _filter_xy_to_fixed_geometry(
            X_u,
            Y_u,
            UREAF_FIXED_GEOMETRY,
            label="ureaF",
        )
        X_u = X_u[_ureaF_model_feature_names(list(X_u.columns))]
        Y_u = _ensure_urea_direct_targets(Y_u)[UREA_DIRECT_OUTPUT_COLUMNS]

    ammonia_bundle = _fit_bundle(
        "ammoniaF_unit",
        X_a,
        Y_a,
        seed=seed,
        output_training_overrides=_resolve_unit_training_overrides(
            "ammoniaF_unit",
            unit_training_overrides,
        ),
    )
    _save_bundle(ammonia_bundle, model_dir)
    _print_unit_validation_report("ammoniaF_unit", ammonia_bundle, X_a, Y_a)

    urea_bundle = _fit_bundle(
        "ureaF_unit",
        X_u,
        Y_u,
        seed=seed + 10,
        output_training_overrides=_resolve_unit_training_overrides(
            "ureaF_unit",
            unit_training_overrides,
        ),
    )
    _save_bundle(urea_bundle, model_dir)
    _print_unit_validation_report("ureaF_unit", urea_bundle, X_u, Y_u)

    _MODEL_CACHE.pop(f"{model_dir.resolve()}::ammoniaF_unit", None)
    _MODEL_CACHE.pop(f"{model_dir.resolve()}::ureaF_unit", None)

    return {
        "ammoniaF_unit": ammonia_bundle,
        "ureaF_unit": urea_bundle,
        "nitrate_unit": ammonia_bundle,
        "urea_unit": urea_bundle,
    }


def _print_unit_validation_report(
    unit_name: str,
    bundle: UnitSurrogateBundle,
    X: pd.DataFrame,
    Y: pd.DataFrame,
) -> None:
    training_config = bundle.training_config or {}
    default_training_config = training_config.get("defaults", training_config)
    default_layers = _normalize_output_training_config(default_training_config)["hidden_layer_sizes"]
    default_max_epochs = _normalize_output_training_config(default_training_config)["max_epochs"]

    print(f"\nValidation report for {unit_name}:")
    print("  model type: ReLU ANN")
    print(f"  inputs:  {list(X.columns)}")
    print(f"  outputs: {list(Y.columns)}")
    print(
        f"  default hidden layers: {list(default_layers)} | "
        f"default max epochs: {int(default_max_epochs)}"
    )
    for output_name in Y.columns:
        bounds = bundle.output_bounds.get(output_name, _nondegenerate_bounds(Y[output_name]))
        metrics = (bundle.model_metrics or {}).get(output_name, {})
        output_training_config = _bundle_output_training_config(bundle, output_name)
        print(
            f"  {output_name}: "
            f"range=[{bounds[0]:.6g}, {bounds[1]:.6g}] | "
            f"MAE={metrics.get('mae', float('nan')):.6g} | "
            f"RMSE={metrics.get('rmse', float('nan')):.6g} | "
            f"R2={metrics.get('r2', float('nan')):.4f} | "
            f"layers={list(output_training_config['hidden_layer_sizes'])} | "
            f"max_epochs={int(output_training_config['max_epochs'])} | "
            f"epochs={int(metrics.get('epochs_ran', 0))}"
        )
    print(
        "  reluMIP compatibility: saved per-output models are tf.keras.Sequential "
        "networks with ReLU hidden layers and a linear output layer."
    )


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def _make_labeled_surrogate_plots(
    model,
    features,
    unit_name: str,
    x_name: str,
    y_name: str,
    z_name: str,
    out_prefix: Path | None = None,
    cmap: str = "viridis",
    save: bool = True,
    grid_points: int = 80,
    model_features=None,
    grid_feature_builder=None,
    point_values=None,
    point_label: str = "Model predictions at training points",
    show_3d_plot: bool = False,
):
    del model_features

    feat_np = _to_numpy_array(features).astype(np.float32)
    if feat_np.ndim != 2 or feat_np.shape[1] < 2:
        raise ValueError("`features` must be a 2D array with at least two columns.")

    x_min, x_max = float(feat_np[:, 0].min()), float(feat_np[:, 0].max())
    y_min, y_max = float(feat_np[:, 1].min()), float(feat_np[:, 1].max())

    x_grid = np.linspace(x_min, x_max, grid_points)
    y_grid = np.linspace(y_min, y_max, grid_points)
    xx, yy = np.meshgrid(x_grid, y_grid)

    if grid_feature_builder is None:
        grid_feature_array = np.column_stack([xx.ravel(), yy.ravel()])
    else:
        grid_feature_array = grid_feature_builder(xx.ravel(), yy.ravel())

    zz = _keras_predict(model, grid_feature_array).reshape(xx.shape)
    if point_values is None:
        point_values_np = _keras_predict(model, feat_np).reshape(-1)
    else:
        point_values_np = _to_numpy_array(point_values).astype(float).reshape(-1)

    saved_files: List[Path] = []

    fig1, ax1 = plt.subplots(figsize=(6.5, 5.0))
    contour = ax1.contourf(xx, yy, zz, levels=20, cmap=cmap)
    ax1.scatter(
        feat_np[:, 0],
        feat_np[:, 1],
        c=point_values_np,
        cmap=cmap,
        edgecolors="black",
        linewidths=0.25,
        s=20,
        alpha=0.9,
    )
    ax1.set_xlabel(x_name)
    ax1.set_ylabel(y_name)
    ax1.set_title(f"{unit_name}: {z_name} (ReLU ANN surface)")
    cbar = fig1.colorbar(contour, ax=ax1, pad=0.02)
    cbar.set_label(z_name)
    fig1.tight_layout()
    if save and out_prefix is not None:
        f1 = Path(f"{out_prefix}_view1.png")
        fig1.savefig(f1, dpi=300, bbox_inches="tight")
        saved_files.append(f1)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(8.0, 6.0))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(xx, yy, zz, cmap=cmap, linewidth=0, antialiased=True, alpha=0.85)
    ax2.scatter(
        feat_np[:, 0],
        feat_np[:, 1],
        point_values_np,
        c="black",
        s=8,
        alpha=0.35,
        depthshade=False,
        label=point_label,
    )
    ax2.set_xlabel(x_name)
    ax2.set_ylabel(y_name)
    ax2.set_zlabel(z_name)
    ax2.set_title(f"{unit_name}: {z_name} (ReLU ANN)")
    ax2.view_init(elev=ISOMETRIC_ELEVATION_DEG, azim=ISOMETRIC_AZIMUTH_DEG)
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

    return saved_files


def save_ammoniaF_tree_plots(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    results_csv: Path | str = AMMONIAF_RESULTS_CSV,
    case_grid_csv: Path | str | None = None,
    inputs_csv: Path | str | None = AMMONIAF_INPUTS_CSV,
    show_3d_plots: bool = SHOW_INTERACTIVE_3D_PLOTS,
) -> None:
    model_dir = Path(model_dir)
    plot_root = model_dir / "plots_phase" / "ammoniaF_unit"
    plot_root.mkdir(parents=True, exist_ok=True)

    X, Y, _ = load_ammoniaF_training_data(
        results_csv=results_csv,
        case_grid_csv=case_grid_csv,
        inputs_csv=inputs_csv,
    )

    if len(X.columns) < 2:
        raise ValueError("At least two ammoniaF inputs are required to generate plots.")

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

    full_feature_df = X[bundle.feature_names].copy()
    full_features = full_feature_df.to_numpy(dtype=np.float32)
    slice_baseline = {name: float(full_feature_df[name].median()) for name in bundle.feature_names}

    all_metrics: List[Dict[str, Any]] = []
    for x_name, y_name in combinations(bundle.feature_names, 2):
        pair_dir = plot_root / f"{x_name}__{y_name}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        plot_features = full_feature_df[[x_name, y_name]].to_numpy(dtype=np.float32)
        fixed_inputs = {
            name: value
            for name, value in slice_baseline.items()
            if name not in {x_name, y_name}
        }

        def build_full_slice_features(x_values, y_values, x_col=x_name, y_col=y_name):
            grid_df = pd.DataFrame(
                {
                    name: np.full(len(x_values), value, dtype=float)
                    for name, value in slice_baseline.items()
                }
            )
            grid_df[x_col] = x_values
            grid_df[y_col] = y_values
            return grid_df[bundle.feature_names].to_numpy(dtype=np.float32)

        for output_name in bundle.output_names:
            model = bundle.models[output_name]
            y_pred = _keras_predict(model, full_features).reshape(-1)
            y_true = Y[output_name].to_numpy(dtype=float).reshape(-1)
            metrics = (bundle.model_metrics or {}).get(output_name, {})
            n_relu_units = int(
                sum(
                    layer.units
                    for layer in model.layers[:-1]
                    if isinstance(layer, tf.keras.layers.Dense)
                )
            )

            print(
                f"ammoniaF_unit | {x_name} vs {y_name} | {output_name} | "
                f"ReLU ANN slice | Units: {n_relu_units} | "
                f"MAE: {mean_absolute_error(y_true, y_pred):.4f} | "
                f"MAPE: {np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100:.2f}% | "
                f"Fixed inputs: {fixed_inputs}"
            )

            _make_labeled_surrogate_plots(
                model=model,
                features=plot_features,
                unit_name="ammoniaF_unit",
                x_name=x_name,
                y_name=y_name,
                z_name=output_name,
                out_prefix=pair_dir / output_name,
                cmap="viridis",
                save=True,
                model_features=full_features,
                grid_feature_builder=build_full_slice_features,
                point_values=y_true,
                point_label="Raw training data at Aspen cases",
                show_3d_plot=show_3d_plots,
            )

            all_metrics.append(
                {
                    "unit": "ammoniaF_unit",
                    "x_feature": x_name,
                    "y_feature": y_name,
                    "output": output_name,
                    "slice_type": "full_reduced_model",
                    "fixed_inputs": json.dumps(fixed_inputs),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
                    "mape": float(
                        np.mean(
                            np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))
                        )
                        * 100.0
                    ),
                    "r2": float(metrics.get("r2", np.nan)),
                    "n_relu_units": n_relu_units,
                }
            )

    pd.DataFrame(all_metrics).to_csv(plot_root / "phase_plot_metrics.csv", index=False)


def save_ureaF_tree_plots(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    results_csv: Path | str = UREAF_RESULTS_CSV,
    include_all_heat_duties: bool = UREAF_TRAIN_ALL_HEAT_DUTIES,
    show_3d_plots: bool = SHOW_INTERACTIVE_3D_PLOTS,
) -> None:
    model_dir = Path(model_dir)
    plot_root = model_dir / "plots_phase" / "ureaF_unit"
    plot_root.mkdir(parents=True, exist_ok=True)

    X, Y, _ = load_ureaF_training_data(
        results_csv=results_csv,
        include_all_heat_duties=include_all_heat_duties,
    )

    if len(X.columns) < 2:
        raise ValueError("At least two ureaF inputs are required to generate plots.")

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
        _MODEL_CACHE.pop(f"{model_dir.resolve()}::ureaF_unit", None)
        bundle = _get_bundle("ureaF_unit", model_dir)

    full_feature_df = X[bundle.feature_names].copy()
    full_features = full_feature_df.to_numpy(dtype=np.float32)
    slice_baseline = {name: float(full_feature_df[name].median()) for name in bundle.feature_names}

    all_metrics: List[Dict[str, Any]] = []
    for x_name, y_name in combinations(bundle.feature_names, 2):
        pair_dir = plot_root / f"{x_name}__{y_name}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        plot_features = full_feature_df[[x_name, y_name]].to_numpy(dtype=np.float32)
        fixed_inputs = {
            name: value
            for name, value in slice_baseline.items()
            if name not in {x_name, y_name}
        }

        def build_full_slice_features(x_values, y_values, x_col=x_name, y_col=y_name):
            grid_df = pd.DataFrame(
                {
                    name: np.full(len(x_values), value, dtype=float)
                    for name, value in slice_baseline.items()
                }
            )
            grid_df[x_col] = x_values
            grid_df[y_col] = y_values
            return grid_df[bundle.feature_names].to_numpy(dtype=np.float32)

        for output_name in bundle.output_names:
            model = bundle.models[output_name]
            y_pred = _keras_predict(model, full_features).reshape(-1)
            y_true = Y[output_name].to_numpy(dtype=float).reshape(-1)
            metrics = (bundle.model_metrics or {}).get(output_name, {})
            n_relu_units = int(
                sum(
                    layer.units
                    for layer in model.layers[:-1]
                    if isinstance(layer, tf.keras.layers.Dense)
                )
            )

            print(
                f"ureaF_unit | {x_name} vs {y_name} | {output_name} | "
                f"ReLU ANN slice | Units: {n_relu_units} | "
                f"MAE: {mean_absolute_error(y_true, y_pred):.6f} | "
                f"MAPE: {np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100:.2f}% | "
                f"Fixed inputs: {fixed_inputs}"
            )

            _make_labeled_surrogate_plots(
                model=model,
                features=plot_features,
                unit_name="ureaF_unit",
                x_name=x_name,
                y_name=y_name,
                z_name=output_name,
                out_prefix=pair_dir / output_name,
                cmap="viridis",
                save=True,
                model_features=full_features,
                grid_feature_builder=build_full_slice_features,
                point_values=y_true,
                point_label="Raw training data at Aspen cases",
                show_3d_plot=show_3d_plots,
            )

            all_metrics.append(
                {
                    "unit": "ureaF_unit",
                    "x_feature": x_name,
                    "y_feature": y_name,
                    "output": output_name,
                    "slice_type": "full_model",
                    "fixed_inputs": json.dumps(fixed_inputs),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
                    "mape": float(
                        np.mean(
                            np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))
                        )
                        * 100.0
                    ),
                    "r2": float(metrics.get("r2", np.nan)),
                    "n_relu_units": n_relu_units,
                    "include_all_heat_duties": bool(include_all_heat_duties),
                }
            )

    pd.DataFrame(all_metrics).to_csv(plot_root / "phase_plot_metrics.csv", index=False)


if __name__ == "__main__":
    model_dir = DEFAULT_MODEL_DIR
    use_ammoniaf_data = AMMONIAF_RESULTS_CSV.exists()
    use_ureaf_data = UREAF_RESULTS_CSV.exists()

    if use_ammoniaf_data or use_ureaf_data:
        if use_ammoniaf_data:
            print(f"Training ammoniaF surrogate from: {AMMONIAF_RESULTS_CSV.resolve()}")
            ammonia_bundle = train_and_save_ammoniaF_surrogate(
                model_dir=model_dir,
                results_csv=AMMONIAF_RESULTS_CSV,
                case_grid_csv=AMMONIAF_CASE_GRID_CSV if AMMONIAF_CASE_GRID_CSV.exists() else None,
                inputs_csv=AMMONIAF_INPUTS_CSV if AMMONIAF_INPUTS_CSV.exists() else None,
            )
            print(
                f"Loaded ammoniaF training data with "
                f"{len(ammonia_bundle.feature_names)} inputs and "
                f"{len(ammonia_bundle.output_names)} outputs."
            )

        if use_ureaf_data:
            print(f"Training ureaF surrogate from: {UREAF_RESULTS_CSV.resolve()}")
            urea_bundle = train_and_save_ureaF_surrogate(
                model_dir=model_dir,
                results_csv=UREAF_RESULTS_CSV,
                include_all_heat_duties=UREAF_TRAIN_ALL_HEAT_DUTIES,
            )
            print(
                f"Loaded ureaF training data with "
                f"{len(urea_bundle.feature_names)} inputs and "
                f"{len(urea_bundle.output_names)} outputs."
            )

"""
Hyperparameter sweep for the ureaF_unit electricity surrogate.

This script trains only the ureaF_unit.electric_kwhph target and writes a
rankable search table. It does not change trained_unit_surrogates/ureaF_unit.joblib.

Example:
    python optimize_ureaF_electricity_parameters.py

For a smaller trial run:
    python optimize_ureaF_electricity_parameters.py --max-runs 12 --epochs 400 1000
"""
# python optimize_ureaF_electricity_parameters.py --max-runs 12 --epochs 400 1000


from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import surrogate_functions as sf


TARGET_OUTPUT = "electric_kwhph"
UNIT_NAME = "ureaF_unit"

DEFAULT_OUTPUT_DIR = sf.THIS_DIR / "ureaF_electricity_parameter_search"
DEFAULT_LAYER_OPTIONS = ("3,3", "4,3", "5,5,3", "6,3", "8,5", "8,5,3")
DEFAULT_EPOCHS = (400, 1000, 3000, 10000)
DEFAULT_LEARNING_RATES = (1e-3, 2e-3, 5e-3)
DEFAULT_BATCH_SIZES = (128, 256)
DEFAULT_PATIENCES = (10, 25)
DEFAULT_L2_REGS = (0.0, 1e-6)

LOWER_IS_BETTER = {
    "val_mae",
    "val_rmse",
    "val_mape_pct",
    "test_mae",
    "test_rmse",
    "test_mape_pct",
}

RESULT_FIELDNAMES = [
    "run_index",
    "total_runs",
    "status",
    "config_id",
    "layers",
    "num_hidden_layers",
    "max_epochs",
    "learning_rate",
    "batch_size",
    "patience",
    "min_delta",
    "l2_reg",
    "seed",
    "train_rows",
    "val_rows",
    "test_rows",
    "duration_sec",
    "epochs_ran",
    "best_train_loss",
    "best_val_loss",
    "train_mae",
    "train_rmse",
    "train_mape_pct",
    "train_r2",
    "val_mae",
    "val_rmse",
    "val_mape_pct",
    "val_r2",
    "test_mae",
    "test_rmse",
    "test_mape_pct",
    "test_r2",
    "trainable_parameters",
    "error_message",
]


@dataclass(frozen=True)
class SearchConfig:
    hidden_layer_sizes: tuple[int, ...]
    max_epochs: int
    learning_rate: float
    batch_size: int
    patience: int
    min_delta: float
    l2_reg: float
    seed: int


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Keras/ReLU hyperparameters for ureaF_unit.electric_kwhph. "
            "The search results are written under --output-dir; the existing "
            "saved ureaF surrogate bundle is left untouched."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=sf.UREAF_RESULTS_CSV,
        help="Path to the ureaF Aspen results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for sweep CSV, metadata JSON, best config JSON, and best model.",
    )
    parser.add_argument(
        "--layer-options",
        nargs="+",
        default=list(DEFAULT_LAYER_OPTIONS),
        help=(
            "Hidden-layer architectures to sweep. Use comma or x separators, "
            "for example: 5,5,3 8x5 10,6,3."
        ),
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=list(DEFAULT_EPOCHS),
        help="Maximum epoch counts to sweep.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=list(DEFAULT_LEARNING_RATES),
        help="Adam learning rates to sweep.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to sweep.",
    )
    parser.add_argument(
        "--patiences",
        nargs="+",
        type=int,
        default=list(DEFAULT_PATIENCES),
        help="Early-stopping patience values to sweep.",
    )
    parser.add_argument(
        "--l2-regs",
        nargs="+",
        type=float,
        default=list(DEFAULT_L2_REGS),
        help="Dense-layer L2 regularization strengths to sweep.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[15],
        help="Training seeds to sweep for each hyperparameter combination.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=15,
        help="Random seed for the train/validation/test split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of rows held out for the final test metrics.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Fraction of rows held out for model selection metrics.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=sorted(LOWER_IS_BETTER | {"val_r2", "test_r2"}),
        default="val_rmse",
        help="Metric used to choose the best configuration.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of configurations to run from the grid.",
    )
    parser.add_argument(
        "--no-save-best-model",
        action="store_false",
        dest="save_best_model",
        default=True,
        help="Do not save the best folded Keras model.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the sweep if one configuration fails.",
    )
    parser.add_argument(
        "--verbose-fit",
        action="store_true",
        help="Print per-epoch Keras training logs.",
    )
    return parser


def parse_layer_spec(raw_value: str) -> tuple[int, ...]:
    normalized = raw_value.lower().replace("x", ",").replace(";", ",")
    pieces = [piece.strip() for piece in normalized.split(",") if piece.strip()]
    if not pieces:
        raise ValueError(f"Empty layer architecture: {raw_value!r}")
    sizes = tuple(int(piece) for piece in pieces)
    if any(size <= 0 for size in sizes):
        raise ValueError(f"Layer widths must be positive: {raw_value!r}")
    return sizes


def config_id(config: SearchConfig) -> str:
    layers = ",".join(str(size) for size in config.hidden_layer_sizes)
    return (
        f"layers={layers}|epochs={config.max_epochs}|lr={config.learning_rate:g}|"
        f"batch={config.batch_size}|patience={config.patience}|"
        f"min_delta={config.min_delta:g}|l2={config.l2_reg:g}|seed={config.seed}"
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float,
    validation_fraction: float,
    split_seed: int,
) -> DataSplits:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    if test_fraction + validation_fraction >= 1.0:
        raise ValueError("--test-fraction plus --validation-fraction must be less than 1.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=split_seed,
        shuffle=True,
    )
    val_within_train_val = validation_fraction / (1.0 - test_fraction)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_within_train_val,
        random_state=split_seed + 1,
        shuffle=True,
    )

    return DataSplits(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True).astype(np.float32)
    std = values.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(np.abs(std) < sf.MIN_SCALE, 1.0, std).astype(np.float32)
    return mean, std


def apply_scaler(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def build_relu_network(
    input_dim: int,
    hidden_layer_sizes: tuple[int, ...],
    learning_rate: float,
    l2_reg: float,
) -> sf.tf.keras.Sequential:
    regularizer = sf.tf.keras.regularizers.l2(l2_reg) if l2_reg > 0.0 else None
    layers: list[sf.tf.keras.layers.Layer] = [
        sf.tf.keras.layers.Input(shape=(input_dim,), dtype=sf.tf.float32, name="input")
    ]
    for idx, units in enumerate(hidden_layer_sizes, start=1):
        layers.append(
            sf.tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                kernel_regularizer=regularizer,
                name=f"hidden_{idx}",
            )
        )
    layers.append(sf.tf.keras.layers.Dense(1, activation="linear", name="output"))

    model = sf.tf.keras.Sequential(layers, name="ureaF_electricity_search")
    model.compile(
        optimizer=sf.tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mape_pct": float(
            np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0
        ),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def evaluate_model(
    model: sf.tf.keras.Sequential,
    X: pd.DataFrame,
    y: pd.Series,
    prefix: str,
) -> dict[str, float]:
    y_true = y.to_numpy(dtype=np.float32).reshape(-1)
    y_pred = sf._keras_predict(model, X.to_numpy(dtype=np.float32)).reshape(-1)
    return regression_metrics(y_true, y_pred, prefix)


def train_and_score(
    splits: DataSplits,
    config: SearchConfig,
    verbose_fit: bool = False,
) -> tuple[sf.tf.keras.Sequential, dict[str, Any]]:
    x_train = splits.X_train.to_numpy(dtype=np.float32)
    x_val = splits.X_val.to_numpy(dtype=np.float32)
    y_train = splits.y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_val = splits.y_val.to_numpy(dtype=np.float32).reshape(-1, 1)

    x_mean, x_std = fit_scaler(x_train)
    y_mean, y_std = fit_scaler(y_train)

    x_train_scaled = apply_scaler(x_train, x_mean, x_std)
    x_val_scaled = apply_scaler(x_val, x_mean, x_std)
    y_train_scaled = apply_scaler(y_train, y_mean, y_std)
    y_val_scaled = apply_scaler(y_val, y_mean, y_std)

    sf.tf.keras.backend.clear_session()
    sf.tf.keras.utils.set_random_seed(config.seed)

    model = build_relu_network(
        input_dim=x_train.shape[1],
        hidden_layer_sizes=config.hidden_layer_sizes,
        learning_rate=config.learning_rate,
        l2_reg=config.l2_reg,
    )
    callbacks = [
        sf.tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            min_delta=config.min_delta,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x=x_train_scaled,
        y=y_train_scaled,
        validation_data=(x_val_scaled, y_val_scaled),
        epochs=config.max_epochs,
        batch_size=max(1, min(config.batch_size, len(x_train_scaled))),
        shuffle=True,
        callbacks=callbacks,
        verbose=2 if verbose_fit else 0,
    )

    folded_model = sf._fold_standardization_into_model(
        model,
        x_mean.reshape(-1),
        x_std.reshape(-1),
        y_mean.reshape(-1),
        y_std.reshape(-1),
    )

    metrics: dict[str, Any] = {
        "epochs_ran": int(len(history.history.get("loss", []))),
        "best_train_loss": float(np.min(history.history.get("loss", [np.nan]))),
        "best_val_loss": float(np.min(history.history.get("val_loss", [np.nan]))),
        "trainable_parameters": int(
            sum(np.prod(weight.shape) for weight in folded_model.trainable_weights)
        ),
    }
    metrics.update(evaluate_model(folded_model, splits.X_train, splits.y_train, "train"))
    metrics.update(evaluate_model(folded_model, splits.X_val, splits.y_val, "val"))
    metrics.update(evaluate_model(folded_model, splits.X_test, splits.y_test, "test"))
    return folded_model, metrics


def build_search_grid(args: argparse.Namespace) -> list[SearchConfig]:
    layer_options = [parse_layer_spec(value) for value in args.layer_options]
    configs = [
        SearchConfig(
            hidden_layer_sizes=layers,
            max_epochs=int(max_epochs),
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            patience=int(patience),
            min_delta=float(sf.ANN_MIN_DELTA),
            l2_reg=float(l2_reg),
            seed=int(seed),
        )
        for layers, max_epochs, learning_rate, batch_size, patience, l2_reg, seed in itertools.product(
            layer_options,
            args.epochs,
            args.learning_rates,
            args.batch_sizes,
            args.patiences,
            args.l2_regs,
            args.seeds,
        )
    ]
    if args.max_runs is not None:
        if args.max_runs <= 0:
            raise ValueError("--max-runs must be positive when provided.")
        configs = configs[: args.max_runs]
    if not configs:
        raise ValueError("The search grid is empty.")
    return configs


def base_result_row(
    run_index: int,
    total_runs: int,
    config: SearchConfig,
    splits: DataSplits,
) -> dict[str, Any]:
    layers = ",".join(str(size) for size in config.hidden_layer_sizes)
    row = {
        "run_index": run_index,
        "total_runs": total_runs,
        "status": "pending",
        "config_id": config_id(config),
        "layers": layers,
        "num_hidden_layers": len(config.hidden_layer_sizes),
        "max_epochs": config.max_epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "patience": config.patience,
        "min_delta": config.min_delta,
        "l2_reg": config.l2_reg,
        "seed": config.seed,
        "train_rows": len(splits.X_train),
        "val_rows": len(splits.X_val),
        "test_rows": len(splits.X_test),
        "duration_sec": float("nan"),
        "epochs_ran": float("nan"),
        "best_train_loss": float("nan"),
        "best_val_loss": float("nan"),
        "train_mae": float("nan"),
        "train_rmse": float("nan"),
        "train_mape_pct": float("nan"),
        "train_r2": float("nan"),
        "val_mae": float("nan"),
        "val_rmse": float("nan"),
        "val_mape_pct": float("nan"),
        "val_r2": float("nan"),
        "test_mae": float("nan"),
        "test_rmse": float("nan"),
        "test_mape_pct": float("nan"),
        "test_r2": float("nan"),
        "trainable_parameters": float("nan"),
        "error_message": "",
    }
    return row


def is_better(candidate: dict[str, Any], best: dict[str, Any] | None, metric: str) -> bool:
    candidate_score = float(candidate[metric])
    if not np.isfinite(candidate_score):
        return False
    if best is None:
        return True

    best_score = float(best[metric])
    if not np.isfinite(best_score):
        return True
    if metric in LOWER_IS_BETTER:
        return candidate_score < best_score
    return candidate_score > best_score


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    configs: list[SearchConfig],
    X: pd.DataFrame,
    y: pd.Series,
    splits: DataSplits,
) -> None:
    metadata = {
        "unit": UNIT_NAME,
        "target_output": TARGET_OUTPUT,
        "results_csv": str(args.results_csv),
        "rows_loaded": int(len(X)),
        "feature_names": list(X.columns),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "train_rows": int(len(splits.X_train)),
        "val_rows": int(len(splits.X_val)),
        "test_rows": int(len(splits.X_test)),
        "selection_metric": args.selection_metric,
        "total_runs": len(configs),
        "configs": [
            {
                **asdict(config),
                "hidden_layer_sizes": list(config.hidden_layer_sizes),
            }
            for config in configs
        ],
    }
    with (output_dir / "search_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=sf._json_default)


def write_best_summary(
    output_dir: Path,
    args: argparse.Namespace,
    best_row: dict[str, Any],
    best_model_path: Path | None,
) -> None:
    layers = [int(piece) for piece in str(best_row["layers"]).split(",") if piece]
    summary = {
        "unit": UNIT_NAME,
        "target_output": TARGET_OUTPUT,
        "selection_metric": args.selection_metric,
        "best_result": best_row,
        "best_model_path": str(best_model_path) if best_model_path is not None else None,
        "portable_training_override": {
            UNIT_NAME: {
                TARGET_OUTPUT: {
                    "mode": "retrain",
                    "layers": layers,
                    "epochs": int(best_row["max_epochs"]),
                }
            }
        },
        "standalone_training_parameters": {
            "hidden_layer_sizes": layers,
            "max_epochs": int(best_row["max_epochs"]),
            "learning_rate": float(best_row["learning_rate"]),
            "batch_size": int(best_row["batch_size"]),
            "patience": int(best_row["patience"]),
            "min_delta": float(best_row["min_delta"]),
            "l2_reg": float(best_row["l2_reg"]),
            "seed": int(best_row["seed"]),
        },
    }
    with (output_dir / "best_ureaF_electricity_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, default=sf._json_default)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, Y, _ = sf.load_ureaF_training_data(
        results_csv=args.results_csv,
        include_all_heat_duties=sf.UREAF_TRAIN_ALL_HEAT_DUTIES,
    )
    if TARGET_OUTPUT not in Y.columns:
        raise KeyError(f"{TARGET_OUTPUT!r} was not found in ureaF targets: {list(Y.columns)}")

    y = Y[TARGET_OUTPUT].copy()
    splits = split_data(
        X,
        y,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    configs = build_search_grid(args)
    write_metadata(args.output_dir, args, configs, X, y, splits)

    results_csv = args.output_dir / "ureaF_electricity_parameter_search.csv"
    best_row: dict[str, Any] | None = None
    best_model_path = (
        args.output_dir / "best_ureaF_unit_electric_kwhph.keras"
        if args.save_best_model
        else None
    )

    print(f"Loaded {len(X)} ureaF rows with features {list(X.columns)}.")
    print(
        f"Running {len(configs)} configurations; selecting best by {args.selection_metric}."
    )
    print(f"Writing results to {results_csv}")

    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()

        for run_index, config in enumerate(configs, start=1):
            row = base_result_row(run_index, len(configs), config, splits)
            start_time = time.perf_counter()
            print(f"[{run_index}/{len(configs)}] {row['config_id']}")

            try:
                model, metrics = train_and_score(
                    splits,
                    config,
                    verbose_fit=args.verbose_fit,
                )
                row.update(metrics)
                row["duration_sec"] = float(time.perf_counter() - start_time)
                row["status"] = "ok"

                if is_better(row, best_row, args.selection_metric):
                    best_row = dict(row)
                    if best_model_path is not None:
                        model.save(best_model_path, overwrite=True)
                    print(
                        f"  new best {args.selection_metric}="
                        f"{float(row[args.selection_metric]):.6g}"
                    )
            except Exception as exc:
                row["duration_sec"] = float(time.perf_counter() - start_time)
                row["status"] = "failed"
                row["error_message"] = str(exc)
                print(f"  failed: {exc}")
                if args.stop_on_error:
                    writer.writerow({name: row.get(name, "") for name in RESULT_FIELDNAMES})
                    handle.flush()
                    raise

            writer.writerow({name: row.get(name, "") for name in RESULT_FIELDNAMES})
            handle.flush()

    if best_row is None:
        raise RuntimeError("No successful configurations completed.")

    write_best_summary(args.output_dir, args, best_row, best_model_path)
    print(
        f"Best {args.selection_metric}: {float(best_row[args.selection_metric]):.6g} "
        f"with layers={best_row['layers']}, epochs={best_row['max_epochs']}, "
        f"learning_rate={best_row['learning_rate']}."
    )
    print(f"Best configuration saved to {args.output_dir / 'best_ureaF_electricity_config.json'}")


if __name__ == "__main__":
    main()

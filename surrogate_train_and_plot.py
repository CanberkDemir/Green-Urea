"""
Train and/or plot the surrogate units using per-output settings declared in
UNIT_TRAINING_OVERRIDES. Set each output's `mode` to `saved` or `retrain`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import plot_surrogate_graphviz_surface as plotter
import surrogate_functions as sf


# `layers`/`epochs` are only applied when `mode="retrain", "saved"`.
UNIT_TRAINING_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "ammoniaF_unit": {
        "ammonia_kgph": {"mode": "saved", "layers": (5, 3), "epochs": 1000},
        "water_kgph": {"mode": "saved", "layers": (5, 5, 3), "epochs": 1000},
        "electric_kwhph": {"mode": "saved", "layers": (6, 3), "epochs": 1000},
    },
    "ureaF_unit": {
        "pure_urea_kgph": {"mode": "saved", "layers": (5, 5, 3), "epochs": 10000}, # (6, 3)
        "product_urea_wtfrac": {"mode": "saved", "layers": (6, 3), "epochs": 10000},
        "electric_kwhph": {"mode": "saved", "layers": (5, 5, 3), "epochs": 10000}, # (3, 3)
    },
}

DEFAULT_MODEL_MODE = "retrain"
VALID_MODEL_MODES = {"saved", "retrain"}
UNIT_TRAINING_SEEDS = {
    "ammoniaF_unit": 15,
    "ureaF_unit": 15,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use the per-output `mode` values in UNIT_TRAINING_OVERRIDES to decide "
            "which saved models are reused and which are retrained, then render "
            "the visualization gallery assets for both units."
        )
    )
    parser.add_argument(
        "--show-other-points",
        action="store_true",
        default=True,
        help="Also show off-slice training points in the surface plots.",
    )
    parser.add_argument(
        "--hide-other-points",
        action="store_false",
        dest="show_other_points",
        help="Hide off-slice training points in the surface plots.",
    )
    return parser


def _load_unit_training_frames(unit_name: str):
    if unit_name == "ammoniaF_unit":
        ammonia_case_grid = sf.AMMONIAF_CASE_GRID_CSV if sf.AMMONIAF_CASE_GRID_CSV.exists() else None
        ammonia_inputs = sf.AMMONIAF_INPUTS_CSV if sf.AMMONIAF_INPUTS_CSV.exists() else None
        X, Y, _ = sf.load_ammoniaF_training_data(
            results_csv=sf.AMMONIAF_RESULTS_CSV,
            case_grid_csv=ammonia_case_grid,
            inputs_csv=ammonia_inputs,
        )
        return X, Y

    if unit_name == "ureaF_unit":
        X, Y, _ = sf.load_ureaF_training_data(
            results_csv=sf.UREAF_RESULTS_CSV,
            include_all_heat_duties=sf.UREAF_TRAIN_ALL_HEAT_DUTIES,
        )
        return X, Y

    raise ValueError(f"Unsupported unit name: {unit_name}")


def _resolve_output_plan(
    unit_name: str,
    output_names: list[str],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    unit_overrides = UNIT_TRAINING_OVERRIDES.get(unit_name, {})
    unknown_outputs = sorted(set(unit_overrides) - set(output_names))
    if unknown_outputs:
        raise KeyError(
            f"Unknown configured outputs for {unit_name}: {unknown_outputs}. "
            f"Available outputs: {output_names}"
        )

    output_modes: dict[str, str] = {}
    output_training_overrides: dict[str, dict[str, Any]] = {}

    for output_name in output_names:
        raw_config = dict(unit_overrides.get(output_name, {}))
        mode = str(raw_config.pop("mode", DEFAULT_MODEL_MODE)).strip().lower()
        if mode not in VALID_MODEL_MODES:
            raise ValueError(
                f"Unsupported mode {mode!r} for {unit_name}.{output_name}. "
                f"Use one of {sorted(VALID_MODEL_MODES)}."
            )
        output_modes[output_name] = mode
        if raw_config:
            output_training_overrides[output_name] = raw_config
            if mode == "saved":
                print(
                    f"{unit_name}.{output_name} is set to saved; its layers/epochs "
                    "overrides will not be applied unless you switch that output "
                    "back to retrain."
                )

    return output_modes, output_training_overrides


def _load_saved_bundle_for_outputs(
    unit_name: str,
    output_names: list[str],
    expected_feature_names: list[str],
) -> sf.UnitSurrogateBundle:
    model_dir = Path(sf.DEFAULT_MODEL_DIR)
    bundle_path = model_dir / f"{unit_name}.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Saved bundle not found for {unit_name}: {bundle_path}. "
            "Set the missing outputs to retrain to build a fresh bundle."
        )

    try:
        bundle = sf._get_bundle(unit_name, model_dir)
    except Exception as exc:
        raise ValueError(
            f"Could not load the saved bundle for {unit_name} from {bundle_path}. "
            "Set the needed outputs to retrain to rebuild it."
        ) from exc

    if getattr(bundle, "model_kind", None) != "relu_ann":
        raise ValueError(
            f"The saved bundle for {unit_name} is not a compatible ReLU bundle. "
            "Set the needed outputs to retrain to rebuild it."
        )

    if list(bundle.feature_names) != list(expected_feature_names):
        raise ValueError(
            f"The saved bundle for {unit_name} uses inputs {bundle.feature_names}, "
            f"but the current training data uses {expected_feature_names}. "
            "Retrain every output in that unit to realign the bundle."
        )

    for output_name in output_names:
        if output_name not in bundle.models:
            raise KeyError(
                f"The saved bundle for {unit_name} is missing output {output_name!r}. "
                "Set that output to retrain to rebuild it."
            )
        if not isinstance(bundle.models[output_name], sf.tf.keras.Sequential):
            raise ValueError(
                f"The saved model for {unit_name}.{output_name} is not a compatible "
                "Keras Sequential model. Retrain that output to rebuild it."
            )
        output_metrics = (bundle.model_metrics or {}).get(output_name, {})
        if not output_metrics.get("train_loss_history"):
            raise ValueError(
                f"The saved model for {unit_name}.{output_name} does not contain "
                "training loss history. Retrain that output once so the HTML can "
                "show its loss curve."
            )

    return bundle


def _build_bundle_training_config(
    seed: int,
    per_output_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    defaults = {
        "hidden_layer_sizes": list(sf.ANN_HIDDEN_LAYER_SIZES),
        "learning_rate": sf.ANN_LEARNING_RATE,
        "max_epochs": sf.ANN_MAX_EPOCHS,
        "patience": sf.ANN_PATIENCE,
        "validation_split": sf.ANN_VALIDATION_SPLIT,
        "batch_size": sf.ANN_BATCH_SIZE,
    }
    return {
        **defaults,
        "seed": seed,
        "defaults": dict(defaults),
        "per_output": {
            output_name: sf._serialize_output_training_config(
                sf._normalize_output_training_config(config)
            )
            for output_name, config in per_output_configs.items()
        },
    }


def _train_full_unit_bundle(
    unit_name: str,
    output_training_overrides: dict[str, dict[str, Any]],
) -> None:
    seed = UNIT_TRAINING_SEEDS[unit_name]
    print(f"Retraining all {unit_name} outputs...")

    if unit_name == "ammoniaF_unit":
        sf.train_and_save_ammoniaF_surrogate(
            model_dir=sf.DEFAULT_MODEL_DIR,
            results_csv=sf.AMMONIAF_RESULTS_CSV,
            case_grid_csv=sf.AMMONIAF_CASE_GRID_CSV,
            inputs_csv=sf.AMMONIAF_INPUTS_CSV,
            seed=seed,
            output_training_overrides=output_training_overrides,
        )
        return

    if unit_name == "ureaF_unit":
        sf.train_and_save_ureaF_surrogate(
            model_dir=sf.DEFAULT_MODEL_DIR,
            results_csv=sf.UREAF_RESULTS_CSV,
            include_all_heat_duties=sf.UREAF_TRAIN_ALL_HEAT_DUTIES,
            seed=seed,
            output_training_overrides=output_training_overrides,
        )
        return

    raise ValueError(f"Unsupported unit name: {unit_name}")


def sync_unit_bundle(unit_name: str) -> None:
    X, Y = _load_unit_training_frames(unit_name)
    output_names = list(Y.columns)
    output_modes, output_training_overrides = _resolve_output_plan(unit_name, output_names)
    outputs_to_retrain = [name for name in output_names if output_modes[name] == "retrain"]
    outputs_to_keep = [name for name in output_names if output_modes[name] == "saved"]

    if not outputs_to_retrain:
        _load_saved_bundle_for_outputs(unit_name, output_names, list(X.columns))
        print(f"Using saved models for all {unit_name} outputs.")
        return

    if len(outputs_to_retrain) == len(output_names):
        _train_full_unit_bundle(unit_name, output_training_overrides)
        return

    print(
        f"Retraining {unit_name} outputs {outputs_to_retrain} and reusing saved models "
        f"for {outputs_to_keep}."
    )

    seed = UNIT_TRAINING_SEEDS[unit_name]
    base_bundle = _load_saved_bundle_for_outputs(unit_name, outputs_to_keep, list(X.columns))
    resolved_output_configs = sf._resolve_output_training_configs(
        unit_name,
        output_names,
        output_training_overrides=output_training_overrides,
    )

    merged_models = dict(base_bundle.models)
    merged_metrics = dict(base_bundle.model_metrics or {})
    per_output_configs = {
        output_name: sf._bundle_output_training_config(base_bundle, output_name)
        for output_name in outputs_to_keep
    }

    for output_index, output_name in enumerate(output_names):
        if output_name not in outputs_to_retrain:
            continue
        model_seed = seed + output_index + 1
        model, metrics = sf._train_single_output_model(
            X,
            Y[output_name],
            seed=model_seed,
            training_config=resolved_output_configs[output_name],
        )
        merged_models[output_name] = model
        merged_metrics[output_name] = metrics
        per_output_configs[output_name] = resolved_output_configs[output_name]

    bundle = sf.UnitSurrogateBundle(
        unit_name=unit_name,
        feature_names=list(X.columns),
        output_names=output_names,
        models={output_name: merged_models[output_name] for output_name in output_names},
        input_bounds=sf._resolve_input_bounds(unit_name, X),
        output_bounds={
            output_name: sf._nondegenerate_bounds(Y[output_name])
            for output_name in output_names
        },
        model_metrics={
            output_name: merged_metrics[output_name] for output_name in output_names
        },
        training_config=_build_bundle_training_config(seed, per_output_configs),
        model_kind="relu_ann",
    )
    model_dir = Path(sf.DEFAULT_MODEL_DIR)
    sf._save_bundle(bundle, model_dir)
    sf._MODEL_CACHE.pop(f"{model_dir.resolve()}::{unit_name}", None)
    sf._print_unit_validation_report(unit_name, bundle, X, Y)


def main() -> None:
    args = build_parser().parse_args()

    for unit_name in ("ammoniaF_unit", "ureaF_unit"):
        sync_unit_bundle(unit_name)

    for unit_name in ("ureaF_unit", "ammoniaF_unit"):
        plotter.run_visualizations(
            unit=unit_name,
            model_dir=sf.DEFAULT_MODEL_DIR,
            show_other_points=args.show_other_points,
            bundle_mode="saved",
        )


if __name__ == "__main__":
    main()

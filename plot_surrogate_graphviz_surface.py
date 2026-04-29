"""
Standalone visualizer for the ReLU surrogate bundles.

Example:
    python plot_surrogate_graphviz_surface.py --unit ammoniaF_unit --output ammonia_kgph --x Ft --y Fh2
    python plot_surrogate_graphviz_surface.py --unit ureaF_unit --output pure_urea_kgph --x Fnh3 --y Fco2
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from graphviz import Digraph
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import surrogate_functions as sf


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = THIS_DIR / "surrogate_visualizations"
ISOMETRIC_ELEVATION_DEG = 20.264389682754654
ISOMETRIC_AZIMUTH_DEG = -45
SURFACE_CAMERA_ELEVATION_DEG = ISOMETRIC_ELEVATION_DEG
SURFACE_CAMERA_AZIMUTH_DEG = ISOMETRIC_AZIMUTH_DEG
SURFACE_CAMERA_ROLL_DEG = 0
SURFACE_AXIS_LABEL_FONTSIZE = 18
SURFACE_TICK_LABEL_FONTSIZE = 12
SURFACE_Z_LABELPAD = 28
SURFACE_EXTERNAL_Z_LABEL_X = 0.90
SURFACE_EXTERNAL_Z_LABEL_Y = 0.52
SURFACE_POINT_SIZE = 2.0
SURFACE_POINT_ALPHA = 0.28
DEFAULT_VIEW = {
    "ammoniaF_unit": {
        "output": "ammonia_kgph",
        "x": "Ft",
        "y": "Fh2",
    },
    "ureaF_unit": {
        "output": "pure_urea_kgph",
        "x": "Fnh3",
        "y": "Fco2",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render Graphviz model diagrams plus data-points-vs-surface views "
            "for one or many trained surrogate outputs."
        )
    )
    parser.add_argument(
        "--unit",
        choices=["ammoniaF_unit", "ureaF_unit"],
        default="ammoniaF_unit",
        help="Which unit surrogate to visualize.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output filter. If omitted, every output is plotted.",
    )
    parser.add_argument(
        "--x",
        default=None,
        help="Optional X feature filter. Use together with --y to plot one pair only.",
    )
    parser.add_argument(
        "--y",
        default=None,
        help="Optional Y feature filter. Use together with --x to plot one pair only.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=sf.DEFAULT_MODEL_DIR,
        help="Directory containing saved surrogate bundles.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Root directory where the figures and Graphviz files will be written.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=75,
        help="Resolution of the 2D slice used to build the surface.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1500,
        help="Maximum number of training points to show in the 3D scatter.",
    )
    parser.add_argument(
        "--show-other-points",
        action="store_true",
        help=(
            "If set, also plot the training points away from the fixed-input slice "
            "as black dots."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for point subsampling.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="If set, show each surface plot interactively as it is generated.",
    )
    parser.add_argument(
        "--bundle-mode",
        choices=["auto", "saved", "retrain"],
        default="auto",
        help=(
            "How to obtain the surrogate bundle before plotting: "
            "`auto` uses the saved bundle and retrains only if needed, "
            "`saved` requires an existing compatible bundle, and "
            "`retrain` always retrains first."
        ),
    )
    return parser


def _load_training_bundle(
    unit_name: str,
    model_dir: Path,
    bundle_mode: str = "auto",
) -> Tuple[sf.UnitSurrogateBundle, pd.DataFrame, pd.DataFrame]:
    def _bundle_is_relu(bundle: sf.UnitSurrogateBundle) -> bool:
        if getattr(bundle, "model_kind", None) != "relu_ann":
            return False
        return all(
            isinstance(model, sf.tf.keras.Sequential)
            for model in getattr(bundle, "models", {}).values()
        )

    def _bundle_has_loss_history(bundle: sf.UnitSurrogateBundle) -> bool:
        model_metrics = getattr(bundle, "model_metrics", {}) or {}
        expected_outputs = getattr(bundle, "output_names", []) or []
        return all(
            model_metrics.get(output_name, {}).get("train_loss_history")
            for output_name in expected_outputs
        )

    def _load_saved_bundle_or_raise(bundle_path: Path, bundle_name: str) -> sf.UnitSurrogateBundle:
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Saved bundle not found for {bundle_name}: {bundle_path}. "
                "Use retrain mode to build it first."
            )
        try:
            bundle = sf._get_bundle(bundle_name, model_dir)
        except Exception as exc:
            raise ValueError(
                f"Could not load the saved bundle for {bundle_name} from {bundle_path}. "
                "Use retrain mode to rebuild it."
            ) from exc
        if not _bundle_is_relu(bundle):
            raise ValueError(
                f"The saved bundle for {bundle_name} is not a compatible ReLU bundle. "
                "Use retrain mode to rebuild it."
            )
        if not _bundle_has_loss_history(bundle):
            raise ValueError(
                f"The saved bundle for {bundle_name} does not contain training loss history. "
                "Use retrain mode once to regenerate it so the HTML can show loss curves."
            )
        return bundle

    bundle_mode = str(bundle_mode).strip().lower()
    if bundle_mode not in {"auto", "saved", "retrain"}:
        raise ValueError(
            f"Unsupported bundle_mode={bundle_mode!r}. Use 'auto', 'saved', or 'retrain'."
        )

    if unit_name == "ammoniaF_unit":
        X, Y, _ = sf.load_ammoniaF_training_data(
            results_csv=sf.AMMONIAF_RESULTS_CSV,
            case_grid_csv=sf.AMMONIAF_CASE_GRID_CSV if sf.AMMONIAF_CASE_GRID_CSV.exists() else None,
            inputs_csv=sf.AMMONIAF_INPUTS_CSV if sf.AMMONIAF_INPUTS_CSV.exists() else None,
        )
        bundle_path = model_dir / "ammoniaF_unit.joblib"
        if bundle_mode == "retrain":
            bundle = sf.train_and_save_ammoniaF_surrogate(model_dir=model_dir)
        elif bundle_mode == "saved":
            bundle = _load_saved_bundle_or_raise(bundle_path, "ammoniaF_unit")
        elif bundle_path.exists():
            try:
                bundle = sf._get_bundle("ammoniaF_unit", model_dir)
            except Exception:
                bundle = sf.train_and_save_ammoniaF_surrogate(model_dir=model_dir)
            else:
                if not _bundle_is_relu(bundle) or not _bundle_has_loss_history(bundle):
                    bundle = sf.train_and_save_ammoniaF_surrogate(model_dir=model_dir)
        else:
            bundle = sf.train_and_save_ammoniaF_surrogate(model_dir=model_dir)
        return bundle, X, Y

    X, Y, _ = sf.load_ureaF_training_data(
        results_csv=sf.UREAF_RESULTS_CSV,
        include_all_heat_duties=sf.UREAF_TRAIN_ALL_HEAT_DUTIES,
    )
    bundle_path = model_dir / "ureaF_unit.joblib"
    if bundle_mode == "retrain":
        bundle = sf.train_and_save_ureaF_surrogate(model_dir=model_dir)
    elif bundle_mode == "saved":
        bundle = _load_saved_bundle_or_raise(bundle_path, "ureaF_unit")
    elif bundle_path.exists():
        try:
            bundle = sf._get_bundle("ureaF_unit", model_dir)
        except Exception:
            bundle = sf.train_and_save_ureaF_surrogate(model_dir=model_dir)
        else:
            if not _bundle_is_relu(bundle) or not _bundle_has_loss_history(bundle):
                bundle = sf.train_and_save_ureaF_surrogate(model_dir=model_dir)
    else:
        bundle = sf.train_and_save_ureaF_surrogate(model_dir=model_dir)
    return bundle, X, Y


def _resolve_requested_outputs(
    unit_name: str,
    bundle: sf.UnitSurrogateBundle,
    output_name: str | None,
) -> list[str]:
    if output_name is None:
        return list(bundle.output_names)
    if output_name not in bundle.output_names:
        raise ValueError(
            f"{output_name!r} is not available for {unit_name}. "
            f"Available outputs: {bundle.output_names}"
        )
    return [output_name]


def _resolve_requested_pairs(
    unit_name: str,
    bundle: sf.UnitSurrogateBundle,
    x_name: str | None,
    y_name: str | None,
) -> list[tuple[str, str]]:
    if x_name is None and y_name is None:
        return list(combinations(bundle.feature_names, 2))
    if x_name is None or y_name is None:
        raise ValueError("Use --x and --y together, or omit both to plot every feature pair.")
    for axis_name in (x_name, y_name):
        if axis_name not in bundle.feature_names:
            raise ValueError(
                f"{axis_name!r} is not an input feature for {unit_name}. "
                f"Available features: {bundle.feature_names}"
            )
    if x_name == y_name:
        raise ValueError("--x and --y must refer to different features.")
    return [(x_name, y_name)]


def _make_output_dir(root: Path, unit_name: str, output_name: str, x_name: str, y_name: str) -> Path:
    out_dir = root / unit_name / output_name / f"{x_name}__{y_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": float(
            np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0
        ),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def _slice_grid(
    feature_names: list[str],
    baseline: Dict[str, float],
    x_name: str,
    y_name: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_df = pd.DataFrame(
        {
            name: np.full(xx.size, baseline[name], dtype=float)
            for name in feature_names
        }
    )
    grid_df[x_name] = xx.ravel()
    grid_df[y_name] = yy.ravel()
    return xx, yy, grid_df


def _slice_mask_for_feature(values: np.ndarray, target: float) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros(values.shape, dtype=bool)

    unique_values = np.unique(finite_values)
    if unique_values.size > 1:
        deltas = np.diff(np.sort(unique_values))
        positive_deltas = deltas[deltas > 0]
        step = float(np.min(positive_deltas)) if positive_deltas.size else 0.0
    else:
        step = 0.0

    atol = max(step / 2.0 if step > 0 else 0.0, float(np.ptp(finite_values)) * 1e-6, 1e-8)
    mask = np.isclose(values, target, atol=atol, rtol=0.0)
    if np.any(mask):
        return mask

    nearest_value = float(unique_values[np.argmin(np.abs(unique_values - target))])
    return np.isclose(values, nearest_value, atol=atol, rtol=0.0)


def _fixed_slice_mask(X: pd.DataFrame, fixed_inputs: Dict[str, float]) -> np.ndarray:
    if not fixed_inputs:
        return np.ones(len(X), dtype=bool)

    mask = np.ones(len(X), dtype=bool)
    for feature_name, fixed_value in fixed_inputs.items():
        values = X[feature_name].to_numpy(dtype=float)
        mask &= _slice_mask_for_feature(values, fixed_value)
    return mask


def render_graphviz_network(
    model,
    feature_names: list[str],
    output_name: str,
    out_dir: Path,
) -> Path:
    dot = Digraph(name=f"{output_name}_network", format="png")
    dot.attr(rankdir="LR", splines="line", overlap="false", nodesep="0.28", ranksep="0.7")

    input_nodes: list[str] = []
    with dot.subgraph(name="cluster_inputs") as sub:
        sub.attr(label="Inputs", color="#cbd5e1")
        for idx, feature_name in enumerate(feature_names):
            node_id = f"in_{idx}"
            sub.node(
                node_id,
                feature_name,
                shape="box",
                style="rounded,filled",
                fillcolor="#dbeafe",
                color="#60a5fa",
            )
            input_nodes.append(node_id)

    hidden_layers = [
        layer
        for layer in model.layers[:-1]
        if isinstance(layer, sf.tf.keras.layers.Dense)
    ]
    previous_nodes = input_nodes
    for layer_idx, layer in enumerate(hidden_layers, start=1):
        current_nodes: list[str] = []
        with dot.subgraph(name=f"cluster_hidden_{layer_idx}") as sub:
            sub.attr(label=f"Hidden {layer_idx}\\nDense({layer.units}, ReLU)", color="#d1d5db")
            for node_idx in range(layer.units):
                node_id = f"h{layer_idx}_{node_idx}"
                sub.node(
                    node_id,
                    "",
                    shape="circle",
                    width="0.18",
                    height="0.18",
                    style="filled",
                    fillcolor="#fde68a",
                    color="#f59e0b",
                )
                current_nodes.append(node_id)
        for src in previous_nodes:
            for dst in current_nodes:
                dot.edge(src, dst, color="#9ca3af", penwidth="0.35")
        previous_nodes = current_nodes

    output_node = "out_0"
    dot.node(
        output_node,
        output_name,
        shape="box",
        style="rounded,filled",
        fillcolor="#dcfce7",
        color="#22c55e",
    )
    for src in previous_nodes:
        dot.edge(src, output_node, color="#9ca3af", penwidth="0.35")

    dot_path = out_dir / "network_graph.dot"
    dot.save(dot_path)
    png_path = out_dir / "network_graph.png"
    png_path.write_bytes(dot.pipe(format="png"))
    return png_path


def render_surface_plot(
    model,
    X: pd.DataFrame,
    y_true: np.ndarray,
    x_name: str,
    y_name: str,
    out_dir: Path,
    output_name: str,
    grid_points: int,
    max_points: int,
    seed: int,
    show_other_points: bool = False,
    interactive: bool = False,
) -> Tuple[Path, np.ndarray, Dict[str, Any]]:
    x_grid = np.linspace(float(X[x_name].min()), float(X[x_name].max()), grid_points)
    y_grid = np.linspace(float(X[y_name].min()), float(X[y_name].max()), grid_points)
    baseline = {name: float(X[name].median()) for name in X.columns}
    xx, yy, grid_df = _slice_grid(list(X.columns), baseline, x_name, y_name, x_grid, y_grid)
    zz = sf._keras_predict(model, grid_df[X.columns].to_numpy(dtype=np.float32)).reshape(xx.shape)

    train_features = X.to_numpy(dtype=np.float32)
    y_pred = sf._keras_predict(model, train_features).reshape(-1)

    fixed_inputs = {name: value for name, value in baseline.items() if name not in {x_name, y_name}}
    fixed_slice_mask = _fixed_slice_mask(X, fixed_inputs)
    slice_indices = np.flatnonzero(fixed_slice_mask)
    other_indices = np.flatnonzero(~fixed_slice_mask)

    rng = np.random.default_rng(seed)
    if len(slice_indices) > max_points:
        slice_indices = np.sort(rng.choice(slice_indices, size=max_points, replace=False))

    if show_other_points:
        remaining_budget = max(max_points - len(slice_indices), 0)
        if len(other_indices) > remaining_budget:
            other_indices = np.sort(rng.choice(other_indices, size=remaining_budget, replace=False))
    else:
        other_indices = np.array([], dtype=int)

    displayed_indices = np.sort(np.concatenate([slice_indices, other_indices]))
    displayed = X.iloc[displayed_indices]
    displayed_y = y_true[displayed_indices]
    slice_points = X.iloc[slice_indices]
    slice_y = y_true[slice_indices]
    other_points = X.iloc[other_indices]
    other_y = y_true[other_indices]

    def _point_payload(points: pd.DataFrame, z_values: np.ndarray) -> Dict[str, list[float]]:
        return {
            "x": points[x_name].to_numpy(dtype=float).tolist(),
            "y": points[y_name].to_numpy(dtype=float).tolist(),
            "z": np.asarray(z_values, dtype=float).tolist(),
        }

    surface_data = {
        "x_name": x_name,
        "y_name": y_name,
        "z_name": output_name,
        "x_grid": x_grid.tolist(),
        "y_grid": y_grid.tolist(),
        "z_grid": zz.tolist(),
        "sampled_points": _point_payload(displayed, displayed_y),
        "slice_points": _point_payload(slice_points, slice_y),
        "other_points": _point_payload(other_points, other_y),
        "fixed_inputs": fixed_inputs,
        "show_other_points": show_other_points,
    }

    fig = plt.figure(figsize=(9.5, 7.0))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor("#d1d5db")
        axis._axinfo["grid"]["color"] = (0.82, 0.86, 0.91, 0.55)
        axis._axinfo["grid"]["linewidth"] = 0.6
        axis.set_tick_params(labelsize=SURFACE_TICK_LABEL_FONTSIZE)
    ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True, alpha=0.82)
    if len(other_points) > 0:
        ax.scatter(
            other_points[x_name].to_numpy(dtype=float),
            other_points[y_name].to_numpy(dtype=float),
            other_y,
            c="#64748b",
            s=SURFACE_POINT_SIZE,
            alpha=SURFACE_POINT_ALPHA,
            depthshade=False,
            label="Other training datapoints",
        )
    if len(slice_points) > 0:
        ax.scatter(
            slice_points[x_name].to_numpy(dtype=float),
            slice_points[y_name].to_numpy(dtype=float),
            slice_y,
            c="#475569",
            s=SURFACE_POINT_SIZE,
            alpha=0.38,
            depthshade=False,
            label="Points on fixed slice",
        )
    ax.set_xlabel(x_name, fontsize=SURFACE_AXIS_LABEL_FONTSIZE, fontweight="bold", labelpad=18)
    ax.set_ylabel(y_name, fontsize=SURFACE_AXIS_LABEL_FONTSIZE, fontweight="bold", labelpad=18)
    ax.set_zlabel("")
    fig.text(
        SURFACE_EXTERNAL_Z_LABEL_X,
        SURFACE_EXTERNAL_Z_LABEL_Y,
        output_name,
        rotation=90,
        va="center",
        ha="center",
        fontsize=SURFACE_AXIS_LABEL_FONTSIZE,
        fontweight="bold",
        color="#0f172a",
    )
    ax.tick_params(axis="both", which="major", labelsize=SURFACE_TICK_LABEL_FONTSIZE, pad=4)
    try:
        ax.view_init(
            elev=SURFACE_CAMERA_ELEVATION_DEG,
            azim=SURFACE_CAMERA_AZIMUTH_DEG,
            roll=SURFACE_CAMERA_ROLL_DEG,
        )
    except TypeError:
        ax.view_init(elev=SURFACE_CAMERA_ELEVATION_DEG, azim=SURFACE_CAMERA_AZIMUTH_DEG)
    fig.tight_layout()
    out_path = out_dir / "surface_vs_datapoints.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    if interactive:
        plt.show(block=True)
    plt.close(fig)
    return out_path, y_pred, surface_data


def render_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    output_name: str,
    metrics: Dict[str, float],
) -> Path:
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(y_true, y_pred, s=8, alpha=0.28, color="#64748b", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel("Actual", fontsize=16, fontweight="bold")
    ax.set_ylabel("Predicted", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_title(f"{output_name}: parity plot", fontsize=13, pad=12)
    ax.text(
        0.03,
        0.97,
        f"MAE  = {metrics['mae']:.4g}\nRMSE = {metrics['rmse']:.4g}\nMAPE = {metrics['mape_pct']:.2f}%\nR2   = {metrics['r2']:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    out_path = out_dir / "parity_plot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_training_loss_plot(
    bundle: sf.UnitSurrogateBundle,
    output_name: str,
    out_dir: Path,
) -> Path | None:
    training_metrics = (bundle.model_metrics or {}).get(output_name, {})
    train_loss = np.asarray(training_metrics.get("train_loss_history", []), dtype=float)
    val_loss = np.asarray(training_metrics.get("val_loss_history", []), dtype=float)

    if train_loss.size == 0 and val_loss.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if train_loss.size > 0:
        ax.plot(
            np.arange(1, train_loss.size + 1),
            train_loss,
            color="#2563eb",
            linewidth=1.8,
            label="Train loss",
        )
    if val_loss.size > 0:
        ax.plot(
            np.arange(1, val_loss.size + 1),
            val_loss,
            color="#dc2626",
            linewidth=1.8,
            label="Validation loss",
        )

    positive_values = np.concatenate(
        [values[values > 0] for values in (train_loss, val_loss) if values.size > 0]
    )
    if positive_values.size > 0:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch", fontsize=15, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.set_title(f"{output_name}: training loss curve", fontsize=13, pad=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    if train_loss.size > 0 or val_loss.size > 0:
        ax.legend()
    fig.tight_layout()

    out_path = out_dir / "training_loss_curve.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def run_visualizations(
    unit: str,
    output: str | None = None,
    x: str | None = None,
    y: str | None = None,
    model_dir: Path | str = sf.DEFAULT_MODEL_DIR,
    out_dir: Path | str = DEFAULT_OUT_ROOT,
    grid_points: int = 75,
    max_points: int = 1500,
    show_other_points: bool = False,
    seed: int = 42,
    interactive: bool = False,
    bundle_mode: str = "auto",
) -> list[Dict[str, Any]]:
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    bundle, X, Y = _load_training_bundle(unit, model_dir, bundle_mode=bundle_mode)
    requested_outputs = _resolve_requested_outputs(unit, bundle, output)
    requested_pairs = _resolve_requested_pairs(unit, bundle, x, y)

    run_summaries: list[Dict[str, Any]] = []
    feature_frame = X[list(bundle.feature_names)].copy()

    for output_name in requested_outputs:
        model = bundle.models[output_name]
        y_true = Y[output_name].to_numpy(dtype=float).reshape(-1)
        y_pred_full = sf._keras_predict(model, feature_frame.to_numpy(dtype=np.float32)).reshape(-1)
        metrics = _prediction_metrics(y_true, y_pred_full)

        for x_name, y_name in requested_pairs:
            pair_out_dir = _make_output_dir(out_dir, unit, output_name, x_name, y_name)
            network_png = render_graphviz_network(
                model=model,
                feature_names=list(bundle.feature_names),
                output_name=output_name,
                out_dir=pair_out_dir,
            )
            surface_png, _, surface_data = render_surface_plot(
                model=model,
                X=feature_frame,
                y_true=y_true,
                x_name=x_name,
                y_name=y_name,
                out_dir=pair_out_dir,
                output_name=output_name,
                grid_points=grid_points,
                max_points=max_points,
                seed=seed,
                show_other_points=show_other_points,
                interactive=interactive,
            )
            parity_png = render_parity_plot(
                y_true=y_true,
                y_pred=y_pred_full,
                out_dir=pair_out_dir,
                output_name=output_name,
                metrics=metrics,
            )
            training_loss_png = render_training_loss_plot(
                bundle=bundle,
                output_name=output_name,
                out_dir=pair_out_dir,
            )

            summary = {
                "unit": unit,
                "output": output_name,
                "x_feature": x_name,
                "y_feature": y_name,
                "feature_names": list(bundle.feature_names),
                "metrics": metrics,
                "training_metrics": (bundle.model_metrics or {}).get(output_name, {}),
                "training_config": sf._bundle_output_training_config(bundle, output_name),
                "network_graph_png": str(network_png),
                "surface_plot_png": str(surface_png),
                "parity_plot_png": str(parity_png),
                "training_loss_png": str(training_loss_png) if training_loss_png else "",
                "surface_data": surface_data,
            }
            with (pair_out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            run_summaries.append(summary)
            print(f"[{output_name} | {x_name} vs {y_name}]")
            print(f"  Graphviz: {network_png}")
            print(f"  Surface : {surface_png}")
            print(f"  Parity  : {parity_png}")

    unit_summary_path = out_dir / unit / "run_summary.json"
    unit_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with unit_summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summaries, f, indent=2)

    print(f"Saved run summary: {unit_summary_path}")
    return run_summaries


def main() -> None:
    args = build_parser().parse_args()
    run_visualizations(
        unit=args.unit,
        output=args.output,
        x=args.x,
        y=args.y,
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        grid_points=args.grid_points,
        max_points=args.max_points,
        show_other_points=args.show_other_points,
        seed=args.seed,
        interactive=args.interactive,
        bundle_mode=args.bundle_mode,
    )


if __name__ == "__main__":
    main()

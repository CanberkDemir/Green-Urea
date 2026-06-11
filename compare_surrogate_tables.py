"""
compare_surrogate_tables.py

Build comparison tables between the ridge HyperplaneTree run
(HT_train_surrogates.py) and the elastic-net HyperplaneTree run
(EN_train_surrogates.py).

Reads the phase_plot_metrics.csv files written by both runs, so it works
regardless of how the training scripts were launched. Run both trainers
first, then:

    python compare_surrogate_tables.py

Prints two tables (sparsity and accuracy) and writes them as CSVs into
surrogate_comparison_tables\.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RIDGE_PLOT_ROOT = Path("trained_unit_surrogates") / "plots_phase"
EN_PLOT_ROOT = Path("trained_unit_surrogates_en") / "plots_phase"
OUT_DIR = Path("surrogate_comparison_tables")

UNITS = ["ammoniaF_unit", "ureaF_unit"]


def _load_run_metrics(plot_root: Path, label: str) -> pd.DataFrame:
    frames = []
    for unit in UNITS:
        csv_path = plot_root / unit / "phase_plot_metrics.csv"
        if not csv_path.exists():
            print(f"WARNING: missing {csv_path}; skipping {unit} for {label}.")
            continue
        frames.append(pd.read_csv(csv_path))
    if not frames:
        raise FileNotFoundError(
            f"No phase_plot_metrics.csv files found under {plot_root}. "
            "Run the corresponding training script first."
        )
    df = pd.concat(frames, ignore_index=True)
    keep = ["unit", "output", "r2", "mae", "mape", "n_leaves"]
    for col in ["n_zero_coefs", "n_total_coefs"]:
        keep.append(col)
        if col not in df.columns:
            df[col] = pd.NA
    df = df[keep].copy()
    return df.rename(columns={col: f"{col}_{label}" for col in df.columns if col not in {"unit", "output"}})


def main() -> None:
    ridge = _load_run_metrics(RIDGE_PLOT_ROOT, "ridge")
    en = _load_run_metrics(EN_PLOT_ROOT, "en")
    merged = ridge.merge(en, on=["unit", "output"], how="outer")

    def zero_total(row, label):
        z, t = row[f"n_zero_coefs_{label}"], row[f"n_total_coefs_{label}"]
        if pd.isna(z) or pd.isna(t):
            return "n/a"
        return f"{int(z)} / {int(t)}"

    sparsity = pd.DataFrame({
        "unit": merged["unit"],
        "output": merged["output"],
        "leaves_ridge": merged["n_leaves_ridge"],
        "leaves_en": merged["n_leaves_en"],
        "zero/total_coefs_ridge": merged.apply(zero_total, axis=1, label="ridge"),
        "zero/total_coefs_en": merged.apply(zero_total, axis=1, label="en"),
        "sparsity_pct_en": (
            100.0 * merged["n_zero_coefs_en"] / merged["n_total_coefs_en"]
        ).round(1),
    })

    accuracy = pd.DataFrame({
        "unit": merged["unit"],
        "output": merged["output"],
        "r2_ridge": merged["r2_ridge"].round(6),
        "r2_en": merged["r2_en"].round(6),
        "mae_ridge": merged["mae_ridge"],
        "mae_en": merged["mae_en"],
        "total_coefs_ridge": merged["n_total_coefs_ridge"],
        "total_coefs_en": merged["n_total_coefs_en"],
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sparsity.to_csv(OUT_DIR / "sparsity_table.csv", index=False)
    accuracy.to_csv(OUT_DIR / "accuracy_table.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== Sparsity: leaves and zero coefficients (ridge HT vs EN-HT) ===")
    print(sparsity.to_string(index=False))
    print("\n=== Accuracy: R^2 / MAE and total coefficients (full data) ===")
    print(accuracy.to_string(index=False))
    print(f"\nTables written to {OUT_DIR.resolve()}")
    print(
        "Note: R^2/MAE here are computed on the full filtered dataset "
        "(as in phase_plot_metrics.csv), not the held-out test split."
    )


if __name__ == "__main__":
    main()

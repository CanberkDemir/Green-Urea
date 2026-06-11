"""
EN_train_surrogates.py

Elastic-net-regularized HyperplaneTree surrogates.

This is the same training/plotting pipeline as HT_train_surrogates.py with
ONE change: the leaf regression inside the HyperplaneTree uses elastic-net
regularization (`leaf_regularization="elasticnet"`) instead of the default
ridge. This requires the `feature/sparse-regularization` branch of
https://github.com/SunoySanyal/systems2atoms.

All other hyperparameters (max_weight search, depth, bins, seeds, splits,
operating-window filters) are inherited from HT_train_surrogates.py.

Outputs are kept separate from the ridge HT run:
    trained_unit_surrogates_en/                 trained bundles + surface plots
    surrogate_visualizations_en/                parity plots + gallery summaries

Run:
    python EN_train_surrogates.py
"""

from __future__ import annotations

from pathlib import Path

import HT_train_surrogates as ht

# =============================================================================
# CONFIG — only the regularization differs from the ridge HT run
# =============================================================================

EN_MODEL_DIR = Path("trained_unit_surrogates_en")
EN_GALLERY_OUT_DIR = Path("surrogate_visualizations_en")

EN_LEAF_L1_RATIO = 0.5
# leaf_alpha=None makes the library fall back to the `ridge` value (1e-5)
# as the elastic-net alpha, i.e. truly "same settings, EN instead of ridge".
EN_LEAF_ALPHA = 5e-3

ht.HT_DEFAULT_HYPERPARAMETERS.update({
    "leaf_regularization": "elasticnet",
    "leaf_alpha": EN_LEAF_ALPHA,
    "leaf_l1_ratio": EN_LEAF_L1_RATIO,
})


# =============================================================================
# MAIN — mirrors HT_train_surrogates.__main__ with separate output dirs
# =============================================================================

if __name__ == "__main__":
    EN_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if ht.AMMONIAF_RESULTS_CSV.exists():
        print(f"Training ammoniaF EN-HT surrogate from: {ht.AMMONIAF_RESULTS_CSV.resolve()}")
        ammonia_bundle = ht.train_and_save_ammoniaF_surrogate(
            model_dir=EN_MODEL_DIR,
            results_csv=ht.AMMONIAF_RESULTS_CSV,
            case_grid_csv=None,
            inputs_csv=ht.AMMONIAF_INPUTS_CSV if ht.AMMONIAF_INPUTS_CSV.exists() else None,
        )
        print(
            f"Loaded ammoniaF training data with "
            f"{len(ammonia_bundle.feature_names)} inputs and "
            f"{len(ammonia_bundle.output_names)} outputs."
        )
        ht.save_ammoniaF_tree_plots(
            model_dir=EN_MODEL_DIR,
            results_csv=ht.AMMONIAF_RESULTS_CSV,
            case_grid_csv=None,
            inputs_csv=ht.AMMONIAF_INPUTS_CSV if ht.AMMONIAF_INPUTS_CSV.exists() else None,
            show_3d_plots=ht.SHOW_INTERACTIVE_3D_PLOTS,
            gallery_out_dir=EN_GALLERY_OUT_DIR,
        )
        print(f"\nAmmoniaF EN-HT plots saved under: {EN_MODEL_DIR / 'plots_phase' / 'ammoniaF_unit'}")
        print(f"AmmoniaF EN-HT gallery saved under: {EN_GALLERY_OUT_DIR / 'ammoniaF_unit_ht' / 'run_summary.json'}")

    if ht.UREAF_RESULTS_CSV.exists():
        print(f"\nTraining ureaF EN-HT surrogate from: {ht.UREAF_RESULTS_CSV.resolve()}")
        ureaF_bundle = ht.train_and_save_ureaF_surrogate(
            model_dir=EN_MODEL_DIR,
            results_csv=ht.UREAF_RESULTS_CSV,
            include_all_heat_duties=ht.UREAF_TRAIN_ALL_HEAT_DUTIES,
        )
        print(
            f"Loaded ureaF training data with "
            f"{len(ureaF_bundle.feature_names)} inputs and "
            f"{len(ureaF_bundle.output_names)} outputs."
        )
        ht.save_ureaF_tree_plots(
            model_dir=EN_MODEL_DIR,
            results_csv=ht.UREAF_RESULTS_CSV,
            include_all_heat_duties=ht.UREAF_TRAIN_ALL_HEAT_DUTIES,
            show_3d_plots=ht.SHOW_INTERACTIVE_3D_PLOTS,
            gallery_out_dir=EN_GALLERY_OUT_DIR,
        )
        print(f"\nUreaF EN-HT plots saved under: {EN_MODEL_DIR / 'plots_phase' / 'ureaF_unit'}")
        print(f"UreaF EN-HT gallery saved under: {EN_GALLERY_OUT_DIR / 'ureaF_unit_ht' / 'run_summary.json'}")

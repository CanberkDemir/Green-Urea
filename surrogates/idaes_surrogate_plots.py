"""
Surface plots of the IDAES training data and the reluMIP-ready ReLU fits
for IPPS sections A, B, C. If optimization/solutions/ipps_idaes_oppoint.json
exists (from optimization/ipps_idaes.py), the IPPS operating point of each
grid mode is marked on every surface.

    python idaes_surrogate_plots.py
Writes figures/idaes_surrogate_{A,B,C}.png.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "surrogates" / "idaes_data"
NET = ROOT / "surrogates" / "trained" / "idaes_relumip"
OP = ROOT / "optimization" / "solutions" / "ipps_idaes_oppoint.json"
FIG = ROOT / "figures"

TITLES = {"A": "A  ammonia section (IDAES ammonia_flowsheet)",
          "B": "B  NH3 recovery (vacuum stripper + rectifier)",
          "C": "C  urea section (IDAES ureaF replica)",
          "D": "D  low-pressure DEF plant (urea section used by the IPPS)"}
LABEL = {"Ft": "Ft brine [kg/h]", "Fh2": "Fh2 [kg/h]",
         "ammonia_kgph": "aq. NH3 [kg/h]", "water_kgph": "water [kg/h]",
         "electric_kwhph": "electricity [kW]",
         "anhydrous_nh3_kgph": "anhydrous NH3 [kg/h]",
         "Fnh3": "Fnh3 [kg/h]", "Fco2": "Fco2 [kg/h]",
         "pure_urea_kgph": "urea [kg/h]",
         "product_urea_wtfrac": "urea wt frac [-]",
         "nh3_makeup_kgph": "NH3 make-up [kg/h]"}
# IPPS op-point keys for each section's inputs and outputs
OPKEY = {"A": (("Ft", "Fh2"), {"ammonia_kgph": "NA", "water_kgph": "WA",
                               "electric_kwhph": "EA"}),
         "B": (("NA", "WA"), {"anhydrous_nh3_kgph": "NB",
                              "electric_kwhph": "EB"}),
         "C": (("Fnh3", "Fco2"), {"pure_urea_kgph": "U",
                                  "product_urea_wtfrac": "wU",
                                  "electric_kwhph": "EC"}),
         "D": (("Fco2",), {"pure_urea_kgph": "U", "nh3_makeup_kgph": "Fnh3",
                           "electric_kwhph": "EC"})}
MODE_STYLE = {"free_grid": ("tab:red", "*"), "grid_10pct": ("tab:orange", "D"),
              "grid_5pct": ("tab:purple", "s"),
              "wind_only": ("tab:green", "^")}


def plot_1d(sec, meta, ops):
    xin = meta[sec]["inputs"][0]
    outs = meta[sec]["outputs"]
    df = pd.read_csv(DATA / f"{sec}.csv")
    lo, hi = meta[sec]["input_bounds"][xin]
    xs = np.linspace(lo, hi, 200, dtype=np.float32).reshape(-1, 1)
    fig, axes = plt.subplots(1, len(outs), figsize=(4.8 * len(outs), 4.2))
    for ax, (out, d) in zip(axes, outs.items()):
        net = tf.keras.models.load_model(NET / d["file"])
        ax.plot(xs, net(xs).numpy(), color="tab:blue", label="ReLU fit")
        ax.scatter(df[xin], df[out], c="k", s=18, zorder=3, label="IDAES data")
        ink, outk = OPKEY[sec]
        for mode, op in ops.items():
            col, mk = MODE_STYLE.get(mode, ("tab:red", "*"))
            ax.scatter([op[ink[0]]], [op[outk[out]]], c=col, marker=mk,
                       s=140, edgecolors="k", zorder=4,
                       label=f"IPPS op. point: {mode}")
        ax.set_xlabel(LABEL[xin])
        ax.set_ylabel(LABEL[out])
        ax.set_title(f"R$^2$ {d['r2']:.4f}, RMSE {d['rmse']:.3g}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(TITLES[sec] + f"  ({meta[sec]['n_points']} IDAES points)")
    fig.tight_layout()
    path = FIG / f"idaes_surrogate_{sec}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    meta = json.loads((NET / "metadata.json").read_text())
    import sys
    op_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OP
    ops = json.loads(op_path.read_text()) if op_path.exists() else {}
    for sec in "ABCD":
        if sec not in meta:
            continue
        if len(meta[sec]["inputs"]) == 1:
            plot_1d(sec, meta, ops)
            continue
        xin = meta[sec]["inputs"]
        outs = meta[sec]["outputs"]
        df = pd.read_csv(DATA / f"{sec}.csv")
        (x0lo, x0hi), (x1lo, x1hi) = (meta[sec]["input_bounds"][k] for k in xin)
        g0, g1 = np.meshgrid(np.linspace(x0lo, x0hi, 45),
                             np.linspace(x1lo, x1hi, 45))
        grid = np.c_[g0.ravel(), g1.ravel()].astype(np.float32)
        fig = plt.figure(figsize=(5.6 * len(outs), 5.2))
        for i, (out, d) in enumerate(outs.items(), 1):
            net = tf.keras.models.load_model(NET / d["file"])
            z = net(grid).numpy().reshape(g0.shape)
            ax = fig.add_subplot(1, len(outs), i, projection="3d")
            ax.plot_surface(g0, g1, z, cmap="viridis", alpha=0.55,
                            linewidth=0, antialiased=True)
            ax.scatter(df[xin[0]], df[xin[1]], df[out], c="k", s=10,
                       depthshade=False, label="IDAES data")
            ink, outk = OPKEY[sec]
            # C (ureaF replica) is not in the IPPS: no operating point
            for mode, op in (ops.items() if sec != "C" else []):
                col, mk = MODE_STYLE.get(mode, ("tab:red", "*"))
                ax.scatter([op[ink[0]]], [op[ink[1]]], [op[outk[out]]],
                           c=col, marker=mk, s=140, edgecolors="k",
                           depthshade=False, label=f"IPPS op. point: {mode}")
            ax.set_xlabel(LABEL[xin[0]])
            ax.set_ylabel(LABEL[xin[1]])
            ax.set_zlabel(LABEL[out])
            ax.set_title(f"{LABEL[out]}\nReLU {meta['hidden_layers']}, "
                         f"R$^2$ {d['r2']:.4f}, RMSE {d['rmse']:.3g}",
                         fontsize=10)
            ax.view_init(elev=25, azim=-135)
            if i == 1:
                ax.legend(loc="upper left", fontsize=7)
        fig.suptitle(TITLES[sec] + f"  ({meta[sec]['n_points']} IDAES points)")
        fig.tight_layout()
        path = FIG / f"idaes_surrogate_{sec}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

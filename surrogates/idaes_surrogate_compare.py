"""
Old vs new surrogates for B and D after the design optimisation
(B: overhead MVR + vacuum 0.06 bar; D: steam 0.2 kg/kg CO2, N/C 3.5).

    python idaes_surrogate_compare.py OLD_NET_DIR NEW_NET_DIR B_OLD.csv D_OLD.csv
Writes figures/idaes_surrogate_compare_{B,D}.png.
"""

import json
import os
import sys
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
FIG = ROOT / "figures"
OP_B = {"ammonia_kgph": 9.85, "water_kgph": 5056.0}   # 10-day IPPS op point
OP_D = 3.63


def net(d, name):
    return tf.keras.models.load_model(Path(d) / name)


def main():
    old_dir, new_dir, b_old, d_old = sys.argv[1:5]
    # ---------------- D: 1-D curves
    do, dn = pd.read_csv(d_old), pd.read_csv(DATA / "D.csv")
    outs = ["pure_urea_kgph", "nh3_makeup_kgph", "electric_kwhph"]
    lab = {"pure_urea_kgph": "urea [kg/h]",
           "nh3_makeup_kgph": "NH3 make-up [kg/h]",
           "electric_kwhph": "electricity [kW]"}
    x = np.linspace(3, 20, 300, dtype=np.float32).reshape(-1, 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, o in zip(ax, outs):
        a.scatter(do.Fco2, do[o], c="0.6", s=22, label="old data (steam 1 kg/kg, N/C 4)")
        a.scatter(dn.Fco2, dn[o], c="k", s=14, label="new data (steam 0.2, N/C 3.5)")
        a.plot(x, net(old_dir, f"D_{o}.keras")(x).numpy(), color="tab:orange",
               ls="--", label="old surrogate")
        a.plot(x, net(new_dir, f"D_{o}.keras")(x).numpy(), color="tab:blue",
               label="new surrogate")
        a.axvline(OP_D, color="tab:red", lw=0.8, ls=":", label="IPPS Fco2 3.63")
        a.set_xlabel("Fco2 [kg/h]")
        a.set_ylabel(lab[o])
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    # zoom inset on electricity at low flow
    ins = ax[2].inset_axes([0.08, 0.5, 0.38, 0.42])
    xs = np.linspace(3, 6, 100, dtype=np.float32).reshape(-1, 1)
    for d_, c_, s_ in ((do, "0.6", 22), (dn, "k", 14)):
        z = d_[d_.Fco2 <= 6]
        ins.scatter(z.Fco2, z.electric_kwhph, c=c_, s=s_)
    ins.plot(xs, net(old_dir, "D_electric_kwhph.keras")(xs).numpy(), "--",
             color="tab:orange")
    ins.plot(xs, net(new_dir, "D_electric_kwhph.keras")(xs).numpy(),
             color="tab:blue")
    ins.set_title("Fco2 3-6", fontsize=7)
    ins.tick_params(labelsize=6)
    fig.suptitle("D  DEF plant: old vs new design / data / surrogate (ReLU (8,))")
    fig.tight_layout()
    fig.savefig(FIG / "idaes_surrogate_compare_D.png", dpi=140)
    plt.close(fig)

    # ---------------- B: slices through the IPPS op point
    bo, bn = pd.read_csv(b_old), pd.read_csv(DATA / "B.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, (var, fixed) in zip(ax, (("ammonia_kgph", "water_kgph"),
                                    ("water_kgph", "ammonia_kgph"))):
        lo, hi = bn[var].min(), bn[var].max()
        g = np.linspace(lo, hi, 200)
        X = np.zeros((200, 2), dtype=np.float32)
        i = 0 if var == "ammonia_kgph" else 1
        X[:, i] = g
        X[:, 1 - i] = OP_B[fixed]
        for d_, c_, lb in ((bo, "0.6", "old data (0.08 bar, no MVR)"),
                           (bn, "k", "new data (0.06 bar + MVR)")):
            near = d_.loc[(d_[fixed] - OP_B[fixed]).abs()
                          <= (d_[fixed].max() - d_[fixed].min()) / 30 + 1e-9]
            a.scatter(near[var], near.electric_kwhph, c=c_, s=12, label=lb)
        a.plot(g, net(old_dir, "B_electric_kwhph.keras")(X).numpy(), "--",
               color="tab:orange", label="old surrogate")
        a.plot(g, net(new_dir, "B_electric_kwhph.keras")(X).numpy(),
               color="tab:blue", label="new surrogate")
        a.axvline(OP_B[var], color="tab:red", lw=0.8, ls=":",
                  label="IPPS op point")
        a.set_xlabel(var.replace("_kgph", " [kg/h]"))
        a.set_ylabel("B electricity [kW]")
        a.set_title(f"slice at {fixed.replace('_kgph', '')} = {OP_B[fixed]:g} kg/h",
                    fontsize=10)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("B  NH3 recovery: old vs new design / data / surrogate (ReLU (8,))")
    fig.tight_layout()
    fig.savefig(FIG / "idaes_surrogate_compare_B.png", dpi=140)
    plt.close(fig)
    print("wrote figures/idaes_surrogate_compare_{B,D}.png")


if __name__ == "__main__":
    main()

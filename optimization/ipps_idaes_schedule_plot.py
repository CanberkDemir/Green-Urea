"""
Hourly schedule of the IPPS solution from ipps_idaes.py.

    python ipps_idaes_schedule_plot.py [tag]      # tag e.g. _10d
Reads optimization/solutions/ipps_idaes_hourly{tag}_<mode>.csv and writes
figures/ipps_idaes_schedule{tag}_<mode>.png: power supply and loads,
ammonia-train / urea on-off, and storage levels.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "optimization" / "solutions"
FIG = ROOT / "figures"


def plot(path, out):
    h = pd.read_csv(path)
    t = h.t
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 2]})
    a = ax[0]
    loads = [("electrolyser_kw", "electrolyser", "tab:blue"),
             ("ammonia_AB_kw", "ammonia A + recovery B", "tab:orange"),
             ("urea_D_kw", "DEF plant D", "tab:green"),
             ("batt_ch_kw", "battery charge", "tab:purple")]
    a.stackplot(t, *[h[c] for c, _, _ in loads],
                labels=[lb for _, lb, _ in loads],
                colors=[c for _, _, c in loads], alpha=0.75, step="post")
    a.step(t, h.wind_avail_kw, where="post", color="k", lw=1,
           label="wind available")
    if h.grid_kw.max() > 1e-6:
        a.step(t, h.grid_kw, where="post", color="tab:red", lw=1,
               label="grid import")
    if h.batt_dis_kw.max() > 1e-6:
        a.step(t, h.batt_dis_kw, where="post", color="tab:pink", lw=1,
               ls="--", label="battery discharge")
    a.set_ylabel("power [kW]")
    a.legend(ncol=3, fontsize=8, loc="upper right")
    a.grid(alpha=0.3)

    a = ax[1]
    a.step(t, h.ammonia_on, where="post", color="tab:orange",
           label="ammonia train on")
    a.step(t, h.urea_on * 0.9, where="post", color="tab:green",
           label="urea section on (x0.9)")
    a.set_ylim(-0.1, 1.15)
    a.set_yticks([0, 1])
    a.legend(fontsize=8, loc="lower right")
    a.grid(alpha=0.3)

    a = ax[2]
    a.plot(t, h.nh3_store_kg, color="tab:orange", label="NH3 store [kg]")
    a.plot(t, h.h2_store_kg, color="tab:blue", label="H2 store [kg]")
    a.set_ylabel("kg")
    b = a.twinx()
    b.plot(t, h.batt_soc_kwh, color="tab:purple", ls="--",
           label="battery SOC [kWh]")
    b.set_ylabel("kWh")
    lines = a.get_lines() + b.get_lines()
    a.legend(lines, [ln.get_label() for ln in lines], fontsize=8,
             loc="upper right")
    a.set_xlabel("hour of the sampled horizon (days are non-consecutive)")
    a.grid(alpha=0.3)
    for d in range(24, len(t), 24):
        for x in ax:
            x.axvline(d, color="0.85", lw=0.6, zorder=0)
    fig.suptitle(f"IPPS schedule (IDAES/reluMIP surrogates): {out.stem}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    for f in sorted(SOL.glob(f"ipps_idaes_hourly{tag}_*.csv")):
        mode = f.stem.split(f"ipps_idaes_hourly{tag}_")[-1]
        plot(f, FIG / f"ipps_idaes_schedule{tag}_{mode}.png")


if __name__ == "__main__":
    main()

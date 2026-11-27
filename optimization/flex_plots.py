"""Charts for flex_planning.py (costs per strategy, two-week dispatch)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "optimization" / "solutions"
FIG = ROOT / "figures"
STRATS = ("battery", "flex_nh3", "flex_def")


def plot(out=None, series=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if out is None:
        out = pd.read_csv(SOL / "flex_planning_summary.csv").to_dict("records")
    if series is None:
        series = {s: pd.read_csv(SOL / f"flex_planning_{s}.csv")
                  for s in STRATS}
    keys = ["cost wind [$/yr]", "cost battery [$/yr]", "cost NH3 tank [$/yr]",
            "cost NO3RR stack [$/yr]"]
    cols = ["#4a90c2", "#e0a030", "#7aa95c", "#9467bd"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, r in enumerate(out):
        b = 0.0
        for k, c in zip(keys, cols):
            v = r[k] / 1e3
            ax.bar(i, v, bottom=b, color=c, width=0.55, edgecolor="white",
                   label=k[5:-8] if i == 0 else None)
            b += v
        ax.text(i, b + 20, f"${r['cost [$/kg urea]']:.2f}/kg urea",
                ha="center", fontweight="bold")
    ax.set_xticks(range(len(out)), ["all continuous\n+ battery",
                                    "NH3 follows wind\n+ NH3 tank",
                                    "+ DEF turndown\nto 50 %"])
    ax.set_ylabel("Annualised cost [k$/yr]")
    ax.set_title("Wind-only operation: storage strategy (2024 Hornsea-2 profile)")
    ax.legend(frameon=False, fontsize=9, loc="upper left",
              bbox_to_anchor=(1.0, 1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "flex_planning_costs.png", dpi=200)
    plt.close(fig)

    ser = series["flex_def"].iloc[24 * 30:24 * 44]
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    h = range(len(ser))
    axs[0].plot(h, ser["wind_kw"], color="#4a90c2", lw=1)
    axs[0].set_ylabel("wind available [kW]")
    axs[1].plot(h, ser["nh3_kgph"], color="#9467bd", lw=1, label="NO3RR NH3")
    axs[1].plot(h, ser["def_nh3_kgph"], color="#d62728", lw=1,
                label="DEF plant NH3 draw")
    axs[1].set_ylabel("NH3 [kg/h]")
    axs[1].legend(frameon=False, fontsize=8)
    axs[2].plot(h, ser["tank_kg"], color="#7aa95c", lw=1.2)
    axs[2].set_ylabel("NH3 in tank [kg]")
    axs[2].set_xlabel("hour (from 31 Jan 2024)")
    axs[0].set_title("Optimal dispatch: NH3 follows wind, DEF may turn down")
    for a_ in axs:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "flex_planning_dispatch.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot()

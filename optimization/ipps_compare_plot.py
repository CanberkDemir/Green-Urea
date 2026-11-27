"""H2 route (existing IPPS results) vs electro route (ipps_electro.py), wind_only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
old = {x["key"]: x for x in json.loads(
    (ROOT / "figures/paper_figures/paper_numbers.json").read_text())}["wind_only"]
hz = pd.read_csv(ROOT / "optimization/solutions/ipps_electro_ipps_horizon.csv") \
    .set_index("grid_mode").loc["wind_only"]
fy = pd.read_csv(ROOT / "optimization/solutions/ipps_electro_fullyear.csv") \
    .set_index("grid_mode").loc["wind_only"]
p = old["capex_parts_gbp"]
h2_ann = 0.05 * sum(v for k, v in p.items() if k not in ("NH3 plant", "Urea plant")) \
    + old["opex_net_gbp_per_y"]
urea = old["urea_prod_kg_per_y"]
labels = ["H2 route\n(IPPS, 888 h)", "electro\n(IPPS horizon)", "electro\n(full year)"]
data = {
    "Wind capacity [kW]": [old["W_cap_kw"], hz["W_cap_kw"], fy["W_cap_kw"]],
    "Battery [kWh]": [old["B_cap_kwh"], hz["B_cap_kwh"], fy["B_cap_kwh"]],
    "NH3 storage [kg]": [old["NH3_cap_kg"], hz["NH3_cap_kg"], fy["NH3_cap_kg"]],
    "Electricity [kWh/kg urea]": [old["elec_intensity_kwh_per_kg"],
                                  hz["elec intensity [kWh/kg urea]"],
                                  fy["elec intensity [kWh/kg urea]"]],
    "GBP/kg urea\n(excl. plant intercept)": [
        h2_ann / urea, hz["GBP per kg urea excl. plant correlation"],
        fy["GBP per kg urea excl. plant correlation"]],
}
fig, axs = plt.subplots(1, 5, figsize=(16, 4))
cols = ["#7f7f7f", "#9467bd", "#5b3c8c"]
for ax, (k, v) in zip(axs, data.items()):
    ax.bar(range(3), v, color=cols)
    for i, x in enumerate(v):
        ax.text(i, x, f"{x:,.0f}" if x >= 10 else f"{x:.2f}", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(range(3), labels, fontsize=8)
    ax.set_title(k, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Wind-only IPPS design, 125 kg urea/day: hydrogenation route vs "
             "electrochemical NO3RR route (same costs, wind data, demand)")
fig.tight_layout()
fig.savefig(ROOT / "figures/ipps_h2_vs_electro.png", dpi=180)
print("wrote figures/ipps_h2_vs_electro.png")

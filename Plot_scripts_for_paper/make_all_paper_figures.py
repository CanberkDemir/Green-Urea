"""Generate every paper figure as a PDF from the IPPS solution CSVs.

All figures are computed from the solution CSVs (no hardcoded results), so
re-running the planning cases and then this script regenerates the full
figure set and a JSON summary of the numbers quoted in the manuscript.

Economic basis
--------------
* Annualized cost: CRF * CAPEX + OPEX_net (identical to the IPPS objective).
* Break-even prices / NPW: the manuscript's NPW framework with
  i = 7 %, N = 10 y, tax rate Phi = 19 %, linear depreciation
  d = 0.68*CAPEX/N, salvage + working capital = 0.32*CAPEX.
  These parameters reproduce the original break-even calculation.
* Gate-to-gate LCA: grid 177 kgCO2e/MWh, offshore wind 13 kgCO2e/MWh;
  all inlet biogenic CO2 counted as removed, all CO2 not incorporated in
  the product vented back, i.e. the material term is -44/60 kgCO2 per kg
  urea. Comparators: SMR-based Haber-Bosch NH3 2.2, conventional
  Bosch-Meiser urea 1.83 kgCO2/kg.
"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from plotting_compat import normalize_result_columns  # noqa: E402

plt.rcParams.update({"font.size": 11})
VIRIDIS = plt.get_cmap("viridis")
VIRIDIS_015 = "#463480"

CASES = [
    ("Unrestricted grid", "free_grid", ROOT / "ipps_solution_smallhorizon_free_grid.csv"),
    ("10% grid", "grid_10pct", ROOT / "ipps_solution_smallhorizon_grid_10pct.csv"),
    ("5% grid", "grid_5pct", ROOT / "ipps_solution_smallhorizon_grid_5pct.csv"),
    ("Wind only", "wind_only", ROOT / "ipps_solution_smallhorizon_wind_only.csv"),
]

# ----------------------------------------------------------------- economics
CRF = 0.05
COST_RATES = {
    "Wind": 864.54545,  # GBP per kW (Hornsea 2: GBP 864,545 per MW)
    "Electrolyzer": 4_130.4,  # GBP per kW (incremental basis)
    "Battery": 104.0,  # GBP per kWh
    "H2 storage": 645.89,  # GBP per kg
    "CO2 storage": 0.42,  # GBP per kg
    "NH3 storage": 1.89,  # GBP per kg
    "Heat exchanger": 2.07,  # GBP per kW
    "Grid electricity": 0.18,  # GBP per kWh
    "CO2 feed": 0.02,  # GBP per kg
    "O2 credit": 0.03,  # GBP per kg
}
CAPITAL_SLICES = [
    ("Wind", "W_cap"),
    ("Electrolyzer", "E_cap_incremental"),
    ("Battery", "B_cap"),
    ("H2 storage", "H_cap"),
    ("CO2 storage", "C_cap"),
    ("NH3 storage", "NH3_cap"),
    ("Heat exchanger", "HX_cap"),
    ("NH3 plant", "nh3_plant_capex"),
    ("Urea plant", "urea_plant_capex"),
]
OPERATING_SLICES = [
    ("Grid electricity", "P_grid"),
    ("CO2 feed", "Fco2"),
]

# NPW parameters (reproduce the manuscript break-even framework)
NPW_I = 0.07
NPW_N = 10
NPW_PHI = 0.19
ANNUITY = sum(1.0 / (1.0 + NPW_I) ** j for j in range(1, NPW_N + 1))
DISCOUNT_N = 1.0 / (1.0 + NPW_I) ** NPW_N
# NPW(s) = -C + (s - OPEX)(1-Phi)A + 0.068*C*A + 0.32*C/(1+i)^N
K_CAPITAL = (1.0 - (0.68 / NPW_N) * ANNUITY - 0.32 * DISCOUNT_N) / ((1.0 - NPW_PHI) * ANNUITY)

UREA_MARKET_GBP_PER_KG = 1.83
CO2_CERT_MARKET_GBP_PER_T = 41.84
GRID_KGCO2_PER_KWH = 0.177
# Headline gate-to-gate accounting: dedicated wind is carbon-neutral at the
# point of use (turbine-embodied emissions lie upstream of the gate).
# Sensitivity: UNECE life-cycle intensity for offshore wind.
WIND_KGCO2_PER_KWH_OPERATIONAL = 0.0
WIND_KGCO2_PER_KWH_LIFECYCLE = 0.013
CO2_IN_UREA_KG_PER_KG = 44.0 / 60.0
CONVENTIONAL_REFS = [("SMR Haber-Bosch\n(NH$_3$)", 2.2), ("Bosch-Meiser\n(urea)", 1.83)]


def first(df, col):
    return float(df[col].iloc[0])


def annualize(df):
    return 8760.0 / len(df)


def capital_breakdown(df):
    parts = {}
    for label, col in CAPITAL_SLICES:
        v = first(df, col)
        parts[label] = v if col.endswith("_capex") else COST_RATES[label] * v
    return parts


def operating_breakdown(df):
    scale = annualize(df)
    parts = {label: scale * COST_RATES[label] * df[col].sum() for label, col in OPERATING_SLICES}
    parts["O2 credit"] = -scale * COST_RATES["O2 credit"] * df["M_O2_prod"].sum()
    return parts


def case_summary(name, key, path):
    df = pd.read_csv(path)
    scale = annualize(df)
    capex_parts = capital_breakdown(df)
    opex_parts = operating_breakdown(df)
    capex_raw = sum(capex_parts.values())
    opex_net = sum(opex_parts.values())
    annualized = CRF * capex_raw + opex_net

    prod = scale * df["F_U"].sum()  # kg urea / y
    co2_t = scale * df["Fco2"].sum() / 1000.0  # t CO2 utilized / y

    bep = (opex_net + K_CAPITAL * capex_raw) / prod
    bep_cert = (opex_net + K_CAPITAL * capex_raw - UREA_MARKET_GBP_PER_KG * prod) / co2_t

    s_market = UREA_MARKET_GBP_PER_KG * prod + CO2_CERT_MARKET_GBP_PER_T * co2_t
    npw = (
        -capex_raw
        + (s_market - opex_net) * (1 - NPW_PHI) * ANNUITY
        + (0.68 * capex_raw / NPW_N) * ANNUITY
        + 0.32 * capex_raw * DISCOUNT_N
    )

    load = df["total_electric_load"].sum()
    grid_kwh = df["P_grid"].sum()
    # P_wind is only bounded above by availability and can float to its bound,
    # so wind CONSUMPTION is inferred from the supply balance instead.
    # total_electric_load already includes ch_B; battery discharge re-enters
    # the supply side, so wind use = load - grid - dis_B.
    wind_kwh = max(0.0, load - df["dis_B"].sum() - grid_kwh)
    urea_kg = df["F_U"].sum()
    emissions_op = (grid_kwh * GRID_KGCO2_PER_KWH
                    + wind_kwh * WIND_KGCO2_PER_KWH_OPERATIONAL) / urea_kg \
        - CO2_IN_UREA_KG_PER_KG
    emissions_lc = (grid_kwh * GRID_KGCO2_PER_KWH
                    + wind_kwh * WIND_KGCO2_PER_KWH_LIFECYCLE) / urea_kg \
        - CO2_IN_UREA_KG_PER_KG

    wind_avail = (df["cf_wind"] * first(df, "W_cap")).sum()
    return {
        "name": name,
        "key": key,
        "hours": len(df),
        "W_cap_kw": first(df, "W_cap"),
        "E_cap_kw": first(df, "E_cap"),
        "B_cap_kwh": first(df, "B_cap"),
        "H_cap_kg": first(df, "H_cap"),
        "NH3_cap_kg": first(df, "NH3_cap"),
        "Fh2_op_kgph": first(df, "Fh2_op"),
        "availability_A_pct": 100.0 * df["y_A_on"].mean(),
        "availability_U_pct": 100.0 * df["y_U_on"].mean(),
        "grid_share_pct": 100.0 * grid_kwh / load,
        "wind_consumed_kwh": wind_kwh,
        "wind_available_kwh": wind_avail,
        "curtailment_pct": 100.0 * max(0.0, 1.0 - wind_kwh / wind_avail) if wind_avail > 0 else 0.0,
        "elec_intensity_kwh_per_kg": load / df["F_U"].sum(),
        "mean_load_kw": load / len(df),
        "share_P_el_pct": 100.0 * df["P_el"].sum() / load,
        "share_E_A_pct": 100.0 * df["E_A"].sum() / load,
        "share_E_U_pct": 100.0 * df["E_U"].sum() / load,
        "heat_recovery_pct": 100.0 * df["Q_rec"].sum() / df["Q_S"].sum() if df["Q_S"].sum() > 0 else 100.0,
        "capex_raw_gbp": capex_raw,
        "capex_parts_gbp": capex_parts,
        "opex_net_gbp_per_y": opex_net,
        "opex_parts_gbp_per_y": opex_parts,
        "annualized_cost_gbp_per_y": annualized,
        "urea_prod_kg_per_y": prod,
        "co2_utilized_t_per_y": co2_t,
        "bep_gbp_per_kg": bep,
        "bep_cert_gbp_per_t": bep_cert,
        "npw_market_gbp": npw,
        "emissions_kgco2_per_kg": emissions_op,
        "emissions_lifecycle_kgco2_per_kg": emissions_lc,
    }


# ----------------------------------------------------------------- figures
def fig_electricity_balance(dfs, out_path, start=390, end=450):
    supply = [("P_wind", "Wind electricity", 0.30), ("P_grid", "Grid import", 0.95),
              ("dis_B", "Battery discharge", 0.55)]
    use = [("P_el", "Electrolyzer", 0.90), ("E_N", "Ammonia unit", 0.06),
           ("E_U", "Urea unit", 0.35), ("P_HU_el", "Electric hot utility", 0.73),
           ("e_Hstor", "H2 storage electricity", 0.20), ("e_Cstor", "CO2 storage electricity", 0.84),
           ("ch_B", "Battery charge", 0.61)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=False)
    labels_seen = {}
    panel_tags = ["(a)", "(b)", "(c)", "(d)"]
    for ax, tag, (name, df) in zip(axes.flat, panel_tags, dfs):
        win = normalize_result_columns(df).iloc[start:end]
        x = win["t"].to_numpy(dtype=float)
        bottom = np.zeros(len(win))
        for col, label, point in supply:
            if col not in win.columns:
                continue
            vals = pd.to_numeric(win[col], errors="coerce").fillna(0.0).to_numpy()
            h = ax.bar(x, vals, width=1.0, bottom=bottom, color=VIRIDIS(point))
            labels_seen.setdefault(label, h)
            bottom += vals
        bottom = np.zeros(len(win))
        for col, label, point in use:
            if col not in win.columns:
                continue
            vals = pd.to_numeric(win[col], errors="coerce").fillna(0.0).to_numpy()
            h = ax.bar(x, -vals, width=1.0, bottom=bottom, color=VIRIDIS(point))
            labels_seen.setdefault(label, h)
            bottom -= vals
        ax.axhline(0.0, color="0.2", linewidth=0.8)
        ax.set_title(f"{tag} {name}", loc="left", fontsize=12)
        ax.set_ylabel("Electricity (kW)")
        ax.set_xlabel("Hour")
        ax2 = ax.twinx()
        inv = pd.to_numeric(win["I_B"], errors="coerce").fillna(0.0).to_numpy()
        (line,) = ax2.plot(x, inv, color="0.15", linewidth=1.4, linestyle="--")
        labels_seen.setdefault("Battery inventory (kWh, right)", line)
        ax2.set_ylabel("Battery inventory (kWh)")
    fig.legend(labels_seen.values(), labels_seen.keys(), loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_storage_capacities(summaries, out_path):
    names = [s["name"] for s in summaries]
    H = [s["H_cap_kg"] for s in summaries]
    N = [s["NH3_cap_kg"] for s in summaries]
    B = [s["B_cap_kwh"] for s in summaries]
    x = np.arange(len(names))
    width = 0.25
    c_H, c_N, c_B = VIRIDIS(0.15), VIRIDIS(0.5), VIRIDIS(0.85)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    b1 = ax1.bar(x - width, H, width, color=c_H)
    b2 = ax1.bar(x, N, width, color=c_N)
    ax1.set_ylabel("Chemical storage capacity (kg)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax2 = ax1.twinx()
    b3 = ax2.bar(x + width, B, width, color=c_B)
    ax2.set_ylabel("Battery capacity (kWh)", color=c_B)
    ax2.tick_params(axis="y", colors=c_B)
    ax2.spines["right"].set_color(c_B)
    if max(B) > 9000:
        ax2.axhline(10000.0, color=c_B, linestyle="--", linewidth=1.0)
        ax2.text(0.02, 10050, "battery sizing bound (10 MWh)", color=c_B, fontsize=9)
        ax2.set_ylim(0, 11500)
    for bars, ax in ((b1, ax1), (b2, ax1), (b3, ax2)):
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.0f}",
                        ha="center", va="bottom", fontsize=8)
    ax1.legend([b1, b2, b3],
               [r"H$_2$ storage (kg)", r"NH$_3$ storage (kg)", "Battery (kWh)"],
               loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


COMPONENT_POINTS = {
    "Wind": 0.06, "Electrolyzer": 0.16, "Battery": 0.26, "H2 storage": 0.36,
    "CO2 storage": 0.46, "NH3 storage": 0.56, "Heat exchanger": 0.66,
    "NH3 plant": 0.42, "Urea plant": 0.90, "Grid electricity": 0.98,
    "CO2 feed": 0.50, "O2 credit": 0.76,
}


def fig_annualized_cost_allocation(summaries, out_path):
    names = [s["name"] for s in summaries]
    x = np.arange(len(names))
    labels = [l for l, _ in CAPITAL_SLICES] + [l for l, _ in OPERATING_SLICES] + ["O2 credit"]
    active = []
    for label in labels:
        vals = []
        for s in summaries:
            if label in s["capex_parts_gbp"]:
                vals.append(CRF * s["capex_parts_gbp"][label] / 1e6)
            else:
                vals.append(s["opex_parts_gbp_per_y"].get(label, 0.0) / 1e6)
        if any(abs(v) > 5e-4 for v in vals):
            active.append((label, np.array(vals)))

    fig, ax = plt.subplots(figsize=(10, 5.6))
    pos = np.zeros(len(names))
    neg = np.zeros(len(names))
    for label, vals in active:
        bottom = np.where(vals >= 0, pos, neg)
        kind = "Capex" if label in dict(CAPITAL_SLICES) else "Opex"
        ax.bar(x, vals, 0.62, bottom=bottom, color=VIRIDIS(COMPONENT_POINTS[label]),
               edgecolor="white", linewidth=0.6, label=f"{kind}: {label}")
        for xi, b, v in zip(x, bottom, vals):
            if abs(v) > 0.35:
                ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        pos += np.where(vals >= 0, vals, 0.0)
        neg += np.where(vals < 0, vals, 0.0)
    totals = pos + neg
    for xi, total, top in zip(x, totals, pos):
        ax.text(xi, top + max(pos) * 0.015, f"{total:.2f}M", ha="center",
                va="bottom", fontsize=9.5)
    ax.axhline(0.0, color="0.25", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Annualized cost (million GBP y$^{-1}$)")
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.98))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_emissions(summaries, out_path):
    """Grouped bars: operational gate-to-gate vs life-cycle-electricity
    sensitivity per case, plus conventional cradle-to-gate references."""
    names = [s["name"] for s in summaries]
    op = [s["emissions_kgco2_per_kg"] for s in summaries]
    lc = [s["emissions_lifecycle_kgco2_per_kg"] for s in summaries]
    x = np.arange(len(names))
    width = 0.38
    c_op, c_lc, c_ref = VIRIDIS(0.25), VIRIDIS(0.60), VIRIDIS(0.85)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    b_op = ax.bar(x - width / 2, op, width, color=c_op,
                  label="Gate-to-gate (dedicated wind zero-rated)")
    b_lc = ax.bar(x + width / 2, lc, width, color=c_lc,
                  label="Incl. life-cycle electricity intensities")
    ref_x = np.arange(len(CONVENTIONAL_REFS)) + len(names) + 0.3
    b_ref = ax.bar(ref_x, [v for _, v in CONVENTIONAL_REFS], width,
                   color=c_ref, label="Conventional route (literature)")
    for bars in (b_op, b_lc, b_ref):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.2f}", ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=8.5)
    ax.axhline(0.0, color="0.25", linewidth=0.9)
    ax.set_xticks(list(x) + list(ref_x))
    ax.set_xticklabels(names + [n for n, _ in CONVENTIONAL_REFS], fontsize=9)
    ax.set_ylabel(r"CO$_2$ emissions (kg$_{\mathrm{CO_2}}$ kg$_{\mathrm{product}}^{-1}$)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _simple_bar(values, names, ylabel, out_path, market=None, market_label=None, fmt="{:.2f}"):
    fig = plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=VIRIDIS_015)
    plt.ylabel(ylabel)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h if h >= 0 else h,
                 fmt.format(h), ha="center", va="bottom" if h >= 0 else "top")
    if market is not None:
        plt.axhline(market, color="#b91c1c", linestyle="--", linewidth=1.2)
        plt.text(len(names) - 0.55, market, f" {market_label}", color="#b91c1c",
                 fontsize=9, va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    summaries = [case_summary(name, key, path) for name, key, path in CASES]
    dfs = [(name, pd.read_csv(path)) for name, _, path in CASES]

    fig_electricity_balance(dfs, OUT_DIR / "electricity_balance_cases.pdf")
    fig_storage_capacities(summaries, OUT_DIR / "storage_capacity_cases.pdf")
    fig_annualized_cost_allocation(summaries, OUT_DIR / "annualized_cost_allocation.pdf")

    names = [s["name"] for s in summaries]
    _simple_bar([s["annualized_cost_gbp_per_y"] / 1e6 for s in summaries], names,
                "Annualized cost (million GBP y$^{-1}$)",
                OUT_DIR / "annualized_cost_cases.pdf")
    _simple_bar([s["bep_gbp_per_kg"] for s in summaries], names,
                r"Break-even price (GBP kg$_{\mathrm{urea}}^{-1}$)",
                OUT_DIR / "bep.pdf",
                market=UREA_MARKET_GBP_PER_KG, market_label="market price 1.83 GBP/kg")
    _simple_bar([s["bep_cert_gbp_per_t"] for s in summaries], names,
                r"Break-even CO$_2$ certificate price (GBP t$_{\mathrm{CO_2}}^{-1}$)",
                OUT_DIR / "bep_cert.pdf",
                market=CO2_CERT_MARKET_GBP_PER_T,
                market_label="market price 41.84 GBP/t", fmt="{:,.0f}")
    fig_emissions(summaries, OUT_DIR / "emissions.pdf")

    with (OUT_DIR / "paper_numbers.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    cols = ["name", "W_cap_kw", "E_cap_kw", "B_cap_kwh", "H_cap_kg", "NH3_cap_kg",
            "Fh2_op_kgph", "availability_A_pct", "grid_share_pct",
            "annualized_cost_gbp_per_y", "bep_gbp_per_kg", "bep_cert_gbp_per_t",
            "npw_market_gbp", "emissions_kgco2_per_kg",
            "emissions_lifecycle_kgco2_per_kg"]
    print(pd.DataFrame(summaries)[cols].to_string(index=False))
    print(f"\nk_capital = {K_CAPITAL:.5f} (i={NPW_I}, N={NPW_N}, Phi={NPW_PHI})")
    print(f"Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

CASES = [
    ("Grid only", ROOT / "ipps_solution_smallhorizon_free_grid.csv"),
    ("10% Grid", ROOT / "ipps_solution_smallhorizon_grid_10pct.csv"),
    ("5% Grid", ROOT / "ipps_solution_smallhorizon_grid_5pct.csv"),
    ("Wind only", ROOT / "ipps_solution_smallhorizon_wind_only.csv"),
]

CRF = 0.05
COST_RATES = {
    "Wind": 864_545.45,
    "Electrolyzer": 4_130.4,
    "Battery": 104.0,
    "H2 storage": 645.89,
    "CO2 storage": 0.42,
    "NH3 storage": 1.89,
    "Heat exchanger": 2.07,
    "Grid electricity": 0.18,
    "CO2 feed": 0.02,
    "Electrolyzer water": 0.00,
    "Feedstock": 0.00,
    "O2 credit": 0.03,
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
    ("Electrolyzer water", "M_H2O_el"),
    ("Feedstock", "Ft"),
]

COST_BAR_ORDER = [
    "Capex: Wind",
    "Capex: Electrolyzer",
    "Capex: Battery",
    "Capex: H2 storage",
    "Capex: CO2 storage",
    "Capex: NH3 storage",
    "Capex: Heat exchanger",
    "Capex: NH3 plant",
    "Capex: Urea plant",
    "Opex: Grid electricity",
    "Opex: CO2 feed",
    "Opex: Electrolyzer water",
    "Opex: Feedstock",
    "Opex: O2 credit",
]

ENERGY_SLICES = [
    ("Electrolyzer", "P_el"),
    ("Ammonia unit", "E_A"),
    ("Urea unit", "E_U"),
    ("Hot utility", "P_HU_el"),
]

VIRIDIS_POINTS = {
    "Wind": 0.06,
    "Electrolyzer": 0.16,
    "Battery": 0.26,
    "H2 storage": 0.36,
    "CO2 storage": 0.46,
    "NH3 storage": 0.56,
    "Heat exchanger": 0.66,
    "NH3 plant": 0.76,
    "Urea plant": 0.86,
    "Grid electricity": 0.96,
    "CO2 feed": 0.46,
    "Electrolyzer water": 0.62,
    "Feedstock": 0.72,
    "O2 credit": 0.92,
    "Ammonia unit": 0.42,
    "Urea unit": 0.68,
    "Hot utility": 0.84,
}


def viridis_colors(labels):
    cmap = plt.get_cmap("viridis")
    return [cmap(VIRIDIS_POINTS[label]) for label in labels]


def color_key(label):
    return label.replace("Capex: ", "").replace("Opex: ", "")


def first_value(df, column):
    return float(df[column].iloc[0])


def annualization_factor(df):
    dt_h = 1.0
    modeled_hours = len(df)
    return 8760.0 / (dt_h * modeled_hours)


def cost_components(df):
    capital = {}
    for label, column in CAPITAL_SLICES:
        if column.endswith("_capex"):
            capital[label] = CRF * first_value(df, column)
        else:
            capital[label] = CRF * COST_RATES[label] * first_value(df, column)

    scale = annualization_factor(df)
    operating = {
        label: scale * COST_RATES[label] * df[column].sum()
        for label, column in OPERATING_SLICES
    }
    oxygen_credit = -scale * COST_RATES["O2 credit"] * df["M_O2_prod"].sum()

    positive_costs = {f"Capex: {k}": v for k, v in capital.items() if abs(v) > 1e-9}
    positive_costs.update({f"Opex: {k}": v for k, v in operating.items() if abs(v) > 1e-9})
    if abs(oxygen_credit) > 1e-9:
        positive_costs["Opex: O2 credit"] = oxygen_credit

    return positive_costs, sum(capital.values()), sum(operating.values()), oxygen_credit


def energy_components(df):
    values = {label: df[column].sum() for label, column in ENERGY_SLICES}
    return {label: value for label, value in values.items() if value > 1e-9}


def autopct_for():
    def format_pct(pct):
        if pct < 3.0:
            return ""
        return f"{pct:.0f}%"

    return format_pct


def draw_capital_stacked_bar(case_data, output_name):
    active_labels = [
        label
        for label in COST_BAR_ORDER
        if label.startswith("Capex: ")
        and any(abs(values.get(label, 0.0)) > 1e-9 for _, values, *_ in case_data)
    ]
    case_names = [case_name for case_name, *_ in case_data]
    x = np.arange(len(case_names))
    bottoms = np.zeros(len(case_names))

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for label in active_labels:
        values = np.array([case_values.get(label, 0.0) / 1e6 for _, case_values, *_ in case_data])
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.62,
            color=viridis_colors([color_key(label)])[0],
            edgecolor="white",
            linewidth=0.7,
            label=label.replace("Capex: ", ""),
        )

        for case_x, bottom, value in zip(x, bottoms, values):
            if value < 1.0:
                continue
            ax.text(
                case_x,
                bottom + value / 2,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        bottoms += values

    for case_x, total in zip(x, bottoms):
        ax.text(
            case_x,
            total + max(bottoms) * 0.015,
            f"{total:.2f}M",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(case_names)
    ax.set_ylabel("Annualized capital cost (million GBP/y)")
    ax.set_title("Capital Cost Distribution by Case", fontsize=13, fontweight="bold")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(bottoms) * 1.12)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def draw_pie_grid(case_data, figure_title, output_name, show_cost_total=False):
    if show_cost_total:
        totals = [capex + opex + credit for _, _, capex, opex, credit in case_data]
    else:
        totals = [sum(values.values()) for _, values, *_ in case_data]
    max_total = max(abs(total) for total in totals)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    legend_labels = []

    for ax, data, total in zip(axes.flat, case_data, totals):
        case_name, values, *extra = data
        labels = list(values)
        sizes = list(values.values())
        radius = 0.62 + 0.38 * np.sqrt(abs(total) / max_total)
        wedges, _, autotexts = ax.pie(
            sizes,
            colors=viridis_colors([color_key(label) for label in labels]),
            startangle=90,
            counterclock=False,
            radius=radius,
            autopct=autopct_for(),
            pctdistance=0.68,
            wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
            textprops={"fontsize": 8, "color": "white"},
        )
        for text in autotexts:
            text.set_fontweight("bold")

        if show_cost_total:
            capex_total, opex_total, oxygen_credit = extra
            subtitle = (
                f"Total: GBP {total / 1e6:.2f}M/y\n"
                f"Capex: {capex_total / 1e6:.2f}M/y | "
                f"Opex: {(opex_total + oxygen_credit) / 1e6:.2f}M/y"
            )
        else:
            subtitle = f"Summed unit use: {total:,.0f} kWh"
        ax.set_title(f"{case_name}\n{subtitle}", fontsize=10)
        legend_labels.extend(label for label in labels if label not in legend_labels)
        ax.set_aspect("equal")

    legend_handles = [
        Patch(facecolor=viridis_colors([color_key(label)])[0], edgecolor="none", label=label)
        for label in legend_labels
    ]
    legend_cols = 4 if show_cost_total else len(legend_labels)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=legend_cols,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(figure_title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.94))
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    cost_data = []
    energy_data = []

    for case_name, path in CASES:
        df = pd.read_csv(path)
        costs, capex_total, opex_total, oxygen_credit = cost_components(df)
        cost_data.append((case_name, costs, capex_total, opex_total, oxygen_credit))
        energy_data.append((case_name, energy_components(df)))

    cost_path = draw_capital_stacked_bar(
        cost_data,
        "capital_cost_distribution_stacked_bars.png",
    )
    energy_path = draw_pie_grid(
        energy_data,
        "Major Energy Consumption Allocation",
        "major_energy_consumption_slices.png",
    )

    print(f"Wrote {cost_path}")
    print(f"Wrote {energy_path}")


if __name__ == "__main__":
    main()

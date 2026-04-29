from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

# Main manual scale knobs. Leave at 1.0 for the current layout.
FONT_SCALE = 0.8
CIRCLE_SCALE = 1.3
FIGURE_SCALE = 1.3

BASE_FONT_SIZE = 13 * FONT_SCALE
CALLOUT_FONT_SIZE = BASE_FONT_SIZE - 0.5
INNER_LABEL_FONT_SIZE = BASE_FONT_SIZE
CAPITAL_AXIS_FONT_SIZE = BASE_FONT_SIZE
CAPITAL_TICK_FONT_SIZE = BASE_FONT_SIZE
CAPITAL_VALUE_FONT_SIZE = BASE_FONT_SIZE - 0.5
ENERGY_TEXT_FONT_SIZE = BASE_FONT_SIZE
PIE_PERCENT_FONT_SIZE = BASE_FONT_SIZE - 0.5
LEGEND_FONT_SIZE = BASE_FONT_SIZE - 1.0

# Figure size and geometry knobs for manual tuning.
CAPITAL_FIGSIZE = (9.0 * FIGURE_SCALE, 5.4 * FIGURE_SCALE)
CAPITAL_BAR_WIDTH = 0.62
CAPITAL_Y_HEADROOM = 1.12
CAPITAL_LEGEND_BBOX = (0.5, -0.12)
CAPITAL_LAYOUT_RECT = (0.02, 0.08, 0.98, 0.98)

NPW_FIGSIZE = CAPITAL_FIGSIZE
NPW_BAR_WIDTH = CAPITAL_BAR_WIDTH
NPW_Y_HEADROOM = 1.10
NPW_LEGEND_BBOX = CAPITAL_LEGEND_BBOX
NPW_LAYOUT_RECT = (0.02, 0.12, 0.98, 0.98)
NPW_COMPONENT_LABEL_MIN = 30.0
NPW_FONT_MULTIPLIER = 1.5
NPW_AXIS_FONT_SIZE = CAPITAL_AXIS_FONT_SIZE * NPW_FONT_MULTIPLIER
NPW_TICK_FONT_SIZE = CAPITAL_TICK_FONT_SIZE * NPW_FONT_MULTIPLIER
NPW_VALUE_FONT_SIZE = CAPITAL_VALUE_FONT_SIZE * NPW_FONT_MULTIPLIER
NPW_LEGEND_FONT_SIZE = LEGEND_FONT_SIZE * NPW_FONT_MULTIPLIER

ENERGY_FIGSIZE = (12.0 * FIGURE_SCALE, 8.5 * FIGURE_SCALE)
ENERGY_MIN_RADIUS = 0.62 * CIRCLE_SCALE
ENERGY_RADIUS_SPAN = 0.38 * CIRCLE_SCALE
ENERGY_LABEL_Y = -1.20 * CIRCLE_SCALE
ENERGY_LEGEND_BBOX = (0.5, 0.01)
ENERGY_LAYOUT_RECT = (0.02, 0.12, 0.98, 0.98)

DETAILED_FIGSIZE = (7.2 * FIGURE_SCALE, 5.2 * FIGURE_SCALE)
DETAILED_INNER_RADIUS = 1.05 * CIRCLE_SCALE
DETAILED_INNER_WIDTH = 0.70 * CIRCLE_SCALE
DETAILED_OUTER_RADIUS = 1.75 * CIRCLE_SCALE
DETAILED_OUTER_WIDTH = 0.50 * CIRCLE_SCALE
DETAILED_INNER_LABEL_DISTANCE = (
    DETAILED_INNER_RADIUS - DETAILED_INNER_WIDTH / 1.2
) / DETAILED_INNER_RADIUS
DETAILED_ANCHOR_SCALE = 1.01
DETAILED_LABEL_Y_SCALE = 1.12
DETAILED_CALLOUT_ELBOW_X = 1.88 * CIRCLE_SCALE
DETAILED_CALLOUT_TEXT_X = 2.17 * CIRCLE_SCALE
DETAILED_X_LIMIT = 2.75 * CIRCLE_SCALE
DETAILED_Y_LIMIT = 2.20 * CIRCLE_SCALE
DETAILED_LABEL_MIN_GAP = 0.46 * FONT_SCALE
DETAILED_LABEL_Y_LIMIT = 2.18 * CIRCLE_SCALE
DETAILED_LINE_WIDTH = 0.9 * CIRCLE_SCALE

GROUP_LABELS = {
    "Electrolyzer": "Electrolyzer",
    "Ammonia": "NH3",
    "Urea": "Urea",
}

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

AMMONIA_DUTY_COLUMNS = ["Qh1", "Qc1", "Qr1", "Qcomp"]
UREA_DUTY_COLUMNS = [
    "QB3",
    "QB6",
    "QB27",
    "QB7_reb",
    "QB7_cond",
    "QB25_reb",
    "QB28_reb",
    "QB28_cond",
    "QR01",
]
MMKCAL_PER_HR_TO_KWHPH = 1_000_000.0 / 860.4206500956024

DUTY_LABELS = {
    "Electrolyzer power": "power",
    "Stripper hot utility": "stripper utility",
    "Qh1": "H1 heater",
    "Qc1": "C1 cooler",
    "Qr1": "B1 reactor",
    "Qcomp": "MCOMPR1 comp.",
    "QB3": "B3 heater",
    "QB6": "B6 heater",
    "QB27": "B27 heater",
    "QB7_reb": "B7 reb.",
    "QB7_cond": "B7 cond.",
    "QB25_reb": "B25 reb.",
    "QB28_reb": "B28 reb.",
    "QB28_cond": "B28 cond.",
    "QR01": "R01 reactor",
}

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
    "Electrolyzer power": 0.16,
    "Stripper hot utility": 0.34,
    "Qh1": 0.28,
    "Qc1": 0.32,
    "Qr1": 0.36,
    "Qcomp": 0.40,
    "QB3": 0.50,
    "QB6": 0.54,
    "QB27": 0.58,
    "QB7_reb": 0.62,
    "QB7_cond": 0.66,
    "QB25_reb": 0.70,
    "QB28_reb": 0.74,
    "QB28_cond": 0.78,
    "QR01": 0.82,
}


def viridis_colors(labels):
    cmap = plt.get_cmap("viridis")
    return [cmap(VIRIDIS_POINTS[label]) for label in labels]


def color_key(label):
    return label.replace("Capex: ", "").replace("Opex: ", "")


def display_label(label):
    return DUTY_LABELS.get(label, label)


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


def total_npw_components(df):
    annual_costs, capex_total, opex_total, oxygen_credit = cost_components(df)
    present_worth_factor = 1.0 / CRF
    npw_costs = {
        label: value * present_worth_factor
        for label, value in annual_costs.items()
    }
    return (
        npw_costs,
        capex_total * present_worth_factor,
        opex_total * present_worth_factor,
        oxygen_credit * present_worth_factor,
    )


def energy_components(df):
    values = {label: df[column].sum() for label, column in ENERGY_SLICES}
    return {label: value for label, value in values.items() if value > 1e-9}


def nearest_training_row(training_df, target, input_columns):
    working = training_df.copy()
    if "run_ok" in working.columns:
        working = working[working["run_ok"].astype(str).isin(["1", "1.0", "True", "true"])]
    working = working.dropna(subset=input_columns)

    normalized_terms = []
    for column in input_columns:
        values = pd.to_numeric(working[column], errors="coerce")
        span = values.max() - values.min()
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        normalized_terms.append(((values - target[column]) / span) ** 2)

    distances = sum(normalized_terms)
    return working.loc[distances.idxmin()]


def scaled_duty_breakdown(training_row, duty_columns, target_total):
    raw = {
        column: abs(float(training_row[column])) * MMKCAL_PER_HR_TO_KWHPH
        for column in duty_columns
        if column in training_row.index and pd.notna(training_row[column])
    }
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {
        column: target_total * value / total
        for column, value in raw.items()
        if target_total * value / total > 1e-9
    }


def spread_label_positions(
    raw_positions,
    min_gap=DETAILED_LABEL_MIN_GAP,
    lower=-DETAILED_LABEL_Y_LIMIT,
    upper=DETAILED_LABEL_Y_LIMIT,
):
    if not raw_positions:
        return {}

    raw_positions = sorted(raw_positions, key=lambda item: item[1])
    available_gap = (upper - lower) / max(1, len(raw_positions) - 1)
    gap = min(min_gap, available_gap)
    adjusted = [[index, y] for index, y in raw_positions]

    for i in range(1, len(adjusted)):
        adjusted[i][1] = max(adjusted[i][1], adjusted[i - 1][1] + gap)

    if adjusted[-1][1] > upper:
        shift = adjusted[-1][1] - upper
        for item in adjusted:
            item[1] -= shift

    for i in range(len(adjusted) - 2, -1, -1):
        adjusted[i][1] = min(adjusted[i][1], adjusted[i + 1][1] - gap)

    if adjusted[0][1] < lower:
        shift = lower - adjusted[0][1]
        for item in adjusted:
            item[1] += shift

    return {index: y for index, y in adjusted}


def annotate_outer_wedges(ax, wedges, labels, values, groups, colors):
    total = sum(values)
    callouts = []
    for index, wedge in enumerate(wedges):
        theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        x = np.cos(theta)
        y = np.sin(theta)
        callouts.append(
            {
                "index": index,
                "anchor": (
                    DETAILED_OUTER_RADIUS * DETAILED_ANCHOR_SCALE * x,
                    DETAILED_OUTER_RADIUS * DETAILED_ANCHOR_SCALE * y,
                ),
                "raw_y": DETAILED_OUTER_RADIUS * DETAILED_LABEL_Y_SCALE * y,
                "side": "right" if x >= 0 else "left",
                "text": f"{GROUP_LABELS[groups[index]]}: {display_label(labels[index])}\n{values[index] / total * 100:.1f}%",
            }
        )

    for side, text_x, line_x, ha in (
        ("right", DETAILED_CALLOUT_TEXT_X, DETAILED_CALLOUT_ELBOW_X, "left"),
        ("left", -DETAILED_CALLOUT_TEXT_X, -DETAILED_CALLOUT_ELBOW_X, "right"),
    ):
        side_callouts = [item for item in callouts if item["side"] == side]
        y_positions = spread_label_positions(
            [(item["index"], item["raw_y"]) for item in side_callouts]
        )
        for item in side_callouts:
            idx = item["index"]
            anchor_x, anchor_y = item["anchor"]
            text_y = y_positions[idx]
            ax.plot(
                [anchor_x, line_x, text_x - 0.03 * np.sign(line_x)],
                [anchor_y, text_y, text_y],
                color=colors[idx],
                linewidth=DETAILED_LINE_WIDTH,
                solid_capstyle="butt",
                clip_on=False,
            )
            ax.text(
                text_x,
                text_y,
                item["text"],
                ha=ha,
                va="center",
                fontsize=CALLOUT_FONT_SIZE,
                color="0.12",
                clip_on=False,
            )


def draw_wind_only_detailed_energy(wind_df, output_name):
    ammonia_target = {
        "Ft": first_value(wind_df, "Ft_op"),
        "Fh2": first_value(wind_df, "Fh2_op"),
    }
    urea_target = {
        "Fnh3": first_value(wind_df, "Fnh3_op"),
        "Fco2": first_value(wind_df, "Fco2_op"),
    }

    ammonia_row = nearest_training_row(
        pd.read_csv(ROOT / "ammoniaF_results_live.csv"),
        ammonia_target,
        ["Ft", "Fh2"],
    )
    urea_row = nearest_training_row(
        pd.read_csv(ROOT / "ureaF_results_live.csv"),
        urea_target,
        ["Fnh3", "Fco2"],
    )

    group_breakdown = {
        "Electrolyzer": {"Electrolyzer power": wind_df["P_el"].sum()},
        "Ammonia": scaled_duty_breakdown(
            ammonia_row,
            AMMONIA_DUTY_COLUMNS,
            wind_df["E_A"].sum(),
        ),
        "Urea": scaled_duty_breakdown(
            urea_row,
            UREA_DUTY_COLUMNS,
            wind_df["E_U"].sum(),
        ),
    }
    if wind_df["P_HU_el"].sum() > 1e-9:
        group_breakdown["Ammonia"]["Stripper hot utility"] = wind_df["P_HU_el"].sum()

    group_colors = {
        "Electrolyzer": viridis_colors(["Electrolyzer"])[0],
        "Ammonia": viridis_colors(["Ammonia unit"])[0],
        "Urea": viridis_colors(["Urea unit"])[0],
    }
    outer_labels = []
    outer_values = []
    outer_colors = []
    outer_groups = []
    group_totals = []

    for group, parts in group_breakdown.items():
        filtered_parts = dict(
            sorted(
                ((label, value) for label, value in parts.items() if value > 1e-9),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        if not filtered_parts:
            continue
        group_totals.append((group, sum(filtered_parts.values())))
        for label, value in filtered_parts.items():
            outer_labels.append(label)
            outer_values.append(value)
            outer_groups.append(group)

    cmap = plt.get_cmap("viridis")
    outer_colors = [cmap(point) for point in np.linspace(0.12, 0.92, len(outer_labels))]

    fig, ax = plt.subplots(figsize=DETAILED_FIGSIZE)
    inner_values = [value for _, value in group_totals]
    inner_labels = [group for group, _ in group_totals]
    ax.pie(
        inner_values,
        radius=DETAILED_INNER_RADIUS,
        colors=[group_colors[group] for group in inner_labels],
        startangle=90,
        counterclock=False,
        labels=[
            f"{group}\n{value / sum(inner_values) * 100:.0f}%"
            if value / sum(inner_values) >= 0.04
            else ""
            for group, value in group_totals
        ],
        labeldistance=DETAILED_INNER_LABEL_DISTANCE,
        textprops={"fontsize": INNER_LABEL_FONT_SIZE, "fontweight": "bold", "color": "white"},
        wedgeprops={"width": DETAILED_INNER_WIDTH, "edgecolor": "white", "linewidth": 1.0},
    )
    outer_wedges, _ = ax.pie(
        outer_values,
        radius=DETAILED_OUTER_RADIUS,
        colors=outer_colors,
        startangle=90,
        counterclock=False,
        labels=None,
        wedgeprops={"width": DETAILED_OUTER_WIDTH, "edgecolor": "white", "linewidth": 0.7},
    )
    annotate_outer_wedges(ax, outer_wedges, outer_labels, outer_values, outer_groups, outer_colors)
    ax.set(aspect="equal")
    ax.set_xlim(-DETAILED_X_LIMIT, DETAILED_X_LIMIT)
    ax.set_ylim(-DETAILED_Y_LIMIT, DETAILED_Y_LIMIT)
    fig.tight_layout()
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


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

    fig, ax = plt.subplots(figsize=CAPITAL_FIGSIZE)
    for label in active_labels:
        values = np.array([case_values.get(label, 0.0) / 1e6 for _, case_values, *_ in case_data])
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=CAPITAL_BAR_WIDTH,
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
                fontsize=CAPITAL_VALUE_FONT_SIZE,
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
            fontsize=CAPITAL_VALUE_FONT_SIZE,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(case_names)
    ax.tick_params(axis="both", labelsize=CAPITAL_TICK_FONT_SIZE)
    ax.set_ylabel("Annualized capital cost (million GBP/y)", fontsize=CAPITAL_AXIS_FONT_SIZE)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(bottoms) * CAPITAL_Y_HEADROOM)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=CAPITAL_LEGEND_BBOX,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )

    fig.tight_layout(rect=CAPITAL_LAYOUT_RECT)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def format_million_value(value):
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_total_npw_stacked_bar(case_data, output_name):
    active_labels = [
        label
        for label in COST_BAR_ORDER
        if any(abs(values.get(label, 0.0)) > 1e-9 for _, values, *_ in case_data)
    ]
    case_names = [case_name for case_name, *_ in case_data]
    x = np.arange(len(case_names))
    positive_bottoms = np.zeros(len(case_names))
    negative_bottoms = np.zeros(len(case_names))

    fig, ax = plt.subplots(figsize=NPW_FIGSIZE)
    for label in active_labels:
        values = np.array([case_values.get(label, 0.0) / 1e6 for _, case_values, *_ in case_data])
        bottoms = np.where(values >= 0.0, positive_bottoms, negative_bottoms)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=NPW_BAR_WIDTH,
            color=viridis_colors([color_key(label)])[0],
            edgecolor="white",
            linewidth=0.7,
            label=label.replace("Capex: ", ""),
        )

        for case_x, bottom, value in zip(x, bottoms, values):
            if abs(value) < NPW_COMPONENT_LABEL_MIN:
                continue
            ax.text(
                case_x,
                bottom + value / 2.0,
                format_million_value(value),
                ha="center",
                va="center",
                color="white",
                fontsize=NPW_VALUE_FONT_SIZE,
                fontweight="bold",
            )

        positive_bottoms += np.where(values >= 0.0, values, 0.0)
        negative_bottoms += np.where(values < 0.0, values, 0.0)

    net_totals = positive_bottoms + negative_bottoms
    label_offset = max(positive_bottoms) * 0.015
    for case_x, total, positive_top in zip(x, net_totals, positive_bottoms):
        ax.text(
            case_x,
            positive_top + label_offset,
            f"{format_million_value(total)}M",
            ha="center",
            va="bottom",
            fontsize=NPW_VALUE_FONT_SIZE,
        )

    ax.axhline(0.0, color="0.25", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(case_names)
    ax.tick_params(axis="both", labelsize=NPW_TICK_FONT_SIZE)
    ax.set_ylabel("Total NPW (million GBP)", fontsize=NPW_AXIS_FONT_SIZE)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(min(negative_bottoms) * NPW_Y_HEADROOM, max(positive_bottoms) * NPW_Y_HEADROOM)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=NPW_LEGEND_BBOX,
        frameon=False,
        fontsize=NPW_LEGEND_FONT_SIZE,
    )

    fig.tight_layout(rect=NPW_LAYOUT_RECT)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def draw_pie_grid(case_data, output_name, show_cost_total=False):
    if show_cost_total:
        totals = [capex + opex + credit for _, _, capex, opex, credit in case_data]
    else:
        totals = [sum(values.values()) for _, values, *_ in case_data]
    max_total = max(abs(total) for total in totals)
    fig, axes = plt.subplots(2, 2, figsize=ENERGY_FIGSIZE)
    legend_labels = []

    for ax, data, total in zip(axes.flat, case_data, totals):
        case_name, values, *extra = data
        labels = list(values)
        sizes = list(values.values())
        radius = ENERGY_MIN_RADIUS + ENERGY_RADIUS_SPAN * np.sqrt(abs(total) / max_total)
        wedges, _, autotexts = ax.pie(
            sizes,
            colors=viridis_colors([color_key(label) for label in labels]),
            startangle=90,
            counterclock=False,
            radius=radius,
            autopct=autopct_for(),
            pctdistance=0.68,
            wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
            textprops={"fontsize": PIE_PERCENT_FONT_SIZE, "color": "white"},
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
        ax.text(0.0, ENERGY_LABEL_Y, f"{case_name}\n{subtitle}", ha="center", va="top", fontsize=ENERGY_TEXT_FONT_SIZE)
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
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=ENERGY_LEGEND_BBOX,
    )
    fig.tight_layout(rect=ENERGY_LAYOUT_RECT)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    cost_data = []
    total_npw_data = []
    energy_data = []
    wind_only_df = None

    for case_name, path in CASES:
        df = pd.read_csv(path)
        costs, capex_total, opex_total, oxygen_credit = cost_components(df)
        npw_costs, npw_capex_total, npw_opex_total, npw_oxygen_credit = total_npw_components(df)
        cost_data.append((case_name, costs, capex_total, opex_total, oxygen_credit))
        total_npw_data.append(
            (case_name, npw_costs, npw_capex_total, npw_opex_total, npw_oxygen_credit)
        )
        energy_data.append((case_name, energy_components(df)))
        if case_name == "Wind only":
            wind_only_df = df

    cost_path = draw_capital_stacked_bar(
        cost_data,
        "capital_cost_distribution_stacked_bars.png",
    )
    total_npw_path = draw_total_npw_stacked_bar(
        total_npw_data,
        "total_npw_cost_allocation_bars.png",
    )
    energy_path = draw_pie_grid(
        energy_data,
        "major_energy_consumption_slices.png",
    )
    detailed_energy_path = draw_wind_only_detailed_energy(
        wind_only_df,
        "wind_only_detailed_energy_duty_breakdown.png",
    )

    print(f"Wrote {cost_path}")
    print(f"Wrote {total_npw_path}")
    print(f"Wrote {energy_path}")
    print(f"Wrote {detailed_energy_path}")


if __name__ == "__main__":
    main()

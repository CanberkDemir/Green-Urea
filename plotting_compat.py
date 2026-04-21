from __future__ import annotations

import numpy as np
import pandas as pd


ALIAS_GROUPS = (
    ("y_A_on", "y_NH3_on"),
    ("E_A", "E_N"),
    ("Fh2", "F_H2"),
    ("Fnh3", "F_NH3"),
    ("Fco2", "F_CO2"),
    ("F_NH3_strip", "M_NH3_out"),
    ("F_U", "M_U"),
)

# Match the optimization-side expressions used when detailed electric auxiliaries
# are not exported in the results CSV.
H_STORAGE_CHARGE_POWER_PER_KGPH = 2.0
H_STORAGE_DISCHARGE_POWER_PER_KGPH = 0.1
CO2_FEED_POWER_PER_KGPH = 0.02


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def first_existing_column(df: pd.DataFrame, candidates) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of these columns were found: {', '.join(candidates)}")


def _copy_alias_group(df: pd.DataFrame, columns) -> None:
    source = next((col for col in columns if col in df.columns), None)
    if source is None:
        return

    values = numeric_series(df, source)
    for column in columns:
        if column not in df.columns:
            df[column] = values


def _first_available_series(df: pd.DataFrame, candidates) -> pd.Series | None:
    source = next((col for col in candidates if col in df.columns), None)
    if source is None:
        return None
    return numeric_series(df, source)


def _sum_available_series(df: pd.DataFrame, candidate_groups) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype=float)
    for candidates in candidate_groups:
        series = _first_available_series(df, candidates)
        if series is not None:
            total = total + series
    return total


def _derive_h2_storage_electricity(df: pd.DataFrame) -> pd.Series | None:
    explicit = _first_available_series(df, ("P_H_storage", "e_Hstor"))
    if explicit is not None:
        return explicit

    if "ch_H" not in df.columns and "dis_H" not in df.columns:
        return None

    ch = numeric_series(df, "ch_H") if "ch_H" in df.columns else pd.Series(0.0, index=df.index, dtype=float)
    dis = numeric_series(df, "dis_H") if "dis_H" in df.columns else pd.Series(0.0, index=df.index, dtype=float)
    return H_STORAGE_CHARGE_POWER_PER_KGPH * ch + H_STORAGE_DISCHARGE_POWER_PER_KGPH * dis


def _derive_co2_storage_electricity(df: pd.DataFrame) -> pd.Series | None:
    explicit = _first_available_series(df, ("P_CO2_feed", "e_Cstor"))
    if explicit is not None:
        return explicit

    feed = _first_available_series(df, ("Fco2", "F_CO2"))
    if feed is None:
        return None
    return CO2_FEED_POWER_PER_KGPH * feed


def _derive_direct_ammonia_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    if not {"F_NH3_A", "F_H2O_A"}.issubset(df.columns):
        return None

    nh3 = numeric_series(df, "F_NH3_A")
    water = numeric_series(df, "F_H2O_A")
    hydrous = nh3 + water

    fallback_wtfrac = 0.0
    if {"F_NH3_A_op", "F_H2O_A_op"}.issubset(df.columns):
        nh3_op = numeric_series(df, "F_NH3_A_op")
        water_op = numeric_series(df, "F_H2O_A_op")
        total_op = nh3_op + water_op
        positive = total_op > 0.0
        if positive.any():
            fallback_wtfrac = float((nh3_op[positive] / total_op[positive]).iloc[0])

    hydrous_values = hydrous.to_numpy(dtype=float)
    nh3_values = nh3.to_numpy(dtype=float)
    wtfrac_values = np.full(len(hydrous_values), fallback_wtfrac, dtype=float)
    np.divide(nh3_values, hydrous_values, out=wtfrac_values, where=hydrous_values > 0.0)
    wtfrac = pd.Series(wtfrac_values, index=df.index, dtype=float)
    return hydrous, wtfrac


def ensure_before_stripper_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    has_legacy_hydrous = any(col in work.columns for col in ("F_HA", "M_HA", "M_HA_sur"))
    has_legacy_wtfrac = any(col in work.columns for col in ("w_HA", "w_HA_sur"))

    if has_legacy_hydrous and has_legacy_wtfrac:
        return work

    derived = _derive_direct_ammonia_columns(work)
    if derived is None:
        return work

    hydrous, wtfrac = derived
    for column in ("F_HA", "M_HA", "M_HA_sur"):
        if column not in work.columns:
            work[column] = hydrous
    for column in ("w_HA", "w_HA_sur"):
        if column not in work.columns:
            work[column] = wtfrac
    return work


def normalize_result_columns(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    for group in ALIAS_GROUPS:
        _copy_alias_group(df, group)

    if "P_HU_el" not in df.columns and "Q_HU" in df.columns:
        df["P_HU_el"] = numeric_series(df, "Q_HU")

    if "P_wind_to_batt" not in df.columns and "ch_B" in df.columns:
        df["P_wind_to_batt"] = numeric_series(df, "ch_B")

    if "e_Hstor" not in df.columns:
        derived_h2_storage = _derive_h2_storage_electricity(df)
        if derived_h2_storage is not None:
            df["e_Hstor"] = derived_h2_storage

    if "e_Cstor" not in df.columns:
        derived_co2_storage = _derive_co2_storage_electricity(df)
        if derived_co2_storage is not None:
            df["e_Cstor"] = derived_co2_storage

    if "total_electric_load" not in df.columns:
        # Do not invent missing auxiliary columns as zeros here; leaving them
        # absent lets the notebook's residual bar expose any still-unreported use.
        total = _sum_available_series(
            df,
            (
                ("P_el",),
                ("E_A", "E_N"),
                ("E_U",),
                ("P_HU_el",),
                ("ch_B",),
                ("e_Hstor",),
                ("e_Cstor",),
                ("e_Wstor",),
            ),
        )
        df["total_electric_load"] = total

    return ensure_before_stripper_columns(df)


def get_before_stripper_series(df: pd.DataFrame) -> dict[str, pd.Series | str]:
    work = ensure_before_stripper_columns(df)

    hydrous_col = first_existing_column(work, ("F_HA", "M_HA", "M_HA_sur"))
    wtfrac_col = first_existing_column(work, ("w_HA", "w_HA_sur"))

    hydrous = numeric_series(work, hydrous_col)
    wtfrac_raw = numeric_series(work, wtfrac_col)
    wtfrac = wtfrac_raw / 100.0 if float(wtfrac_raw.max()) > 1.0 else wtfrac_raw

    return {
        "hydrous_col": hydrous_col,
        "wtfrac_col": wtfrac_col,
        "hydrous_kgph": hydrous,
        "nh3_wtfrac": wtfrac,
    }

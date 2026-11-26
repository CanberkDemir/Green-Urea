"""Provisional Ni2P gas-liquid packed-bed PFR for A4.

Uses the batch-derived apparent rate and NH3 selectivity recorded in
docs/A4_Ni2P_PFR_Parameters.md. No Pd-Cu kinetic terms are used. The
H2 transfer coefficients, H2 saturation form, supported-catalyst activity,
and brine selectivity are engineering scenarios rather than fitted data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from math import expm1, isfinite, pi

if __package__:
    from .h2_transfer_reactor import MW_H2, MW_NH3, MW_NO3, h2_solubility_mol_m3
else:
    from h2_transfer_reactor import MW_H2, MW_NH3, MW_NO3, h2_solubility_mol_m3


@dataclass(frozen=True)
class Inputs:
    brine_kgph: float = 8500.0
    water_kgph: float = 7621.015
    chloride_kgph: float = 453.9
    nitrate_kgph: float = 78.625
    brine_density_kg_m3: float = 1030.0  # assumed
    h2_kgph: float = 10.0
    pressure_bar: float = 27.0  # proposed plant scenario
    temperature_k: float = 333.15
    sechenov_kg_mol: float = 0.10  # assumed mixed-brine NaCl proxy
    tubes: int = 1  # provisional single bed in A4 parameter document
    diameter_m: float = 1.0
    length_m: float = 10.0
    bed_void_fraction: float = 0.40
    liquid_saturation: float = 0.50  # liquid share of bed voids
    packed_bulk_density_kg_m3: float = 600.0
    active_fraction: float = 0.122  # active Ni2P per kg supported catalyst
    activity_factor: float = 1.0  # unvalidated supported-catalyst effectiveness
    k_w_m3_kg_s: float = 9.08e-4  # batch-derived at pH 2, 60 C, 0.075 atm H2
    nh3_selectivity: float = 0.96  # batch reference, unvalidated in brine
    h2_half_saturation_mol_m3: float | None = None  # default = lab reference C*
    kla_s: float = 0.001  # gas/liquid transfer, per liquid volume; assumed
    ksa_s: float = 0.01  # liquid/catalyst transfer, per liquid volume; assumed
    steps: int = 4000


@dataclass(frozen=True)
class Result:
    equilibrium_h2_in_mol_m3: float
    reference_h2_mol_m3: float
    nacl_molal: float
    liquid_flow_m3_s: float
    liquid_volume_m3: float
    supported_catalyst_kg: float
    active_ni2p_kg: float
    nitrate_conversion: float
    nh3_kgph: float
    nitrate_out_kgph: float
    h2_gas_out_kgph: float
    h2_liquid_out_kgph: float
    h2_consumed_kgph: float
    n2_gas_out_mol_s: float
    acid_equivalents_mol_s: float
    surface_h2_out_mol_m3: float
    h2_balance_error_mol_s: float
    nitrogen_balance_error_mol_s: float


def run(p: Inputs) -> Result:
    positive = (p.brine_kgph, p.water_kgph, p.brine_density_kg_m3,
                p.pressure_bar, p.temperature_k, p.tubes, p.diameter_m,
                p.length_m, p.packed_bulk_density_kg_m3, p.steps)
    nonnegative = (p.chloride_kgph, p.nitrate_kgph, p.h2_kgph,
                   p.sechenov_kg_mol, p.activity_factor, p.k_w_m3_kg_s,
                   p.kla_s, p.ksa_s)
    if any(not isfinite(x) or x <= 0 for x in positive):
        raise ValueError("flow, pressure, temperature and bed geometry must be positive")
    if any(not isfinite(x) or x < 0 for x in nonnegative):
        raise ValueError("feeds and coefficients must be nonnegative")
    if not 0 < p.bed_void_fraction < 1 or not 0 < p.liquid_saturation <= 1:
        raise ValueError("bed void fraction and liquid saturation must be between 0 and 1")
    if not 0 < p.active_fraction <= 1 or not 0 <= p.nh3_selectivity <= 1:
        raise ValueError("active fraction must be in (0,1], selectivity in [0,1]")
    if p.brine_kgph < p.water_kgph + p.chloride_kgph + p.nitrate_kgph:
        raise ValueError("brine mass flow is below the sum of listed components")
    nacl_molal = p.chloride_kgph / 0.035453 / p.water_kgph
    if nacl_molal > 6:
        raise ValueError("NaCl-equivalent molality exceeds single-phase brine range")
    q_liq = p.brine_kgph / p.brine_density_kg_m3 / 3600
    v_bed = p.tubes * pi * p.diameter_m**2 / 4 * p.length_m
    v_liq = v_bed * p.bed_void_fraction * p.liquid_saturation
    w_supported = v_bed * p.packed_bulk_density_kg_m3
    w_active = w_supported * p.active_fraction
    dt = v_liq / q_liq / p.steps
    c_ref = h2_solubility_mol_m3(0.075 * 1.01325, p.temperature_k, 0, 0)
    k_half = c_ref if p.h2_half_saturation_mol_m3 is None else p.h2_half_saturation_mol_m3
    if not isfinite(k_half) or k_half <= 0:
        raise ValueError("H2 half-saturation concentration must be positive")
    f_ref = c_ref / (c_ref + k_half)
    c_star_in = h2_solubility_mol_m3(p.pressure_bar, p.temperature_k,
                                    nacl_molal, p.sechenov_kg_mol)
    gas_h2 = p.h2_kgph / MW_H2 / 3600
    gas_h2_in = gas_h2
    gas_n2 = 0.0
    c_h2 = 0.0  # H2 enters as gas
    c_no3 = p.nitrate_kgph / MW_NO3 / 3600 / q_liq
    c_no3_in = c_no3
    c_nh3 = 0.0
    reacted_h2 = 0.0
    acid_equivalents = 0.0
    surface_h2 = 0.0
    h2_per_no3 = 2.5 + 1.5 * p.nh3_selectivity
    active_density_liquid = w_active / v_liq

    def transfer(g_h2: float, g_n2: float, c_liq: float, duration: float):
        if p.kla_s == 0 or duration == 0 or g_h2 + g_n2 <= 0:
            return g_h2, c_liq
        partial = p.pressure_bar * g_h2 / (g_h2 + g_n2)
        c_eq = h2_solubility_mol_m3(partial, p.temperature_k,
                                    nacl_molal, p.sechenov_kg_mol)
        delta = (c_eq - c_liq) * (-expm1(-p.kla_s * duration))
        delta = max(-c_liq, min(delta, g_h2 / q_liq))
        return g_h2 - delta * q_liq, c_liq + delta

    for _ in range(p.steps):
        gas_h2, c_h2 = transfer(gas_h2, gas_n2, c_h2, dt / 2)

        def rate(c_surface: float) -> float:
            if c_no3 <= 0 or c_surface <= 0:
                return 0.0
            h2_factor = (c_surface / (c_surface + k_half)) / f_ref
            return p.activity_factor * active_density_liquid * p.k_w_m3_kg_s * c_no3 * h2_factor

        if p.ksa_s == 0 or c_h2 == 0:
            surface_h2 = 0.0
        else:
            lo, hi = 0.0, c_h2
            for _ in range(35):
                mid = (lo + hi) / 2
                if p.ksa_s * (c_h2 - mid) > h2_per_no3 * rate(mid):
                    lo = mid
                else:
                    hi = mid
            surface_h2 = (lo + hi) / 2
        converted = min(rate(surface_h2) * dt, c_no3, c_h2 / h2_per_no3)
        c_no3 -= converted
        c_nh3 += p.nh3_selectivity * converted
        gas_n2 += (1 - p.nh3_selectivity) * converted * q_liq / 2
        c_h2 -= h2_per_no3 * converted
        reacted_h2 += h2_per_no3 * converted * q_liq
        acid_equivalents += (1 + p.nh3_selectivity) * converted * q_liq
        gas_h2, c_h2 = transfer(gas_h2, gas_n2, c_h2, dt / 2)

    h2_out = gas_h2 + c_h2 * q_liq
    n_in = c_no3_in * q_liq
    n_out = c_no3 * q_liq + c_nh3 * q_liq + 2 * gas_n2
    return Result(c_star_in, c_ref, nacl_molal, q_liq, v_liq,
                  w_supported, w_active, 1 - c_no3 / c_no3_in if c_no3_in else 0,
                  c_nh3 * q_liq * MW_NH3 * 3600,
                  c_no3 * q_liq * MW_NO3 * 3600,
                  gas_h2 * MW_H2 * 3600, c_h2 * q_liq * MW_H2 * 3600,
                  reacted_h2 * MW_H2 * 3600, gas_n2, acid_equivalents,
                  surface_h2, gas_h2_in - h2_out - reacted_h2, n_in - n_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in (("tubes", int), ("pressure-bar", float), ("h2-kgph", float),
                       ("kla-s", float), ("ksa-s", float), ("activity-factor", float),
                       ("k-w-m3-kg-s", float), ("nh3-selectivity", float),
                       ("h2-half-saturation-mol-m3", float), ("liquid-saturation", float),
                       ("length-m", float), ("diameter-m", float)):
        parser.add_argument("--" + name, type=kind)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    p = replace(Inputs(), **args)
    r = run(p)
    print("Provisional Ni2P PFR; batch-derived kinetics, assumed H2 transfer")
    print(f"Bed / active Ni2P          {r.supported_catalyst_kg:.0f} / {r.active_ni2p_kg:.0f} kg")
    print(f"Liquid volume / flow       {r.liquid_volume_m3:.3f} m3 / {r.liquid_flow_m3_s*3600:.3f} m3/h")
    print(f"H2 equilibrium inlet       {r.equilibrium_h2_in_mol_m3:.3f} mol/m3")
    print(f"NO3 conversion             {100*r.nitrate_conversion:.2f}%")
    print(f"NH3 product                {r.nh3_kgph:.3f} kg/h")
    print(f"H2 gas / dissolved out     {r.h2_gas_out_kgph:.3f} / {r.h2_liquid_out_kgph:.3f} kg/h")
    print(f"H2 reacted                 {r.h2_consumed_kgph:.3f} kg/h")
    print(f"H2 / N balance residual    {r.h2_balance_error_mol_s:.2e} / {r.nitrogen_balance_error_mol_s:.2e} mol/s")


if __name__ == "__main__":
    main()

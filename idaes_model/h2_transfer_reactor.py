"""Screening model for H2 delivery to an aqueous nitrate hydrogenation bed.

This is deliberately separate from the Aspen-replica IDAES flowsheet. It
tracks gas H2, dissolved H2, nitrate, nitrite, NH3 and N2 along a wetted bed.
The chemistry uses charge-balanced net reactions; kinetic prefactors and
NH3 branching must be measured for the chosen catalyst and brine.

Run ``python idaes_model/h2_transfer_reactor.py --help`` for inputs.
No third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import exp, expm1, isfinite, pi


# Solubility form: C* = Hcp(T) p_H2 10**(-k_s m_NaCl).
# Sander (2023), H2 in pure water at 298.15 K: Hcp = 7.8e-6 mol/m3/Pa,
# d ln(Hcp)/d(1/T) = 500 K. The salt coefficient is an exposed SCENARIO
# parameter, not a fit to this mixed brine. See README for sources/limits.
HCP_298 = 7.8e-6  # mol / m3 / Pa
HCP_T_COEFF = 500.0  # K
MW_H2 = 2.01588e-3  # kg / mol
MW_NO3 = 62.0049e-3  # kg / mol
MW_NH3 = 17.0305e-3  # kg / mol


@dataclass(frozen=True)
class Inputs:
    water_kgph: float = 7621.015
    chloride_kgph: float = 453.9
    nitrate_kgph: float = 78.625
    total_brine_kgph: float = 8500.0
    brine_density_kg_m3: float = 1030.0  # scenario assumption
    h2_kgph: float = 10.0
    pressure_bar: float = 27.0
    temperature_k: float = 333.15
    na_cl_sechenov_kg_mol: float = 0.10  # scenario assumption
    kla_s: float = 0.001  # scenario assumption; gas-to-liquid, per liquid volume
    ksa_s: float = 0.01  # scenario assumption; liquid-to-catalyst H2 transfer
    tubes: int = 15
    tube_diameter_m: float = 1.0
    bed_length_m: float = 10.0
    liquid_holdup: float = 0.5  # fraction of geometric bed volume occupied by liquid
    steps: int = 4000
    # Apparent rate at reference concentrations, mol N/(m3 liquid s).
    # Exponents are from Mikami et al. (2003); prefactors are unspecified.
    k_no3_ref: float = 0.0
    k_no2_ref: float = 0.0
    ammonia_branch: float = 0.5  # fraction of reacted nitrite N going to NH3


@dataclass(frozen=True)
class ProfilePoint:
    z_m: float
    gas_h2_mol_s: float
    liquid_h2_mol_m3: float
    surface_h2_mol_m3: float
    nitrate_mol_m3: float
    nitrite_mol_m3: float
    ammonia_mol_m3: float
    nitrogen_mol_m3: float
    gas_liquid_transfer_mol_m3_s: float
    nitrate_rate_mol_m3_s: float
    nitrite_rate_mol_m3_s: float


@dataclass(frozen=True)
class Result:
    nacl_molal: float
    liquid_m3_s: float
    liquid_volume_m3: float
    residence_s: float
    c_star_in_mol_m3: float
    h2_gas_out_mol_s: float
    h2_liquid_out_mol_m3: float
    nitrate_out_mol_m3: float
    nitrite_out_mol_m3: float
    nh3_out_mol_m3: float
    n2_out_mol_m3: float
    h2_reacted_mol_s: float
    h2_transferred_mol_s: float
    h2_transfer_max_in_mol_s: float
    surface_h2_near_outlet_mol_m3: float
    profile: tuple[ProfilePoint, ...]


def h2_solubility_mol_m3(pressure_h2_bar: float, temperature_k: float,
                         nacl_molal: float, sechenov_kg_mol: float) -> float:
    """Equilibrium dissolved H2; dilute Henry/Sechenov screening form."""
    if min(pressure_h2_bar, nacl_molal, sechenov_kg_mol) < 0 or temperature_k <= 0:
        raise ValueError("pressure, salt and Sechenov coefficient must be nonnegative; T > 0")
    hcp = HCP_298 * exp(HCP_T_COEFF * (1 / temperature_k - 1 / 298.15))
    return hcp * pressure_h2_bar * 1e5 * 10 ** (-sechenov_kg_mol * nacl_molal)


def run(inputs: Inputs) -> Result:
    """Integrate a steady cocurrent gas/liquid PFR using split finite steps.

    Gas H2 transfers into a well-mixed liquid cross-section, then across a
    lumped liquid-to-catalyst film. A quasi-steady surface H2 concentration
    balances that film flux against reaction demand. Gas is assumed H2-only
    and its partial pressure equals total pressure while H2 remains. Each
    axial cell conserves H2 and N. Gas composition, pressure drop, nitrate
    film transport and catalyst pore diffusion are omitted.
    """
    positive = (inputs.water_kgph, inputs.total_brine_kgph,
                inputs.brine_density_kg_m3, inputs.pressure_bar,
                inputs.temperature_k, inputs.tubes, inputs.tube_diameter_m,
                inputs.bed_length_m, inputs.steps)
    if any(v <= 0 or not isfinite(v) for v in positive):
        raise ValueError("water, flow, density, pressure, temperature and geometry must be positive")
    nonnegative = (inputs.chloride_kgph, inputs.nitrate_kgph, inputs.h2_kgph,
                   inputs.na_cl_sechenov_kg_mol, inputs.kla_s, inputs.ksa_s,
                   inputs.k_no3_ref, inputs.k_no2_ref)
    if any(v < 0 or not isfinite(v) for v in nonnegative):
        raise ValueError("feed and rate parameters must be nonnegative and finite")
    if not 0 < inputs.liquid_holdup <= 1 or not 0 <= inputs.ammonia_branch <= 1:
        raise ValueError("liquid holdup must be in (0,1]; NH3 branch in [0,1]")
    if inputs.total_brine_kgph < inputs.water_kgph + inputs.chloride_kgph + inputs.nitrate_kgph:
        raise ValueError("total brine flow is smaller than listed components")

    # Chloride is used as an NaCl-equivalent proxy. Other salts are omitted
    # from the salting-out correction; this is not a full mixed-salt model.
    chloride_mol_s = inputs.chloride_kgph / 35.453e-3 / 3600
    nacl_molal = chloride_mol_s * 3600 / inputs.water_kgph
    if nacl_molal > 6:
        raise ValueError("NaCl-equivalent molality exceeds 6 mol/kg; a single-phase brine is not credible")
    q_liq = inputs.total_brine_kgph / inputs.brine_density_kg_m3 / 3600
    v_liq = inputs.tubes * pi * (inputs.tube_diameter_m / 2) ** 2 * inputs.bed_length_m * inputs.liquid_holdup
    residence_s = v_liq / q_liq
    dt = residence_s / inputs.steps
    c_star = h2_solubility_mol_m3(inputs.pressure_bar, inputs.temperature_k,
                                 nacl_molal, inputs.na_cl_sechenov_kg_mol)
    gas = inputs.h2_kgph / MW_H2 / 3600
    gas_in = gas
    c_h2 = 0.0  # initially H2-free brine; inlet gas remains a separate phase
    c_no3 = inputs.nitrate_kgph / MW_NO3 / 3600 / q_liq
    c_no2 = c_nh3 = c_n2 = 0.0
    consumed_h2 = 0.0
    surface_h2 = 0.0
    profile = [ProfilePoint(0, gas, c_h2, 0, c_no3, 0, 0, 0,
                            inputs.kla_s * c_star, 0, 0)]

    def transfer(gas_flow: float, liquid_c: float, duration: float) -> tuple[float, float]:
        if inputs.kla_s == 0 or duration == 0 or gas_flow <= 0:
            return gas_flow, liquid_c
        # Exact relaxation for constant gas p during this small step.
        # If gas is depleted, transfer is limited by its remaining inventory.
        delta_c = (c_star - liquid_c) * (-expm1(-inputs.kla_s * duration))
        delta_c = min(delta_c, gas_flow / q_liq)
        delta_c = max(delta_c, -liquid_c)
        return gas_flow - delta_c * q_liq, liquid_c + delta_c

    for step in range(inputs.steps):
        gas, c_h2 = transfer(gas, c_h2, dt / 2)
        # Mikami apparent orders in p_H2 are represented by C_H2 here only
        # under local Henry equilibrium. These exponents do not define the
        # missing prefactors or NH3/N2 branching for this brine.
        h2_for_r2 = 1.5 + 1.5 * inputs.ammonia_branch

        def rates(surface_h2: float) -> tuple[float, float]:
            if surface_h2 <= 0:
                return 0.0, 0.0
            factor = surface_h2 / 10.0
            r_no3 = inputs.k_no3_ref * (c_no3 / 100.0) ** 0.2 * factor ** 0.2 if c_no3 > 0 else 0.0
            r_no2 = inputs.k_no2_ref * (c_no2 / 100.0) * factor ** 0.3 if c_no2 > 0 else 0.0
            return r_no3, r_no2

        # ksa (C_bulk-C_surface) = H2 demand at the catalyst. Bisection
        # gives a nonnegative surface concentration even for strong kinetics.
        if inputs.ksa_s == 0:
            surface_h2 = 0.0
        elif not (inputs.k_no3_ref or inputs.k_no2_ref):
            surface_h2 = c_h2
        else:
            lo, hi = 0.0, c_h2
            for _ in range(35):
                mid = (lo + hi) / 2
                r_no3, r_no2 = rates(mid)
                if inputs.ksa_s * (c_h2 - mid) > r_no3 + h2_for_r2 * r_no2:
                    lo = mid
                else:
                    hi = mid
            surface_h2 = (lo + hi) / 2
        r1, r2 = rates(surface_h2)
        d1 = min(r1 * dt, c_no3)
        d2 = min(r2 * dt, c_no2 + d1)
        h2_need = d1 + h2_for_r2 * d2
        if h2_need > c_h2:
            scale = c_h2 / h2_need
            d1 *= scale
            d2 *= scale
            h2_need = c_h2
        c_no3 -= d1
        c_no2 += d1 - d2
        c_nh3 += inputs.ammonia_branch * d2
        c_n2 += (1 - inputs.ammonia_branch) * d2 / 2
        c_h2 -= h2_need
        consumed_h2 += h2_need * q_liq
        gas, c_h2 = transfer(gas, c_h2, dt / 2)
        if (step + 1) % max(1, inputs.steps // 100) == 0 or step + 1 == inputs.steps:
            profile.append(ProfilePoint(
                inputs.bed_length_m * (step + 1) / inputs.steps,
                gas, c_h2, surface_h2, c_no3, c_no2, c_nh3, c_n2,
                inputs.kla_s * (c_star - c_h2) if gas > 0 else 0,
                d1 / dt, d2 / dt,
            ))

    transferred_h2 = gas_in - gas
    return Result(nacl_molal, q_liq, v_liq, residence_s, c_star,
                  gas, c_h2, c_no3, c_no2, c_nh3, c_n2,
                  consumed_h2, transferred_h2, inputs.kla_s * c_star * v_liq,
                  surface_h2, tuple(profile))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind, help_text in (
        ("water-kgph", float, "water in brine; default is un-concentrated feed"),
        ("chloride-kgph", float, "chloride in brine, used for NaCl-equivalent salinity"),
        ("nitrate-kgph", float, "nitrate in brine"),
        ("total-brine-kgph", float, "total liquid feed mass flow"),
        ("brine-density-kg-m3", float, "assumed liquid density"),
        ("h2-kgph", float, "gaseous H2 feed"),
        ("pressure-bar", float, "H2 partial pressure if gas is pure H2"),
        ("temperature-k", float, "reactor temperature"),
        ("kla-s", float, "gas-liquid volumetric transfer coefficient (assumed)"),
        ("ksa-s", float, "liquid-catalyst H2 transfer coefficient (assumed)"),
        ("sechenov", float, "NaCl salting-out coefficient (assumed; kg/mol)"),
        ("liquid-holdup", float, "fraction of reactor geometric volume occupied by liquid"),
        ("tubes", int, "number of reactor tubes"),
        ("tube-diameter-m", float, "inner diameter of each tube"),
        ("bed-length-m", float, "reactor length"),
        ("k-no3", float, "nitrate rate at reference concentrations (mol/m3/s)"),
        ("k-no2", float, "nitrite rate at reference concentrations (mol/m3/s)"),
        ("ammonia-branch", float, "fraction of converted nitrite N to NH3"),
        ("steps", int, "axial integration steps"),
    ):
        parser.add_argument("--" + name, type=kind, help=help_text)
    parser.add_argument("--csv", help="write sampled axial profile to this CSV path")
    args = vars(parser.parse_args())
    csv_path = args.pop("csv")
    aliases = {"sechenov": "na_cl_sechenov_kg_mol", "k_no3": "k_no3_ref", "k_no2": "k_no2_ref"}
    kwargs = {aliases.get(k, k): v for k, v in args.items() if v is not None}
    p = Inputs(**kwargs)
    r = run(p)
    gas_in = p.h2_kgph / MW_H2 / 3600
    nitrate_in = p.nitrate_kgph / MW_NO3 / 3600
    print("H2 gas-liquid-catalyst screening model (not a calibrated NH3 prediction)")
    print(f"NaCl-equivalent salinity      {r.nacl_molal:.3f} mol/kg water")
    print(f"H2 equilibrium in brine      {r.c_star_in_mol_m3:.3f} mol/m3 ({r.c_star_in_mol_m3 * MW_H2 * 1000:.2f} mg/L)")
    print(f"Liquid flow / hold-up         {r.liquid_m3_s * 3600:.3f} m3/h / {r.liquid_volume_m3:.2f} m3")
    print(f"Liquid residence             {r.residence_s / 3600:.2f} h")
    print(f"Catalyst-surface H2 near out {r.surface_h2_near_outlet_mol_m3:.3f} mol/m3")
    print(f"H2 at liquid outlet          {r.h2_liquid_out_mol_m3:.3f} mol/m3 ({r.h2_liquid_out_mol_m3 * r.liquid_m3_s * MW_H2 * 3600:.3f} kg/h)")
    print(f"H2 gas out / reacted         {r.h2_gas_out_mol_s * MW_H2 * 3600:.3f} / {r.h2_reacted_mol_s * MW_H2 * 3600:.3f} kg/h")
    print(f"H2 transferred from gas      {r.h2_transferred_mol_s * MW_H2 * 3600:.3f} kg/h ({100*r.h2_transferred_mol_s/gas_in if gas_in else 0:.1f}% of feed)")
    print(f"Inlet transfer ceiling       {r.h2_transfer_max_in_mol_s * MW_H2 * 3600:.3f} kg/h at C_liq=0")
    print(f"kLa needed for inlet ceiling {gas_in/(r.c_star_in_mol_m3*r.liquid_volume_m3) if r.c_star_in_mol_m3 else float('inf'):.4g} 1/s to match H2 feed")
    print(f"NO3 conversion               {100*(1-r.nitrate_out_mol_m3*r.liquid_m3_s/nitrate_in) if nitrate_in else 0:.2f}%")
    print(f"NH3 produced                 {r.nh3_out_mol_m3*r.liquid_m3_s*MW_NH3*3600:.3f} kg/h")
    if p.k_no3_ref or p.k_no2_ref:
        print("KINETIC PREFACTORS AND NH3 BRANCH ARE USER SCENARIOS, NOT FITTED TO THIS CATALYST/BRINE.")
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(ProfilePoint.__dataclass_fields__)
            writer.writerows(tuple(vars(point).values()) for point in r.profile)
        print(f"Axial profile written to {csv_path}")


if __name__ == "__main__":
    main()

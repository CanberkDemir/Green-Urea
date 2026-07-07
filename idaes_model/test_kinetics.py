"""
Standalone verification of the nitrate-hydrogenation kinetics.

Two checks, runnable directly (``python test_kinetics.py``):

1. Integrate the MAIN1/MAIN2/SIDE1/SIDE2 ODE network (as implemented in
   aspen/ammoniaF.bkp - see reactions/ammonia_reactions.py for the
   documented discrepancy with SI Table S7) at the reactor temperature
   (60 degC) and confirm near-complete NO3- conversion, negligible nitrite
   accumulation, and an ultimate NH3 selectivity of ~55% - consistent with
   the recorded Aspen results (NH3_out 12.75 kg/h at Ft=8500/Fh2=10, i.e.
   59% realized yield vs 21.6 kg/h stoichiometric maximum).

2. Build the IDAES GenericReactionParameterBlock against the liquid-only
   property package and check the rate constants and first-order rate form
   match the hand-computed values (unit-consistency check).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

R_CAL = 8.314462618 / 4.184  # cal/mol/K (thermochemical calorie, matches
# the 4.184 J/cal conversion used in reactions/ammonia_reactions.py)
T_REACTOR = 333.15  # K (60 degC, Aspen H1/B1 spec)

# aspen/ammoniaF.bkp REAC-DATA (PRE-EXP [1/s], Ea [cal/mol], consumed ion,
# ion moles per extent)
NETWORK = {
    "MAIN1": (3.0e3, 9560.0, "NO3-", 1),
    "MAIN2": (3.5e5, 9000.0, "NO2-", 2),
    "SIDE1": (8.2e3, 11970.0, "NO2-", 2),
    "SIDE2": (2.0e2, 8370.0, "NO3-", 2),
}


def ion_consumption_constants(T):
    """Per-ion first-order constants d[ion]/dt = -k_eff*C (T-EXP folded in).

    Aspen extent rate = k0 * T * exp(-Ea/RT) * C_ion consumes
    `ion_per_extent` moles of the ion per extent.
    """
    out = {}
    for name, (k0, ea, ion, n_ion) in NETWORK.items():
        out[name] = k0 * n_ion * T * np.exp(-ea / (R_CAL * T))
    return out


def check_ode_selectivity():
    k = ion_consumption_constants(T_REACTOR)
    print("per-ion consumption constants at 60 degC [1/s]:")
    for r in sorted(k):
        print(f"  {r}: {k[r]:.4e}")

    # analytic branching of the linear network:
    #   NO3- splits MAIN1 : SIDE2, NO2- splits MAIN2 : SIDE1
    s_no3 = k["MAIN1"] / (k["MAIN1"] + k["SIDE2"])
    s_no2 = k["MAIN2"] / (k["MAIN2"] + k["SIDE1"])
    s_nh3 = s_no3 * s_no2
    print(f"branch NO3->NO2: {s_no3:.4f}   branch NO2->NH3: {s_no2:.4f}")
    print(f"analytic ultimate NH3 selectivity: {s_nh3:.4f}")

    # implicit-Euler ODE integration as an independent check (the network is
    # stiff: k_MAIN2 ~ 3e2 1/s with the T prefactor folded in)
    c_no3, c_no2, c_nh3, c_n2 = 1.0, 0.0, 0.0, 0.0
    t, dt, t_end = 0.0, 0.01, 60.0
    while t < t_end:
        c_no3 = c_no3 / (1.0 + (k["MAIN1"] + k["SIDE2"]) * dt)
        c_no2 = (c_no2 + dt * k["MAIN1"] * c_no3) / (
            1.0 + (k["MAIN2"] + k["SIDE1"]) * dt)
        c_nh3 += dt * k["MAIN2"] * c_no2  # 1 NH3 per NO2- consumed via MAIN2
        c_n2 += dt * 0.5 * (k["SIDE1"] * c_no2 + k["SIDE2"] * c_no3)
        t += dt

    conv = 1.0 - c_no3
    sel = c_nh3 / conv if conv > 0 else float("nan")
    print(f"ODE @ {t_end:.0f} s: NO3- conversion {conv:.5f}, "
          f"NH3 selectivity {sel:.4f}, residual NO2- {c_no2:.2e}")

    assert conv > 0.999, "expected near-complete NO3- conversion"
    assert 0.50 <= sel <= 0.62, (
        f"NH3 selectivity {sel:.3f} inconsistent with the Aspen-implemented "
        f"network (~0.55 analytic; Aspen realized yield 0.59)")
    assert c_no2 < 1e-3, "unexpected nitrite accumulation"
    print("ODE selectivity check PASSED (bkp network, ~55%)\n")
    return k


def check_idaes_reaction_block(k_expected):
    from pyomo.environ import ConcreteModel, value
    from idaes.core import FlowsheetBlock
    from idaes.models.properties.modular_properties.base.generic_property import (
        GenericParameterBlock,
    )
    from idaes.models.properties.modular_properties.base.generic_reaction import (
        GenericReactionParameterBlock,
    )

    from properties.enrtl_ammonia import get_prop
    from reactions.ammonia_reactions import get_rxn

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.liq = GenericParameterBlock(**get_prop(("Liq",)))
    m.fs.rxn = GenericReactionParameterBlock(**get_rxn(m.fs.liq))

    m.fs.sb = m.fs.liq.build_state_block([0], defined_state=True)
    m.fs.rb = m.fs.rxn.build_reaction_block(
        [0], state_block=m.fs.sb, has_equilibrium=False
    )

    sb = m.fs.sb[0]
    comp = {"H2O": 21.6, "Na+": 4.0, "Cl-": 3.56, "NO3-": 0.352,
            "SO4-2": 0.046, "NH3": 1e-8, "H2": 1.38, "N2": 1e-8,
            "NO2-": 1e-8, "OH-": 1e-8}
    total = sum(comp.values())
    sb.flow_mol.fix(total)
    for c, f in comp.items():
        sb.mole_frac_comp[c].fix(f / total)
    sb.temperature.fix(T_REACTOR)
    sb.pressure.fix(27e5)

    rb = m.fs.rb[0]
    print("IDAES rate constants at 333.15 K vs hand-computed:")
    ok = True
    for r in sorted(k_expected):
        k_idaes = value(rb.k_rxn[r])
        rel = abs(k_idaes - k_expected[r]) / k_expected[r]
        print(f"  {r}: idaes {k_idaes:.4e}  expected {k_expected[r]:.4e}  "
              f"rel.err {rel:.2e}")
        ok &= rel < 1e-6
    assert ok, "IDAES rate constants disagree with hand calculation"

    c_no3 = value(sb.conc_mol_phase_comp["Liq", "NO3-"])
    r1 = value(rb.reaction_rate["MAIN1"])
    expected = value(rb.k_rxn["MAIN1"]) * c_no3
    rel = abs(r1 - expected) / expected
    print(f"MAIN1 rate: {r1:.4e} mol/m3/s (k*C = {expected:.4e}, "
          f"rel.err {rel:.1e})")
    assert rel < 1e-6, "rate form is not first order in NO3- molarity"
    print("IDAES reaction block check PASSED")


if __name__ == "__main__":
    k = check_ode_selectivity()
    check_idaes_reaction_block(k)
    print("\nAll kinetics checks passed.")

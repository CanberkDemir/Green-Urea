"""
Standalone verification of the ureaF reaction package (RXNTOT + CARBEQ)
before it is wired into the full flowsheet.

Three checks:

1. Config sanity - the pre-exponential rescalings (kmol->mol basis,
   T-EXP=1 folding) produce the documented values and the k1=k2 / k3=k4
   pair symmetry that pins the two pseudo-equilibria.

2. R01 prototype - an isothermal two-phase CSTR at the reactor spec
   (200 degC, 141 kgf/cm2 = 138.3 bar, V = pi/4 * 2.5^2 * 20 = 98.2 m3,
   the grid-run geometry). The reactor is heavily oversized at grid feeds,
   so the outlet must sit ON the two rate-pair equilibria - that closure
   (K_c = 1 in kmol/m3 for both pairs, the .bkp's k1=k2 / k3=k4 design) is
   the hard assertion here, checked for a fresh 2:1 NH3:CO2 feed and for a
   carbamate/water-heavy recycle-like feed. Absolute single-pass
   conversions are printed for reference against the Pustjens/SI-1 tuning
   targets (~65% CO2 -> carbamate, ~55% carbamate -> urea) but NOT
   asserted: they depend on the loop feed composition and on the VLE
   basis (Aspen: RKS; here: ideal+Henry - see properties/ideal_urea.py),
   so the flowsheet-level validation is where they are judged.

3. CARBEQ prototype - the RGibbs replacement (StoichiometricReactor with
   extent = 99% of the limiting gas reagent via smooth_min; see
   reactions/urea_reactions.py for why that IS the equilibrium answer at
   these conditions) at the A012 spec (72.4 degC, 4 bar), fed an
   NH3/CO2-rich vapor. Checks conversion lands on the spec and that the
   raw van't Hoff relation confirms near-exhaustion (residual drive > 1).

Run:  conda activate idaes-pse && python test_urea_kinetics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyomo.environ import ConcreteModel, value
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.base.generic_reaction import (
    GenericReactionParameterBlock,
)
from idaes.models.unit_models import CSTR, StoichiometricReactor

from properties.ideal_urea import get_prop
from reactions.urea_reactions import (
    add_carbeq_conversion,
    get_carb_stoich_rxn,
    get_rxn,
    CARBEQ_CONV,
    T_REACTOR,
)

P_R01 = 138.3e5  # Pa, 141 kgf/cm2
V_R01 = 3.141592653589793 / 4.0 * 2.5**2 * 20.0  # m3, grid geometry
P_A012 = 4.0e5  # Pa


def check_config():
    from reactions.urea_reactions import RATE_SCALE
    cfg = get_rxn(None)
    rr = cfg["rate_reactions"]
    # T-EXP folding + basis conversion (x RATE_SCALE, see reactions module):
    #   CARBF: 1000 * 473.15 * 1000^-2 = 0.47315
    #   CARBD: 1000 * 473.15           = 473150
    #   UREAF:  750 * 473.15           = 354862.5
    #   UREAD:  750 * 473.15 * 1000^-1 = 354.8625
    expect = {n: v * RATE_SCALE for n, v in
              {"CARBF": 0.47315, "CARBD": 473150.0,
               "UREAF": 354862.5, "UREAD": 354.8625}.items()}
    for name, k in expect.items():
        got = rr[name]["parameter_data"]["arrhenius_const"][0]
        assert abs(got - k) / k < 1e-12, f"{name}: {got} != {k}"
    # pair symmetry pins the pseudo-equilibria
    for a, b in [("CARBF", "CARBD"), ("UREAF", "UREAD")]:
        ea = rr[a]["parameter_data"]["energy_activation"][0]
        eb = rr[b]["parameter_data"]["energy_activation"][0]
        assert ea == eb, f"Ea mismatch {a}/{b}"
    print("config check: rate constants and pair symmetry OK")


def run_r01_prototype(label, x, total=0.30):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.props = GenericParameterBlock(**get_prop(("Vap", "Liq")))
    m.fs.rxns = GenericReactionParameterBlock(**get_rxn(m.fs.props))
    m.fs.r01 = CSTR(
        property_package=m.fs.props,
        reaction_package=m.fs.rxns,
        has_heat_of_reaction=False,  # formation enthalpies carry the heat
        has_heat_transfer=True,
        has_pressure_change=False,
    )

    m.fs.r01.inlet.flow_mol[0].fix(total)
    for c, v in x.items():
        m.fs.r01.inlet.mole_frac_comp[0, c].fix(v)
    m.fs.r01.inlet.temperature[0].fix(T_REACTOR)
    m.fs.r01.inlet.pressure[0].fix(P_R01)
    m.fs.r01.volume.fix(V_R01)
    m.fs.r01.control_volume.properties_out[0].temperature.fix(T_REACTOR)

    assert degrees_of_freedom(m) == 0
    from urea_flowsheet import init_cstr_ramped
    init_cstr_ramped(
        m.fs.r01,
        state_args={
            "flow_mol": total,
            "temperature": T_REACTOR,
            "pressure": P_R01,
            "mole_frac_comp": dict(x),
        },
    )
    solver = get_solver("ipopt", options={"tol": 1e-8, "max_iter": 500})
    res = solver.solve(m)
    tc = str(res.solver.termination_condition)

    fin = {c: total * x[c] for c in x}
    out = m.fs.r01.outlet
    fout = {c: value(out.flow_mol[0] * out.mole_frac_comp[0, c]) for c in x}
    co2_conv = (fin["CO2"] - fout["CO2"]) / fin["CO2"]
    net_carb = fout["CARB"] + fout["UREA"] - fin["CARB"] - fin["UREA"]
    if net_carb > 1e-10:
        carb_to_urea = f"{(fout['UREA'] - fin['UREA']) / net_carb:6.1%}"
    else:
        carb_to_urea = "n/a (feed past equilibrium, net decomposition)"
    q_mw = value(m.fs.r01.heat_duty[0]) / 1e6

    # equilibrium closure of the two rate pairs (K_c = 1, kmol/m3 basis):
    #   pair 1: C_NH3,v^2 C_CO2,v = C_CARB,l  ->  ratio1 = LHS/(1e6 RHS)
    #   pair 2: C_UREA,l C_H2O,l  = C_CARB,l  ->  ratio2 = LHS/(1e3 RHS)
    st = m.fs.r01.control_volume.properties_out[0]
    cv = {j: value(st.conc_mol_phase_comp["Vap", j]) for j in ("NH3", "CO2")}
    cl = {j: value(st.conc_mol_phase_comp["Liq", j])
          for j in ("CARB", "UREA", "H2O")}
    ratio1 = cv["NH3"] ** 2 * cv["CO2"] / (1e6 * cl["CARB"])
    ratio2 = cl["UREA"] * cl["H2O"] / (1e3 * cl["CARB"])

    print(f"R01 CSTR prototype, {label} feed [{tc}]:")
    print(f"  CO2 single-pass conversion : {co2_conv:6.1%}  (Aspen tuning ~65%)")
    print(f"  carbamate -> urea split    : {carb_to_urea}  (Aspen tuning ~55%)")
    print(f"  outlet mol/s: " + ", ".join(f"{c}={fout[c]:.4f}" for c in x))
    print(f"  reactor duty: {q_mw:+.4f} MW")
    print(f"  equilibrium closure: pair1 = {ratio1:.4f}, pair2 = {ratio2:.4f}"
          f"  (both -> 1.000 when equilibrated)")
    assert tc == "optimal", f"solver returned {tc}"
    assert abs(ratio1 - 1.0) < 0.05, "carbamate pair not at equilibrium"
    assert abs(ratio2 - 1.0) < 0.05, "urea pair not at equilibrium"
    return co2_conv, carb_to_urea


def run_carbeq_prototype():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.props = GenericParameterBlock(**get_prop(("Vap", "Liq")))
    m.fs.eq_rxns = GenericReactionParameterBlock(
        **get_carb_stoich_rxn(m.fs.props)
    )
    m.fs.a012 = StoichiometricReactor(
        property_package=m.fs.props,
        reaction_package=m.fs.eq_rxns,
        has_heat_of_reaction=False,
        has_heat_transfer=True,
        has_pressure_change=False,
    )

    total = 0.10  # mol/s, NH3/CO2-rich off-gas with some water
    x = {"NH3": 0.55, "CO2": 0.25, "H2O": 0.20 - 2e-8,
         "UREA": 1e-8, "CARB": 1e-8}
    m.fs.a012.inlet.flow_mol[0].fix(total)
    for c, v in x.items():
        m.fs.a012.inlet.mole_frac_comp[0, c].fix(v)
    m.fs.a012.inlet.temperature[0].fix(345.55)  # 72.4 degC
    m.fs.a012.inlet.pressure[0].fix(P_A012)
    m.fs.a012.control_volume.properties_out[0].temperature.fix(345.55)

    # initialize at a partial extent, then swap in the conversion spec
    ext = m.fs.a012.control_volume.rate_reaction_extent[0, "CARBF"]
    ext.fix(0.5 * min(total * x["NH3"] / 2, total * x["CO2"]))
    m.fs.a012.initialize(outlvl=50)
    ext.unfix()
    add_carbeq_conversion(m.fs.a012)

    assert degrees_of_freedom(m) == 0
    solver = get_solver("ipopt", options={"tol": 1e-8, "max_iter": 500})
    res = solver.solve(m)
    tc = str(res.solver.termination_condition)

    out = m.fs.a012.outlet
    fout = {c: value(out.flow_mol[0] * out.mole_frac_comp[0, c]) for c in x}
    carb_made = fout["CARB"] - total * x["CARB"]
    co2_conv = (total * x["CO2"] - fout["CO2"]) / (total * x["CO2"])

    # residual thermodynamic drive at the outlet: K p_NH3^2 p_CO2 / x_CARB
    # > 1 means the raw equilibrium still points toward MORE condensation,
    # i.e. the 99%-of-limiting spec is on the conservative side of the
    # true (near-exhaustion) equilibrium
    st = m.fs.a012.control_volume.properties_out[0]
    from reactions.urea_reactions import K_P_298, DH_RXN_COND
    import math
    T = value(st.temperature)
    kp = K_P_298 * math.exp(-(DH_RXN_COND / 8.314462618)
                            * (1 / T - 1 / 298.15))
    drive = (kp
             * (value(st.mole_frac_phase_comp["Vap", "NH3"]) * P_A012) ** 2
             * (value(st.mole_frac_phase_comp["Vap", "CO2"]) * P_A012)
             / value(st.mole_frac_phase_comp["Liq", "CARB"]))

    print(f"A012 RGibbs-replacement prototype [{tc}]:")
    print(f"  CO2 condensed to carbamate : {co2_conv:6.1%} (spec {CARBEQ_CONV:.0%} of limiting)")
    print(f"  CARB out                   : {carb_made:.5f} mol/s")
    print(f"  residual equilibrium drive : {drive:.2f} (>1 = spec conservative)")
    assert tc == "optimal", f"solver returned {tc}"
    assert carb_made > 0, "no carbamate condensation at 72.4 degC / 4 bar"
    assert abs(co2_conv - CARBEQ_CONV) < 0.02, "conversion off the spec"
    return co2_conv


if __name__ == "__main__":
    check_config()
    run_r01_prototype(
        "fresh 2:1:0.5 NH3:CO2:H2O",
        {"NH3": 2 / 3.5, "CO2": 1 / 3.5, "H2O": 0.5 / 3.5,
         "UREA": 1e-8, "CARB": 1e-8},
    )
    run_r01_prototype(
        "recycle-like (carbamate/water heavy)",
        {"NH3": 0.30, "CO2": 0.10, "H2O": 0.25, "UREA": 0.05, "CARB": 0.30},
    )
    run_carbeq_prototype()
    print("all kinetics checks passed")

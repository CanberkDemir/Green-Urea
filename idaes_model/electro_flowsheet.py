"""
IDAES flowsheet of the direct electrochemical (NO3RR) ammonia section.

    brine feed (8500 kg/h, 25 C)
      -> pre   Heater to the cell temperature (65 C)
      -> cell  StoichiometricReactor = cathode compartments of the 4-stage
               NO3RR cascade (electro_no3rr.py sets extents, current, V)
      -> gls   Flash: HER hydrogen (+ a little NH3/H2O) off the catholyte
      -> strip 10-stage vacuum stripper, 0.08 bar (nh3_stripper.build_column,
               rigorous MESH, recovery spec on the bottoms)
      -> rectifier to anhydrous NH3 (shortcut, ammonia_energy) -> NH3 tank

Catholyte chemistry (acid anolyte, CEM, H+ crosses - see electro_no3rr.py):
    R_N  : NO3- (+ 8 H+ + 8 e- from the anode side) -> NH3 + OH- + 2 H2O
    R_HER: (2 H+ + 2 e-)                             -> H2
The H+ (anolyte water) is not a component of the brine property package;
its mass is book-kept as anolyte make-up water. The reactor runs with
outlet T fixed; the electrochemical heat (I*(V - V_tn)) is computed from
the cell model, not from the property package's enthalpy of formation.

    python electro_flowsheet.py                 # central parameters
    python electro_flowsheet.py --case optimistic
"""

import argparse
import contextlib
import io
import logging
import sys
from pathlib import Path

from pyomo.environ import (Block, ConcreteModel, Constraint, TransformationFactory,
                           units as pyunits, value)
from pyomo.network import Arc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idaes.core import FlowsheetBlock  # noqa: E402
from idaes.core.solvers import get_solver  # noqa: E402
from idaes.core.util.initialization import propagate_state  # noqa: E402
from idaes.models.properties.modular_properties.base.generic_property import (  # noqa: E402
    GenericParameterBlock)
from idaes.models.properties.modular_properties.base.generic_reaction import (  # noqa: E402
    GenericReactionParameterBlock)
from idaes.models.unit_models import (Feed, Flash, Heater,  # noqa: E402
                                      StoichiometricReactor)

import electro_no3rr as EN  # noqa: E402
import solver_path  # noqa: E402  (Ipopt 3.14 needed here too)
import nh3_stripper as NS  # noqa: E402
from properties.enrtl_ammonia import get_prop  # noqa: E402

MW = {"H2O": 18.0153e-3, "NH3": 17.0305e-3, "H2": 2.01588e-3,
      "N2": 28.0134e-3, "Na+": 22.98977e-3, "Cl-": 35.453e-3,
      "NO3-": 62.0049e-3, "NO2-": 46.0055e-3, "SO4-2": 96.0626e-3,
      "OH-": 17.0073e-3}
COMPS = list(MW)
EPS = 1e-10
T_CELL = 65.0 + 273.15
P_CELL = 1.3e5
V_TN_NO3RR, V_TN_HER = 0.607, 1.481
P_STRIP_BAR, N_STAGES, RECOVERY = 0.08, 10, 0.99
_OPT = {"tol": 1e-6, "max_iter": 3000, "mu_strategy": "adaptive"}


def rxn_config(props):
    return {
        "property_package": props,
        "base_units": {"time": pyunits.s, "length": pyunits.m,
                       "mass": pyunits.kg, "amount": pyunits.mol,
                       "temperature": pyunits.K},
        "rate_reactions": {
            "NO3RR": {"stoichiometry": {("Liq", "NO3-"): -1,
                                        ("Liq", "NH3"): 1,
                                        ("Liq", "OH-"): 1,
                                        ("Liq", "H2O"): 2}},
            "HER": {"stoichiometry": {("Vap", "H2"): 1}},
        },
    }


def build(case="central", stages=4, conversion=0.95):
    cell_res, cell = EN.design(EN.CASES[case], "raw", stages, conversion)
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    fs = m.fs
    fs.props_vle = GenericParameterBlock(**get_prop(("Liq", "Vap")))
    fs.rxns = GenericReactionParameterBlock(**rxn_config(fs.props_vle))

    fs.feed = Feed(property_package=fs.props_vle)
    fs.pre = Heater(property_package=fs.props_vle, has_pressure_change=False)
    fs.cell = StoichiometricReactor(property_package=fs.props_vle,
                                    reaction_package=fs.rxns,
                                    has_heat_of_reaction=False,
                                    has_heat_transfer=True,
                                    has_pressure_change=False)
    fs.gls = Flash(property_package=fs.props_vle, has_pressure_change=False)
    fs.a1 = Arc(source=fs.feed.outlet, destination=fs.pre.inlet)
    fs.a2 = Arc(source=fs.pre.outlet, destination=fs.cell.inlet)
    fs.a3 = Arc(source=fs.cell.outlet, destination=fs.gls.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    flows = {c: EN.BRINE_KGPH.get(c, 0.0) / 3600 / MW[c] for c in COMPS}
    tot = sum(flows.values())
    o = fs.feed.outlet
    o.flow_mol[0].fix(tot)
    for c in COMPS:
        o.mole_frac_comp[0, c].fix(max(flows[c], EPS * tot) / tot)
    o.temperature[0].fix(298.15)
    o.pressure[0].fix(P_CELL)
    fs.pre.outlet.temperature[0].fix(T_CELL)
    fs.cell.outlet.temperature[0].fix(T_CELL)
    n_n = flows["NO3-"] * conversion
    fs.cell.rate_reaction_extent[0, "NO3RR"].fix(n_n)
    fs.cell.rate_reaction_extent[0, "HER"].fix(cell["h2_kgph"] / 3600 / MW["H2"])
    fs.gls.heat_duty.fix(0.0)
    m.cell_design, m.cell_stages = cell, cell_res
    return m


def solve(m):
    with solver_path.ipopt314():
        return _solve(m)


def _solve(m):
    fs = m.fs
    s = get_solver("ipopt", options=_OPT)
    with contextlib.redirect_stdout(io.StringIO()):
        fs.feed.initialize(outlvl=50)
        propagate_state(fs.a1)
        fs.pre.initialize(outlvl=50, optarg=_OPT)
        propagate_state(fs.a2)
        fs.cell.initialize(outlvl=50, optarg=_OPT)
        propagate_state(fs.a3)
        fs.gls.initialize(outlvl=50, optarg=_OPT)
    r = s.solve(m)
    assert str(r.solver.termination_condition) == "optimal", r
    # rigorous vacuum stripper on the degassed catholyte
    liq = fs.gls.control_volume.properties_out[0]
    from pyomo.environ import Reference  # noqa: F401
    feed_state = _LiqState(fs.gls.liq_outlet)
    p_pa = P_STRIP_BAR * 1e5
    t_b = NS.water_tsat(P_STRIP_BAR)
    col = NS.build_column(m, fs.props_vle, feed_state, p_pa, N_STAGES,
                          t_b - 2.0)
    nh3_in = value(fs.gls.liq_outlet.flow_mol[0]
                   * fs.gls.liq_outlet.mole_frac_comp[0, "NH3"])
    q_guess = 160e3
    with contextlib.redirect_stdout(io.StringIO()):
        NS.initialize_column(col, N_STAGES, q_guess, s)
    for blk in col.component_data_objects(Block, descend_into=True):
        if hasattr(blk, "eps_1_Vap_Liq"):
            blk.eps_1_Vap_Liq = 1.0
    reb = getattr(col, f"tray{N_STAGES}")
    s.solve(col)
    bot = reb.liq_outlet
    col.recovery_spec = Constraint(
        expr=bot.flow_mol[0] * bot.mole_frac_comp[0, "NH3"]
        == (1 - RECOVERY) * nh3_in)
    reb.heat_duty.unfix()
    for _ in range(3):
        r = s.solve(col)
        if str(r.solver.termination_condition) == "optimal":
            break
    assert str(r.solver.termination_condition) == "optimal", r
    del liq
    return m


# rectifier pressure -> (sump T = water Tsat [C], NH3 condensing T [C],
#                        NH3 latent heat at that T [kJ/kg])
RECT_LEVELS = {15.5: (198.0, 40.0, 1100.0), 6.2: (160.0, 10.0, 1226.0)}


def rectifier_streams(nh3_kgph, w_kmolph, p_bar=15.5):
    """Rectifier to 99.5 mol% NH3 (Underwood Rmin, R = 1.5 Rmin).

    15.5 bar: NH3 condenses on cooling water (40 C) but the sump boils at
    198 C (above heat-pump reach). 6.2 bar: sump 160 C (heat pump), NH3
    condenses at 10 C (chiller). TODO G2 compares the two."""
    import ammonia_energy as AE
    from heat_integration import Stream
    n_nh3 = nh3_kgph / 17.031
    x_f = n_nh3 / (n_nh3 + w_kmolph)
    a = AE.ALPHA_RECT
    r_min = (1 / (a - 1)) * (AE.X_D / x_f - a * (1 - AE.X_D) / (1 - x_f))
    r = AE.R_FACTOR * max(r_min, 0.1)
    t_bot, t_cond, dh = RECT_LEVELS[p_bar]
    q_cond = (r + 1) * nh3_kgph * dh / 3600.0
    w_kg = w_kmolph * 18.015
    q_reb = q_cond + w_kg * AE.CP_WATER * (t_bot - AE.T_RECT_FEED) / 3600
    st = [
        Stream("rectifier feed preheat", 40.0, AE.T_RECT_FEED,
               (w_kg + nh3_kgph) * AE.CP_WATER * (AE.T_RECT_FEED - 40.0)
               / 3600, section="recovery"),
        Stream(f"rectifier reboiler ({p_bar:g} bar)", t_bot, t_bot + 0.1,
               q_reb, section="recovery"),
        Stream(f"rectifier condenser (NH3, {t_cond:g} C)", t_cond + 0.1,
               t_cond, q_cond, section="recovery"),
        Stream("rectifier bottoms cooler", t_bot, AE.T_CW + 5.0,
               w_kg * AE.CP_WATER * (t_bot - AE.T_CW - 5.0) / 3600,
               section="recovery"),
    ]
    return st, {"rectifier R": r, "rectifier feed NH3 [mol%]": 100 * x_f}


class _LiqState:
    """Adapter: a port looks like a state block to nh3_stripper.build_column."""
    def __init__(self, port):
        self.flow_mol = port.flow_mol[0]
        self.temperature = port.temperature[0]
        self.pressure = port.pressure[0]
        self.mole_frac_comp = {j: port.mole_frac_comp[0, j]
                               for j in COMPS}


P_MVR_BAR = 0.25     # overhead recompressed to here: condenses at ~60 C
ETA_MVR = 0.75


def mvr_work_kw(n_mol_s, p_in_bar, p_out_bar, t_in_k, k=1.33):
    r = p_out_bar / p_in_bar
    return n_mol_s * k / (k - 1) * 8.314 * t_in_k * (r ** ((k - 1) / k) - 1)         / ETA_MVR / 1e3


def results(m, mvr=False, rect_p=15.5):
    """Energy streams and numbers for pinch / plant_energy.

    mvr=True: the stripper overhead (0.08 bar, ~40 C) is recompressed to
    0.25 bar so it condenses at ~60 C instead of being chilled to 10 C; its
    heat then serves the 42 C reboiler through the pinch match."""
    import ammonia_energy as AE
    from heat_integration import Stream
    fs, cell = m.fs, m.cell_design
    col = fs.strip
    reb = getattr(col, f"tray{N_STAGES}")
    top = col.tray1.vap_outlet
    i_n = 8 * EN.F * value(fs.cell.rate_reaction_extent[0, "NO3RR"])
    i_h = cell["current"] - i_n
    v_avg = cell["power_kw"] * 1e3 / cell["current"]
    q_cell = (i_n * (v_avg - V_TN_NO3RR) + i_h * (v_avg - V_TN_HER)) / 1e3
    nh3_top = value(top.flow_mol[0] * top.mole_frac_comp[0, "NH3"])
    h2o_top = value(top.flow_mol[0] * top.mole_frac_comp[0, "H2O"])
    nh3_kgph = nh3_top * MW["NH3"] * 3600
    gas = fs.gls.vap_outlet
    nh3_lost_gas = value(gas.flow_mol[0] * gas.mole_frac_comp[0, "NH3"]) \
        * MW["NH3"] * 3600
    t_b = NS.water_tsat(P_STRIP_BAR) - 273.15
    q_pre = value(fs.pre.heat_duty[0]) / 1e3
    q_spre = value(col.pre.heat_duty[0]) / 1e3          # < 0: cooling
    q_reb = value(reb.heat_duty[0]) / 1e3
    t_reb = value(reb.control_volume.properties_out[0].temperature) - 273.15
    bot = reb.liq_outlet
    m_bot = sum(value(bot.flow_mol[0] * bot.mole_frac_comp[0, j]) * MW[j]
                for j in COMPS) * 3600
    q_ovh = h2o_top * 42.0 + nh3_top * 30.0              # kW, rough latent
    streams = [
        Stream("cell feed preheat", 25.0, 65.0, q_pre, section="ammonia"),
        Stream("cell heat (65 C loop)", 65.0, 64.9, q_cell, section="ammonia"),
        Stream("stripper feed cooler", 65.0, t_b - 2.0, -q_spre,
               section="recovery"),
        Stream("stripper reboiler", t_reb, t_reb + 0.1, q_reb,
               section="recovery"),
        (Stream("stripper overhead condenser (after MVR, 0.25 bar)",
                NS.water_tsat(P_MVR_BAR) - 273.15 - 2.0,
                NS.water_tsat(P_MVR_BAR) - 273.15 - 12.0, q_ovh,
                section="recovery") if mvr else
         Stream("stripper overhead condenser", t_b - 2.0, 10.0, q_ovh,
                section="recovery")),
        Stream("stripper bottoms cooler", t_reb, 30.0,
               m_bot * AE.CP_BRINE * (t_reb - 30.0) / 3600,
               section="recovery"),
    ]
    # rectifier to anhydrous NH3: shortcut (ammonia_energy S2 tail) fed with
    # the RIGOROUS stripper overhead (NH3 + water flows)
    rs, r_info = rectifier_streams(nh3_kgph, h2o_top * 3.6, rect_p)
    streams += rs
    r_shaft = mvr_work_kw(value(top.flow_mol[0]), P_STRIP_BAR, P_MVR_BAR,
                          t_b + 273.15) if mvr else 0.0
    out = {
        "NH3 product [kg/h]": nh3_kgph * 1.0,
        "NH3 lost in HER gas [kg/h]": nh3_lost_gas,
        "cell power [kW]": cell["power_kw"],
        "cell area [m2]": cell["area"], "cell FE": cell["fe"],
        "cell heat [kW]": q_cell,
        "HER H2 [kg/h]": cell["h2_kgph"],
        "anolyte make-up water [kg/h]": (4 * i_n + i_h) / (4 * EN.F)
        * 2 * MW["H2O"] * 3600,
        "stripper reboiler [kW]": q_reb, "stripper T bottom [C]": t_reb,
        "stripper overhead NH3 [mol%]": 100 * nh3_top / (nh3_top + h2o_top),
        "feed preheat [kW]": q_pre,
        "rectifier reflux ratio": r_info["rectifier R"],
        "overhead MVR [kW]": r_shaft,
    }
    return streams, cell["power_kw"] + r_shaft, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="central", choices=list(EN.CASES))
    ap.add_argument("--mvr", action="store_true",
                    help="recompress the stripper overhead (see results())")
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    m = solve(build(a.case))
    streams, shaft, out = results(m, a.mvr)
    print(f"=== electrochemical NH3 section ({a.case}) ===")
    for k, v in out.items():
        print(f"  {k:36s} {v:10.3f}")
    print("streams:")
    for s_ in streams:
        print(f"  {s_.name:34s} {s_.kind:4s} {s_.t_in:6.1f}->{s_.t_out:6.1f} C "
              f"{s_.duty_kw:8.1f} kW")
    from heat_integration import pinch
    res = pinch(streams, 10.0)
    print("  " + res.summary())


if __name__ == "__main__":
    main()

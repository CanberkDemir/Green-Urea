"""
Low-pressure DEF urea flowsheet (no HP stripper), built from scratch after
US 10,759,745 B2 (Stamicarbon; Pustjens & Van Den Tillaart), "Urea
production process and plant" for a purified urea solution such as DEF.

Design intent (ours, following the patent concept):
  * produce DEF directly: 32.5 wt% urea in water, no evaporation/prilling,
  * keep the high-pressure section minimal: reactor only (no HP stripper,
    no HP carbamate condenser), at the patent's lower pressure bound,
  * do the separation at medium/low pressure and under vacuum,
  * the water the reaction makes plus the stripping steam leaves with the
    product, so water cannot accumulate in the recycle loop (the problem
    found in the ureaF replica, where the B25 overhead returns to HP).

Flowsheet (patent claim 1 steps A-E):

  CO2 (at P_SYN; compression work computed separately)
  NH3 make-up -> HP pump ----------------+
  carbamate recycle -> HP pump ----------+-> M_R -> REACTOR (150 bar, 190 C)
      -> V_MP (20 bar) -> F_MP (160 C) -[V]-> MP condenser
                                       -[L]-> V_LP (4 bar) -> F_LP (135 C)
      F_LP -[V]-> LP condenser;  -[L]-> V_PU (0.5 bar) -> + LP steam ->
      STRIPPER (adiabatic flash, 0.5 bar) -[L]-> cooler 30 C -> + demin
      water -> DEF (32.5 wt%)
      STRIPPER -[V]-> vacuum condenser (40 C) -> condensate pump (4 bar)
         -> SPLIT: fraction to LP condenser (patent: all), rest -> DEF
            dilution (our "skip the HP loop" option; watch DEF alkalinity)
      LP condenser: LP vap + condensate -> carbamate condensation (4 bar,
         72 C) -> flash -[L]-> pump 20 bar; -[V]-> compressor 20 bar
      MP condenser: MP vap + LP liquid + LP gas -> carbamate condensation
         (20 bar, 90 C) -> cooler 35 C (NH3 condensation) -> flash
         -[L]-> carbamate recycle;  -[V]-> vent

Modelling basis and assumptions (all explicit, all adjustable):
  * thermo: properties/ideal_urea.py (ideal gas + Raoult/Henry; UREA and
    CARB liquid-only) - the same basis as the ureaF replica,
  * reactor: StoichiometricReactor. Inlet carbamate is decomposed, then
    CO2 -> urea at a conversion from a simple trend correlation
        X = 0.64 + 0.05 (N/C - 4) - 0.09 (H/C - 0.5)
    (typical non-stripping total-recycle reactors at ~190 C; the slopes
    reproduce the usual literature trends - more NH3 helps, recycled
    water hurts). The patent gives no conversion; replace with the
    kinetic/equilibrium R01 model once validated.
  * N/C (molar, NH3/CO2 at reactor inlet, carbamate counted as 2+1) is
    fixed at 4.0 (patent preferred 3.5-5.0); the NH3 make-up is solved.
  * carbamate condensers: the RGibbs-replacement conversion form from
    reactions/urea_reactions.py (99% of the limiting reagent).
  * CO2 enters at reactor pressure; its 4-stage intercooled compression
    work is computed analytically (ideal gas, eta_s = 0.75) for the
    energy table.

Run:
    python def_lowp_flowsheet.py                 # patent routing
    python def_lowp_flowsheet.py --direct 0.5    # 50% condensate -> DEF
"""

import argparse
import math
import sys
from pathlib import Path

from pyomo.environ import (
    ConcreteModel, Constraint, Expression, TransformationFactory, Var,
    value, units as pyunits,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.base.generic_reaction import (
    GenericReactionParameterBlock,
)
from idaes.models.unit_models import (
    Feed, Flash, Heater, Mixer, Product, PressureChanger, Separator,
    StoichiometricReactor,
)
from idaes.models.unit_models.mixer import MomentumMixingType
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models.unit_models.separator import SplittingType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from properties.ideal_urea import get_prop  # noqa: E402
from reactions.urea_reactions import (  # noqa: E402
    add_carbeq_conversion, get_carb_stoich_rxn,
)

# ---------------------------------------------------------------- specs
BAR = 1e5
P_SYN = 150 * BAR      # patent 150-250 bar (claim 14: 120-300); low end
T_SYN = 463.15         # 190 C ("higher than 190 C")
N_C = 4.0              # reactor inlet NH3/CO2 (patent pref. 3.5-5.0)
P_MP, T_MP = 20 * BAR, 433.15    # 15-25 bar preferred; 160 C decomposer
P_LP, T_LP = 4 * BAR, 408.15     # 3-5 bar (claim 18); 135 C
P_PU = 0.5 * BAR                 # purification 0.01-1.5 bar (claim 12)
P_STEAM, T_STEAM = 3 * BAR, 413.15   # LP stripping steam (2-6 bar)
T_VAC_COND = 313.15              # vacuum condenser 40 C
T_LPC, T_MPC = 345.15, 363.15    # carbamate condensers 72 C / 90 C
T_NH3C = 308.15                  # NH3 condensation 35 C
T_DEF = 303.15                   # product ~30 C
W_DEF = 0.325                    # DEF urea mass fraction (ISO 22241)
ETA_PUMP, ETA_COMP = 0.75, 0.75

COMPS = ["NH3", "CO2", "H2O", "UREA", "CARB"]
MW = {"NH3": 17.0305e-3, "CO2": 44.0095e-3, "H2O": 18.0153e-3,
      "UREA": 60.0553e-3, "CARB": 78.0707e-3}
EPS = 1e-8
KGPH = 1.0 / 3600.0
_OPTARG = {"tol": 1e-6, "max_iter": 3000, "mu_strategy": "adaptive"}


def conversion(nc, hc):
    """CO2 -> urea single-pass conversion trend (see module docstring)."""
    return 0.64 + 0.05 * (nc - 4.0) - 0.09 * (hc - 0.5)


# ------------------------------------------------------------- helpers
def _set_port(port, flows, T, P):
    tot = sum(flows.get(c, 0.0) for c in COMPS)
    tot = max(tot, 1e-6)
    port.flow_mol[0].set_value(tot)
    for c in COMPS:
        port.mole_frac_comp[0, c].set_value(max(flows.get(c, 0.0) / tot, EPS))
    port.temperature[0].set_value(T)
    port.pressure[0].set_value(P)


def _fix_port(port, flows, T, P):
    _set_port(port, flows, T, P)
    for v in port.vars.values():
        v.fix()


def _prop(arc):
    propagate_state(arc)
    port = arc.destination
    for idx in port.mole_frac_comp:
        v = port.mole_frac_comp[idx]
        if v.value is not None and v.value < EPS:
            v.set_value(EPS)


def _valve(fs, name, p_out):
    u = Heater(property_package=fs.props, has_pressure_change=True)
    setattr(fs, name, u)
    u.heat_duty.fix(0.0)
    u.control_volume.properties_out[0].pressure.fix(p_out)
    return u


def _flash(fs, name, T=None, duty=None, dP=0.0):
    u = Flash(property_package=fs.props, has_pressure_change=True)
    setattr(fs, name, u)
    u.deltaP.fix(dP)
    if T is not None:
        u.control_volume.properties_out[0].temperature.fix(T)
    if duty is not None:
        u.heat_duty.fix(duty)
    return u


def _heater(fs, name, T, p_out=None):
    u = Heater(property_package=fs.props, has_pressure_change=p_out is not None)
    setattr(fs, name, u)
    u.control_volume.properties_out[0].temperature.fix(T)
    if p_out is not None:
        u.control_volume.properties_out[0].pressure.fix(p_out)
    return u


def _pump(fs, name, p_out):
    u = PressureChanger(property_package=fs.props,
                        thermodynamic_assumption=ThermodynamicAssumption.pump)
    setattr(fs, name, u)
    u.efficiency_pump.fix(ETA_PUMP)
    u.control_volume.properties_out[0].pressure.fix(p_out)
    return u


def _mixer(fs, name, inlets, p):
    u = Mixer(property_package=fs.props, inlet_list=inlets,
              momentum_mixing_type=MomentumMixingType.none)
    setattr(fs, name, u)
    u.mixed_state[0].pressure.fix(p)
    return u


def _carb_condenser(fs, name, T, p):
    u = StoichiometricReactor(property_package=fs.props,
                              reaction_package=fs.carb_rxns,
                              has_heat_of_reaction=False,
                              has_heat_transfer=True,
                              has_pressure_change=True)
    setattr(fs, name, u)
    u.control_volume.properties_out[0].temperature.fix(T)
    u.control_volume.properties_out[0].pressure.fix(p)
    add_carbeq_conversion(u)
    return u


def _urea_rxn_config(props):
    """Reactor stoichiometry: carbamate decomposition + net urea formation
    (2NH3 + CO2 -> UREA + H2O). Extents are closed by constraints."""
    return {
        "property_package": props,
        "base_units": {"time": pyunits.s, "length": pyunits.m,
                       "mass": pyunits.kg, "amount": pyunits.mol,
                       "temperature": pyunits.K},
        "rate_reactions": {
            "CARBD": {"stoichiometry": {("Liq", "CARB"): -1,
                                        ("Liq", "NH3"): 2,
                                        ("Liq", "CO2"): 1}},
            "UREA": {"stoichiometry": {("Liq", "NH3"): -2,
                                       ("Liq", "CO2"): -1,
                                       ("Liq", "UREA"): 1,
                                       ("Liq", "H2O"): 1}},
        },
    }


# --------------------------------------------------------------- build
def build_model(fco2_kgph=15.0, steam_kgph=20.0, direct_frac=0.0):
    m = ConcreteModel()
    fs = m.fs = FlowsheetBlock(dynamic=False)
    fs.props = GenericParameterBlock(**get_prop(("Vap", "Liq")))
    fs.carb_rxns = GenericReactionParameterBlock(**get_carb_stoich_rxn(fs.props))
    fs.urea_rxns = GenericReactionParameterBlock(**_urea_rxn_config(fs.props))

    # feeds / products
    for nm in ("co2_feed", "nh3_feed", "steam_feed", "water_feed"):
        setattr(fs, nm, Feed(property_package=fs.props))
    for nm in ("def_prod", "vent_mp", "vent_vac"):
        setattr(fs, nm, Product(property_package=fs.props))

    # HP synthesis
    _pump(fs, "nh3_pump", P_SYN)
    _pump(fs, "rec_pump", P_SYN)
    _mixer(fs, "m_r", ["co2", "nh3", "rec"], P_SYN)
    fs.reactor = StoichiometricReactor(
        property_package=fs.props, reaction_package=fs.urea_rxns,
        has_heat_of_reaction=False, has_heat_transfer=True,
        has_pressure_change=True)
    cvr = fs.reactor.control_volume
    cvr.properties_out[0].temperature.fix(T_SYN)
    cvr.properties_out[0].pressure.fix(P_SYN)
    st = cvr.properties_in[0]
    f = {c: st.flow_mol * st.mole_frac_comp[c] for c in COMPS}
    fs.n_nh3 = Expression(expr=f["NH3"] + 2 * f["CARB"])
    fs.n_co2 = Expression(expr=f["CO2"] + f["CARB"])
    fs.nc_ratio = Expression(expr=fs.n_nh3 / fs.n_co2)
    fs.hc_ratio = Expression(expr=f["H2O"] / fs.n_co2)
    fs.reactor.carbd = Constraint(
        expr=cvr.rate_reaction_extent[0, "CARBD"] == f["CARB"])
    fs.x_co2 = Var(initialize=0.6, bounds=(0.05, 0.95))
    fs.x_def = Constraint(expr=fs.x_co2 == conversion(fs.nc_ratio,
                                                      fs.hc_ratio))
    fs.reactor.urea_ext = Constraint(
        expr=cvr.rate_reaction_extent[0, "UREA"] == fs.x_co2 * fs.n_co2)
    fs.nc_spec = Constraint(expr=fs.n_nh3 == N_C * fs.n_co2)

    # MP / LP recovery
    _valve(fs, "v_mp", P_MP)
    _flash(fs, "f_mp", T=T_MP)
    _valve(fs, "v_lp", P_LP)
    _flash(fs, "f_lp", T=T_LP)

    # purification: vacuum steam stripper + DEF finishing
    _valve(fs, "v_pu", P_PU)
    _valve(fs, "v_steam", P_PU)
    _mixer(fs, "m_strip", ["sol", "steam"], P_PU)
    _flash(fs, "stripper", duty=0.0)
    _heater(fs, "def_cool", T_DEF)
    _mixer(fs, "m_def", ["sol", "water", "cond"], 1.0 * BAR)
    _heater(fs, "def_trim", T_DEF)

    # vacuum condenser + condensate routing
    _flash(fs, "vac_cond", T=T_VAC_COND)
    _pump(fs, "cond_pump", P_LP)
    fs.cond_split = Separator(property_package=fs.props,
                              outlet_list=["to_lp", "to_def"],
                              split_basis=SplittingType.totalFlow)
    # a zero-flow branch cannot be initialized; 1e-4 is a negligible trickle
    fs.cond_split.split_fraction[0, "to_def"].fix(
        min(max(direct_frac, 1e-4), 0.99))
    _valve(fs, "v_cond_def", 1.0 * BAR)

    # LP carbamate condenser
    _mixer(fs, "m_lpc", ["vap", "cond"], P_LP)
    _carb_condenser(fs, "lpc", T_LPC, P_LP)
    _flash(fs, "f_lpc", T=T_LPC)
    _pump(fs, "lp_pump", P_MP)
    # LP off-gas compressor (stands in for the patent's scrubber/ejector
    # route). The property package has no liquid entropy, so the unit is
    # isothermal (aftercooled) and its shaft work is computed
    # analytically in the report.
    fs.lp_comp = PressureChanger(
        property_package=fs.props, compressor=True,
        thermodynamic_assumption=ThermodynamicAssumption.isothermal)
    fs.lp_comp.control_volume.properties_out[0].pressure.fix(P_MP)

    # MP carbamate condenser + NH3 condenser
    _mixer(fs, "m_mpc", ["vap", "liq", "gas"], P_MP)
    _carb_condenser(fs, "mpc", T_MPC, P_MP)
    _heater(fs, "nh3_cond", T_NH3C)
    _flash(fs, "f_mpc", T=T_NH3C)

    # ------------------------------------------------------ arcs
    A = []

    def arc(name, s, d):
        setattr(fs, name, Arc(source=s, destination=d))
        A.append(getattr(fs, name))

    arc("s_co2", fs.co2_feed.outlet, fs.m_r.co2)
    arc("s_nh3a", fs.nh3_feed.outlet, fs.nh3_pump.inlet)
    arc("s_nh3b", fs.nh3_pump.outlet, fs.m_r.nh3)
    arc("s_rec", fs.rec_pump.outlet, fs.m_r.rec)
    arc("s_rin", fs.m_r.outlet, fs.reactor.inlet)
    arc("s_rout", fs.reactor.outlet, fs.v_mp.inlet)
    arc("s_mpin", fs.v_mp.outlet, fs.f_mp.inlet)
    arc("s_mpl", fs.f_mp.liq_outlet, fs.v_lp.inlet)
    arc("s_lpin", fs.v_lp.outlet, fs.f_lp.inlet)
    arc("s_lpl", fs.f_lp.liq_outlet, fs.v_pu.inlet)
    arc("s_pu", fs.v_pu.outlet, fs.m_strip.sol)
    arc("s_st1", fs.steam_feed.outlet, fs.v_steam.inlet)
    arc("s_st2", fs.v_steam.outlet, fs.m_strip.steam)
    arc("s_strin", fs.m_strip.outlet, fs.stripper.inlet)
    arc("s_strl", fs.stripper.liq_outlet, fs.def_cool.inlet)
    arc("s_defa", fs.def_cool.outlet, fs.m_def.sol)
    arc("s_wat", fs.water_feed.outlet, fs.m_def.water)
    arc("s_defb", fs.m_def.outlet, fs.def_trim.inlet)
    arc("s_defc", fs.def_trim.outlet, fs.def_prod.inlet)
    arc("s_strv", fs.stripper.vap_outlet, fs.vac_cond.inlet)
    arc("s_vacv", fs.vac_cond.vap_outlet, fs.vent_vac.inlet)
    arc("s_vacl", fs.vac_cond.liq_outlet, fs.cond_pump.inlet)
    arc("s_cnd", fs.cond_pump.outlet, fs.cond_split.inlet)
    arc("s_cdef1", fs.cond_split.to_def, fs.v_cond_def.inlet)
    arc("s_cdef2", fs.v_cond_def.outlet, fs.m_def.cond)
    arc("s_clp", fs.cond_split.to_lp, fs.m_lpc.cond)
    arc("s_lpv", fs.f_lp.vap_outlet, fs.m_lpc.vap)
    arc("s_lpc1", fs.m_lpc.outlet, fs.lpc.inlet)
    arc("s_lpc2", fs.lpc.outlet, fs.f_lpc.inlet)
    arc("s_lpcl", fs.f_lpc.liq_outlet, fs.lp_pump.inlet)
    arc("s_lpcv", fs.f_lpc.vap_outlet, fs.lp_comp.inlet)
    arc("s_mpv", fs.f_mp.vap_outlet, fs.m_mpc.vap)
    arc("s_lpp", fs.lp_pump.outlet, fs.m_mpc.liq)
    arc("s_lpg", fs.lp_comp.outlet, fs.m_mpc.gas)
    arc("s_mpc1", fs.m_mpc.outlet, fs.mpc.inlet)
    arc("s_mpc2", fs.mpc.outlet, fs.nh3_cond.inlet)
    arc("s_mpc3", fs.nh3_cond.outlet, fs.f_mpc.inlet)
    arc("s_vent", fs.f_mpc.vap_outlet, fs.vent_mp.inlet)
    arc("s_recl", fs.f_mpc.liq_outlet, fs.rec_pump.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)
    fs.arc_list = A

    # widen SmoothVLE smoothing (saturated liquids feed the next unit)
    from pyomo.environ import Block
    for blk in m.component_data_objects(Block, descend_into=True):
        if hasattr(blk, "eps_1_Vap_Liq"):
            blk.eps_1_Vap_Liq = 1.0

    # ------------------------------------------------------ feeds
    n_co2 = fco2_kgph * KGPH / MW["CO2"]
    _fix_port(fs.co2_feed.outlet, {"CO2": n_co2}, 313.15, P_SYN)
    _fix_port(fs.nh3_feed.outlet, {"NH3": 2 * n_co2}, 298.15, 20 * BAR)
    fs.nh3_feed.outlet.flow_mol[0].unfix()          # solved by N/C spec
    _fix_port(fs.steam_feed.outlet,
              {"H2O": steam_kgph * KGPH / MW["H2O"]}, T_STEAM, P_STEAM)
    _fix_port(fs.water_feed.outlet, {"H2O": 0.1}, 298.15, 1.0 * BAR)
    fs.water_feed.outlet.flow_mol[0].unfix()        # solved by DEF spec
    fs.water_feed.outlet.flow_mol[0].setlb(0.0)

    dp = fs.def_prod.properties[0]
    fs.def_mass = Expression(
        expr=dp.flow_mol * sum(dp.mole_frac_comp[c] * MW[c] for c in COMPS)
        / KGPH, doc="DEF product [kg/h]")
    fs.w_urea = Expression(
        expr=dp.mole_frac_comp["UREA"] * MW["UREA"]
        / sum(dp.mole_frac_comp[c] * MW[c] for c in COMPS))
    fs.def_spec = Constraint(expr=fs.w_urea == W_DEF)
    return m


# ---------------------------------------------------------- initialize
def initialize(m, passes=12):
    fs = m.fs
    solver = get_solver("ipopt", options=dict(_OPTARG))
    n_co2 = value(fs.co2_feed.outlet.flow_mol[0])

    # stage 1: hold NH3 make-up, conversion and dilution water fixed
    fs.nc_spec.deactivate()
    fs.x_def.deactivate()
    fs.x_co2.fix(0.6)
    fs.def_spec.deactivate()
    fs.nh3_feed.outlet.flow_mol[0].fix(2.0 * n_co2)
    fs.water_feed.outlet.flow_mol[0].fix(0.05)

    for f in (fs.co2_feed, fs.nh3_feed, fs.steam_feed, fs.water_feed):
        f.initialize(outlvl=50)
    # recycle guess: carbamate solution carrying the excess NH3 of N/C=4
    rec_guess = {"NH3": 2.2 * n_co2, "CARB": 0.4 * n_co2,
                 "H2O": 0.6 * n_co2, "CO2": EPS}

    def init(u, *arcs):
        for a in arcs:
            _prop(a)
        was_fixed = {id(v): v.fixed for v in u.component_data_objects(Var)}
        try:
            u.initialize(outlvl=50, optarg=_OPTARG)
        except Exception:
            # a failed IDAES initialize raises before releasing the inlet
            # state it fixed - restore the free/fixed pattern first
            for v in u.component_data_objects(Var):
                if v.fixed and not was_fixed.get(id(v), False):
                    v.unfix()
            ports = [p for p in u.component_objects(
                     ctype=type(fs.co2_feed.outlet), descend_into=False)
                     if "out" not in p.local_name
                     and p.local_name not in ("vap_outlet", "liq_outlet",
                                              "to_lp", "to_def")]
            fixed = []
            for p in ports:
                for v in p.vars.values():
                    for i in v:
                        if not v[i].fixed:
                            v[i].fix()
                            fixed.append(v[i])
            res = solver.solve(u)
            for v in fixed:
                v.unfix()
            tc = str(res.solver.termination_condition)
            if tc != "optimal":
                print(f"  warning: {u.name} init {tc}")

    _set_port(fs.rec_pump.inlet, rec_guess, T_NH3C, P_MP)
    for it in range(passes):
        init(fs.nh3_pump, fs.s_nh3a)
        init(fs.rec_pump)
        init(fs.m_r, fs.s_co2, fs.s_nh3b, fs.s_rec)
        init(fs.reactor, fs.s_rin)
        init(fs.v_mp, fs.s_rout)
        init(fs.f_mp, fs.s_mpin)
        init(fs.v_lp, fs.s_mpl)
        init(fs.f_lp, fs.s_lpin)
        init(fs.v_pu, fs.s_lpl)
        init(fs.v_steam, fs.s_st1)
        init(fs.m_strip, fs.s_pu, fs.s_st2)
        init(fs.stripper, fs.s_strin)
        init(fs.vac_cond, fs.s_strv)
        init(fs.cond_pump, fs.s_vacl)
        init(fs.cond_split, fs.s_cnd)
        init(fs.v_cond_def, fs.s_cdef1)
        init(fs.def_cool, fs.s_strl)
        init(fs.m_def, fs.s_defa, fs.s_wat, fs.s_cdef2)
        init(fs.def_trim, fs.s_defb)
        init(fs.m_lpc, fs.s_lpv, fs.s_clp)
        init(fs.lpc, fs.s_lpc1)
        init(fs.f_lpc, fs.s_lpc2)
        init(fs.lp_pump, fs.s_lpcl)
        init(fs.lp_comp, fs.s_lpcv)
        init(fs.m_mpc, fs.s_mpv, fs.s_lpp, fs.s_lpg)
        init(fs.mpc, fs.s_mpc1)
        init(fs.nh3_cond, fs.s_mpc2)
        init(fs.f_mpc, fs.s_mpc3)
        old = value(fs.rec_pump.inlet.flow_mol[0])
        _prop(fs.s_recl)
        new = value(fs.rec_pump.inlet.flow_mol[0])
        print(f"  tear pass {it + 1}: recycle = {new:.4f} mol/s")
        if abs(new - old) < 1e-4 * max(new, 1e-6):
            break
    for a in (fs.s_vacv, fs.s_vent, fs.s_defc):
        _prop(a)
    for p in (fs.vent_vac, fs.vent_mp, fs.def_prod):
        p.initialize(outlvl=50)
    return solver


def solve(m, solver):
    fs = m.fs
    # stage 2: recycle closed, conversion still fixed
    res = solver.solve(m)
    print("  closed loop (fixed X):", res.solver.termination_condition)
    # stage 3: release the specs one at a time
    fs.x_co2.unfix()
    fs.x_def.activate()
    res = solver.solve(m)
    print("  + conversion trend:", res.solver.termination_condition)
    fs.nh3_feed.outlet.flow_mol[0].unfix()
    fs.nc_spec.activate()
    res = solver.solve(m)
    print("  + N/C = 4 spec:", res.solver.termination_condition)
    fs.water_feed.outlet.flow_mol[0].unfix()
    fs.def_spec.activate()
    print(f"  dof = {degrees_of_freedom(m)}")
    res = solver.solve(m)
    for _ in range(2):
        if str(res.solver.termination_condition) == "optimal":
            break
        res = solver.solve(m)
    print("  + DEF 32.5 wt% spec:", res.solver.termination_condition)
    return res


# -------------------------------------------------------------- report
def co2_compression_kw(n_mol_s, p_in=1.0 * BAR, p_out=P_SYN, stages=4,
                       T_in=313.15, k=1.29, eta=ETA_COMP):
    """Intercooled multistage ideal-gas compression work [kW]."""
    r = (p_out / p_in) ** (1.0 / stages)
    w_stage = (k / (k - 1)) * 8.314 * T_in * (r ** ((k - 1) / k) - 1) / eta
    return stages * w_stage * n_mol_s / 1e3


def report(m):
    fs = m.fs
    kw = lambda e: value(e) / 1e3  # noqa: E731

    def mass(port, c=None):
        F = value(port.flow_mol[0])
        if c:
            return F * value(port.mole_frac_comp[0, c]) * MW[c] / KGPH
        return F * sum(value(port.mole_frac_comp[0, j]) * MW[j]
                       for j in COMPS) / KGPH

    print("\n============ low-pressure DEF plant (US 10,759,745 concept) "
          "============")
    print(f"reactor: P = {P_SYN / BAR:.0f} bar, T = {T_SYN - 273.15:.0f} C, "
          f"N/C = {value(fs.nc_ratio):.2f}, H/C = {value(fs.hc_ratio):.2f}, "
          f"CO2 conversion = {value(fs.x_co2):.3f}")
    print(f"feeds: CO2 {mass(fs.co2_feed.outlet):.2f} kg/h, NH3 make-up "
          f"{mass(fs.nh3_feed.outlet):.2f} kg/h, LP steam "
          f"{mass(fs.steam_feed.outlet):.1f} kg/h, demin water "
          f"{mass(fs.water_feed.outlet):.1f} kg/h")
    p = fs.def_prod.inlet
    print(f"DEF product: {value(fs.def_mass):.2f} kg/h, urea "
          f"{mass(p, 'UREA'):.2f} kg/h ({value(fs.w_urea) * 100:.1f} wt%), "
          f"NH3 {mass(p, 'NH3') / max(value(fs.def_mass), 1e-9) * 1e6:.0f} "
          f"ppm (ISO limit 2000 as NH3)")
    print(f"carbamate recycle to HP: {mass(fs.rec_pump.inlet):.1f} kg/h "
          f"(water {mass(fs.rec_pump.inlet, 'H2O'):.1f} kg/h)")
    print(f"vents: MP {mass(fs.vent_mp.inlet):.3f} kg/h, vacuum "
          f"{mass(fs.vent_vac.inlet):.3f} kg/h")

    n_co2 = value(fs.co2_feed.outlet.flow_mol[0])
    elec = {
        "CO2 compressor 1->150 bar (analytic)": co2_compression_kw(n_co2),
        "NH3 HP pump": kw(fs.nh3_pump.work_mechanical[0]),
        "carbamate HP pump": kw(fs.rec_pump.work_mechanical[0]),
        "LP->MP pump": kw(fs.lp_pump.work_mechanical[0]),
        "LP gas compressor 4->20 bar (analytic)": co2_compression_kw(
            value(fs.lp_comp.inlet.flow_mol[0]), p_in=P_LP, p_out=P_MP,
            stages=1, T_in=value(fs.lp_comp.inlet.temperature[0])),
        "condensate pump": kw(fs.cond_pump.work_mechanical[0]),
    }
    heat_in = {
        "MP decomposer (160 C)": kw(fs.f_mp.heat_duty[0]),
        "LP decomposer (135 C)": kw(fs.f_lp.heat_duty[0]),
        "reactor": kw(fs.reactor.heat_duty[0]),
        "stripping steam (latent+sensible vs 25 C liquid)":
            value(fs.steam_feed.outlet.flow_mol[0]) * (
                75.4 * (373.15 - 298.15) + 40.7e3
                + 34.0 * (T_STEAM - 373.15)) / 1e3,
    }
    cool = {
        "vacuum condenser": kw(fs.vac_cond.heat_duty[0]),
        "LP carbamate condenser": kw(fs.lpc.heat_duty[0])
        + kw(fs.f_lpc.heat_duty[0]),
        "MP carbamate condenser": kw(fs.mpc.heat_duty[0]),
        "NH3 condenser": kw(fs.nh3_cond.heat_duty[0])
        + kw(fs.f_mpc.heat_duty[0]),
        "DEF cooler": kw(fs.def_cool.heat_duty[0])
        + kw(fs.def_trim.heat_duty[0]),
    }
    urea_kgph = mass(p, "UREA")
    print("\nelectric / shaft [kW]")
    for k_, v in elec.items():
        print(f"  {k_:48s} {v:8.3f}")
    print(f"  {'TOTAL':48s} {sum(elec.values()):8.3f}")
    print("heat input (positive = heating) [kW]")
    for k_, v in heat_in.items():
        print(f"  {k_:48s} {v:8.3f}")
    heat_pos = sum(v for v in heat_in.values() if v > 0)
    print(f"  {'TOTAL heating':48s} {heat_pos:8.3f}")
    print("cooling [kW]")
    for k_, v in cool.items():
        print(f"  {k_:48s} {v:8.3f}")
    if urea_kgph > 0:
        print(f"\nspecific: {sum(elec.values()) / urea_kgph * 1e3:.0f} kWh_e/t "
              f"urea, {heat_pos / urea_kgph * 1e3:.0f} kWh_th/t urea "
              f"({urea_kgph:.2f} kg/h urea)")
    hp = elec["CO2 compressor 1->150 bar (analytic)"] + elec["NH3 HP pump"] \
        + elec["carbamate HP pump"]
    print(f"HP-section shaft load: {hp:.3f} kW "
          f"({hp / max(sum(elec.values()), 1e-9) * 100:.0f}% of electric)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--co2", type=float, default=15.0, help="CO2 feed kg/h")
    ap.add_argument("--steam", type=float, default=20.0,
                    help="LP stripping steam kg/h")
    ap.add_argument("--direct", type=float, default=0.0,
                    help="fraction of stripper condensate sent straight to "
                         "DEF dilution instead of the HP loop")
    a = ap.parse_args()
    m = build_model(fco2_kgph=a.co2, steam_kgph=a.steam,
                    direct_frac=a.direct)
    s = initialize(m)
    solve(m, s)
    report(m)


if __name__ == "__main__":
    main()

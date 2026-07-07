"""
IDAES replica of the ureaF (Bosch-Meiser urea synthesis) Aspen Plus
flowsheet, extracted block-by-block from ``aspen/ureaF.bkp`` (based on
Pustjens & Van Den Tillaart, US 10,759,745 B2, Stamicarbon). See
``personal/urea_idaes_plan.md`` for the full extraction tables.

Topology (Aspen block -> IDAES unit; pressures from the .bkp with
unit-row <8> = kgf/cm2, so the synthesis loop "141" = 138.3 bar):

  HP synthesis loop (200 degC / 138.3 bar)
    NH3-IN + S41 -> B5(mix) -> NH3P1(20 bar) -> NH3SP1(10% to S17)
      -> NH3P2(200 bar) -> B6(200 degC) -> M01(mix, + CO2-IN + S31 + REC3)
      -> B27(200 degC, 200 bar) -> R01(CSTR, isothermal 200 degC,
         V = pi/4 * 2.5^2 * 20 m3 - the GRID geometry, kinetics RXNTOT)
  MP decomposition / recycle (20 bar). The urea-bearing LIQUID runs down
  the decomposition trains; gas branches feed the condensers (the .bkp
  M-codes are graphics indices, not phase tags - see the arc block).
    B2(valve) -> F01(160C flash) -[L]-> A02(carbamate decomp) -> B1(160C
      flash) -[V]-> A01(RGibbs repl.) -> B11 <- F01[V]; B11+S34 -> B15
      -> B10hot(40C) -> B12(100C flash) -[L]-> B9(0.9 REC1 -> B13 pump ->
      M01, the carbamate-solution recycle); -[V]-> B7 column -[V]->
      B19(0.9 REC2 -> B16hot(40C) -> B17(40C/1bar flash) -[L aqua
      ammonia]-> S41 recycle, -[V]-> S42); B7[L] = S34
  LP decomposition (4 bar)
    B1[L] -> B22(valve) -> F012(135C flash) -[L]-> A022(decomp) ->
      CARBMIX2 <- F012[V]; -> B8(135C flash) -[V]-> A012(RGibbs repl.)
      -> B3(20C) -> B18(1 bar adiabatic flash) -[L]-> NH3OUT
  Finishing (5 -> 1 bar)
    B8[L] -> X3(valve) -> B4(1 bar adiabatic flash) -[V]-> A013(RGibbs
      repl.) -> B24 <- B18[V]; -> B20hot(40C) -> NH3OUT3
    B4[L] -> P1(5 bar) -> [S5: design-spec stream, 60 wt% urea] -> B25
      column (+STRIPH2O) -[L]-> B28 column -> DEF-OUT (product) + S30
    B25[V] -> B23(valve) -> NH3OUT2; NH3OUT+NH3OUT2+NH3OUT3 -> B14 ->
      B26(0.95 REC3 -> M01, 5% PURGE3)

Simplifications, all documented inline where they bite:
  * B21 (single-inlet mixer) is skipped (S46 == S47).
  * HeatX B10/B16/B20 run in Aspen shortcut mode with a hot-outlet-T spec
    against dummy cooling-water streams; only the hot side is modeled
    (a Heater to 40 degC); the CW side duty is minus the hot duty.
  * R01 is a single CSTR rather than a 20-element PFR: at grid-scale
    feeds the 98 m3 reactor is orders of magnitude oversized, so the
    outlet sits on the two kinetic-pair equilibria regardless of
    discretisation (verified in test_urea_kinetics.py).
  * A02/A022 (RStoic, 100% carbamate decomposition) reuse the CARBF
    stoichiometry with a negative extent pinned to the inlet carbamate.
  * A01/A012/A013 (RGibbs) use the conversion-form carbamate condensation
    replacement (see reactions/urea_reactions.py).

COLUMN MODULARITY: the three columns (B7 absorber/rectifier, B25
stripper, B28 DISTL concentrator) are built through the COLUMN_IMPLS
registry. An implementation is a dict of three builder functions, each
returning a dict of named ports; the flowsheet wires arcs only to those
port names, so a rigorous TrayColumn - or the reactive distillation
variant of B7 (the .bkp defines the vapor-phase equilibrium set REACT1
= "2NH3 + CO2 <-> CARB" for exactly that purpose, though it is attached
to no block in the shipped model) - can be dropped in without touching
the rest of the flowsheet:

    port contract
      b7 : {"s32_in", "s17_in", "s27_out" (vap), "s34_out" (liq)}
      b25: {"s5_in", "steam_in", "s16_out" (vap), "urea_out" (liq)}
      b28: {"feed_in", "def_out" (distillate), "s30_out" (bottoms)}

The default "single_stage" implementation honours each column's .bkp
specification with one equilibrium stage: B7 = flash with V/F = the D:F
= 0.9 spec at 20 bar; B25 = flash with the 50,000 kJ/h (13.9 kW)
reboiler duty at 5 bar; B28 = mole-balance splitter reproducing DISTL's
D:F = 0.8 spillover (all volatiles + water overhead first, urea/carb
overflow after - with liquid-only urea this is what Aspen's shortcut
does through its own mole balance).

Run:  conda activate idaes-pse && python urea_flowsheet.py [Fnh3 Fco2]
      (defaults Fnh3 = 10 kg/h, Fco2 = 15 kg/h; validation rows exist in
      aspen/data/ureaF_results_live.csv)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    TransformationFactory,
    units as pyunits,
    value,
    Var,
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.core.util.math import smooth_min
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.base.generic_reaction import (
    GenericReactionParameterBlock,
)
from idaes.models.unit_models import (
    CSTR,
    Feed,
    Flash,
    Heater,
    Mixer,
    Product,
    Pump,
    Separator,
    StoichiometricReactor,
)
from idaes.models.unit_models.mixer import MomentumMixingType
from idaes.models.unit_models.separator import SplittingType

# swap either import (same signatures) to change the thermo/kinetics basis;
# properties/srk_urea.py is the faithful RKS variant, blocked on this
# machine by the cubic_roots.so platform issue documented there
from properties.ideal_urea import get_prop
from reactions.urea_reactions import (
    add_carbeq_conversion,
    get_carb_stoich_rxn,
    get_rxn,
)

# ---------------------------------------------------------------------------
# Specifications extracted from aspen/ureaF.bkp
# ---------------------------------------------------------------------------
KGF = 0.980665e5  # Pa per kgf/cm2 (Aspen pressure unit-row <8>)

P_SYN = 141.0 * KGF  # R01 / CO2-IN / A01 / A013 pressure, 138.27 bar
P_NH3_FEED = 160.0 * KGF  # NH3-IN, 156.9 bar
P_PUMP_HP = 200e5  # NH3P2 / B13 / B27 discharge
P_MP = 20e5  # B2 letdown, F01/A02/B1/B12/B7 level
P_LP = 4e5  # B22 letdown, F012/A022/B8/A012/B3 level
P_ATM = 1e5  # X3/B23/B17/B18/B4/B28 level
P_B25 = 5e5  # P1 discharge / B25 stage pressure

T_NH3_FEED = 307.15  # 34 degC
T_CO2_FEED = 373.15  # 100 degC
T_SYN = 473.15  # 200 degC (B6, B27, R01)
T_MP_FLASH = 433.15  # 160 degC (F01, A02, B1)
T_LP_FLASH = 408.15  # 135 degC (F012, A022, B8)
T_GIBBS = 345.55  # 72.4 degC (A01, A012, A013)
T_B12 = 373.15  # 100 degC
T_B3 = 293.15  # 20 degC
T_HX_HOT_OUT = 313.15  # 40 degC (B10, B16, B20 hot-side spec; B17 flash)
T_STEAM = 433.15  # STRIPH2O, 160 degC

R01_VOLUME = 3.141592653589793 / 4.0 * 2.5**2 * 20.0  # m3, grid geometry
FRAC_REC1 = 0.9  # B9
FRAC_REC2 = 0.9  # B19
FRAC_REC3 = 0.95  # B26
FRAC_S17 = 0.1  # NH3SP1 (design spec D1 varies this in 0.1-0.9)
B7_D_F = 0.9  # B7 COL-SPECS D:F (mole basis)
B25_QN = 50000e3 / 3600.0  # W; .bkp QN = 50000 kJ/h = 13.89 kW (reported)
# B25 single-stage bottoms temperature. Water saturation at 5 bar is
# 425 K; sitting exactly there makes the water split degenerate, and the
# value controls how much water leaves with the urea bottoms (i.e. the
# DEF-OUT urea fraction). This is THE calibration knob of the stripper
# surrogate - a rigorous TrayColumn implementation replaces it.
T_B25_BOT = 427.15
B28_D_F = 0.8  # B28 DISTL D:F
STEAM_KGPH = 60.0  # STRIPH2O
DECOMP_CONV = 0.9999  # A02/A022 RStoic CONV = 1.0 (kept off the corner)
PUMP_ETA = 0.75  # nominal; pump work is not a validated output

COMPS = ["NH3", "CO2", "H2O", "UREA", "CARB"]
MW = {"NH3": 17.0305e-3, "CO2": 44.0095e-3, "H2O": 18.0153e-3,
      "UREA": 60.0553e-3, "CARB": 78.0707e-3}
EPS = 1e-8
W_TO_MMKCAL_H = 1.0 / 1.163e6

_KGPH = 1.0 / 3600.0  # kg/h -> kg/s

# solver options for every unit-level initialization solve; several of the
# mixed flashes (notably B5's liquid-NH3 + recycle-vapor merge) need more
# than IPOPT's default iteration budget from a cold start, and the
# adaptive barrier strategy is what gets IPOPT through states that sit
# exactly on a phase boundary (saturated flash liquids feeding the next
# unit) - the default monotone strategy stalls there
_OPTARG = {"tol": 1e-6, "max_iter": 3000, "mu_strategy": "adaptive"}


# ---------------------------------------------------------------------------
# Small helpers (ammonia-flowsheet conventions)
# ---------------------------------------------------------------------------

def _fracs(comp_mol_flows):
    """Trace-padded, exactly normalised mole fractions (sum == 1)."""
    total = sum(max(f, 0.0) for f in comp_mol_flows.values())
    raw = {c: max(comp_mol_flows.get(c, 0.0), EPS * total) for c in COMPS}
    adj = sum(raw.values())
    return total, {c: f / adj for c, f in raw.items()}


def _set_state(port, comp_mol_flows, T, P):
    total, fracs = _fracs(comp_mol_flows)
    port.flow_mol[0].fix(total)
    for c in COMPS:
        port.mole_frac_comp[0, c].fix(fracs[c])
    port.temperature[0].fix(T)
    port.pressure[0].fix(P)


def _guess_state(state, comp_mol_flows, T, P):
    total, fracs = _fracs(comp_mol_flows)
    state.flow_mol.set_value(total)
    for c in COMPS:
        state.mole_frac_comp[c].set_value(fracs[c])
    state.temperature.set_value(T)
    state.pressure.set_value(P)


def _prop(arc):
    """propagate_state + clamp tiny negative trace mole fractions."""
    propagate_state(arc)
    port = arc.destination
    if hasattr(port, "mole_frac_comp"):
        for idx in port.mole_frac_comp:
            v = port.mole_frac_comp[idx].value
            if v is not None and v < EPS:
                port.mole_frac_comp[idx].set_value(EPS)


def _valve(fs, name, p_out):
    """Aspen Valve -> isenthalpic Heater with fixed outlet pressure."""
    unit = Heater(property_package=fs.props, has_pressure_change=True)
    setattr(fs, name, unit)
    unit.heat_duty.fix(0.0)
    unit.control_volume.properties_out[0].pressure.fix(p_out)
    return unit


def _heater(fs, name, t_out, p_out=None):
    """Heater with fixed outlet T (and optionally outlet P)."""
    unit = Heater(property_package=fs.props,
                  has_pressure_change=p_out is not None)
    setattr(fs, name, unit)
    unit.control_volume.properties_out[0].temperature.fix(t_out)
    if p_out is not None:
        unit.control_volume.properties_out[0].pressure.fix(p_out)
    return unit


def _flash(fs, name, t_out=None, p_out=None, duty=None):
    """Flash: fix any two of (T, P-change/P, Q). deltaP = 0 if no p_out."""
    unit = Flash(property_package=fs.props,
                 has_pressure_change=True)
    setattr(fs, name, unit)
    if t_out is not None:
        unit.control_volume.properties_out[0].temperature.fix(t_out)
    if duty is not None:
        unit.heat_duty.fix(duty)
    if p_out is not None:
        unit.control_volume.properties_out[0].pressure.fix(p_out)
    else:
        unit.deltaP.fix(0.0)
    return unit


def init_cstr_ramped(cstr, state_args=None,
                     steps=(1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0), outlvl=50):
    """Initialize an equilibrium-dominated CSTR by volume homotopy.

    The RXNTOT pairs pin two equilibria with large rate constants, so a
    cold start at the full (heavily oversized) reactor volume is too
    stiff for IPOPT. Solving at 1e-4 x V (extent ~ 0) and ramping the
    volume up walks the outlet smoothly onto the equilibrium manifold;
    each step starts from the previous converged state. Intermediate
    steps are allowed to fail (only the final full-volume step must
    initialize).
    """
    v_target = cstr.volume[0.0].value
    for i, f in enumerate(steps):
        cstr.volume.fix(f * v_target)
        try:
            cstr.initialize(
                outlvl=outlvl,
                state_args=state_args if i == 0 else None,
                optarg=_OPTARG,
            )
        except Exception:
            if f == steps[-1]:
                cstr.volume.fix(v_target)
                raise
    cstr.volume.fix(v_target)


def _pump(fs, name, p_out):
    unit = Pump(property_package=fs.props)
    setattr(fs, name, unit)
    unit.control_volume.properties_out[0].pressure.fix(p_out)
    unit.efficiency_pump.fix(PUMP_ETA)
    return unit


def _decomp_reactor(fs, name, t_out, p_out):
    """A02/A022: total carbamate decomposition at fixed T (RStoic CONV=1).

    Implemented as the CARBF reaction run backward: extent pinned at
    -DECOMP_CONV * inlet carbamate.
    """
    unit = StoichiometricReactor(
        property_package=fs.props,
        reaction_package=fs.carb_rxns,
        has_heat_of_reaction=False,
        has_heat_transfer=True,
        has_pressure_change=True,
    )
    setattr(fs, name, unit)
    unit.control_volume.properties_out[0].temperature.fix(t_out)
    unit.control_volume.properties_out[0].pressure.fix(p_out)
    st_in = unit.control_volume.properties_in[0]
    unit.decomp_extent = Constraint(
        expr=unit.control_volume.rate_reaction_extent[0, "CARBF"]
        == -DECOMP_CONV * st_in.flow_mol * st_in.mole_frac_comp["CARB"]
    )
    return unit


def _gibbs_replacement(fs, name, p_out):
    """A01/A012/A013: carbamate condensation to the equilibrium limit."""
    unit = StoichiometricReactor(
        property_package=fs.props,
        reaction_package=fs.carb_rxns,
        has_heat_of_reaction=False,
        has_heat_transfer=True,
        has_pressure_change=True,
    )
    setattr(fs, name, unit)
    unit.control_volume.properties_out[0].temperature.fix(T_GIBBS)
    unit.control_volume.properties_out[0].pressure.fix(p_out)
    add_carbeq_conversion(unit)
    return unit


# ---------------------------------------------------------------------------
# Column implementations (swappable - see module docstring port contract)
# ---------------------------------------------------------------------------

def _build_b7_single_stage(fs):
    """B7 (RadFrac, 10 stages, partial-V condenser, 20 bar, D:F = 0.9) as
    one equilibrium stage: mix the two feeds, flash with V/F = 0.9."""
    fs.b7_mix = Mixer(
        property_package=fs.props,
        inlet_list=["s32", "s17"],
        momentum_mixing_type=MomentumMixingType.none,
    )
    fs.b7_mix.mixed_state[0].pressure.fix(P_MP)
    fs.b7 = Flash(property_package=fs.props, has_pressure_change=True)
    fs.b7.deltaP.fix(0.0)
    fs.b7.control_volume.properties_out[0].phase_frac["Vap"].fix(B7_D_F)
    fs.a_b7 = Arc(source=fs.b7_mix.outlet, destination=fs.b7.inlet)
    return {
        "s32_in": fs.b7_mix.s32,
        "s17_in": fs.b7_mix.s17,
        "s27_out": fs.b7.vap_outlet,
        "s34_out": fs.b7.liq_outlet,
    }


def _build_b25_single_stage(fs):
    """B25 (RadFrac stripper, no condenser, reboiler duty 50,000 kJ/h,
    stage-5 P = 5 bar, live steam to the bottom) as one equilibrium stage.

    The .bkp spec is the reboiler DUTY, but on one stage at grid-scale
    feeds 13.9 kW drives the outlet far past any physical column state
    (a 10-stage column disposes of that heat through its internal vapor
    traffic). The single-stage stand-in therefore fixes the outlet at the
    5-bar water boiling point (152 degC - the stripper bottoms
    temperature) and reports the computed duty for comparison against
    Aspen's QB25_reb instead of imposing it."""
    fs.b25_mix = Mixer(
        property_package=fs.props,
        inlet_list=["s5", "steam"],
        momentum_mixing_type=MomentumMixingType.none,
    )
    fs.b25_mix.mixed_state[0].pressure.fix(P_B25)
    fs.b25 = Flash(property_package=fs.props, has_pressure_change=True)
    fs.b25.deltaP.fix(0.0)
    fs.b25.control_volume.properties_out[0].temperature.fix(T_B25_BOT)
    fs.a_b25 = Arc(source=fs.b25_mix.outlet, destination=fs.b25.inlet)
    return {
        "s5_in": fs.b25_mix.s5,
        "steam_in": fs.b25_mix.steam,
        "s16_out": fs.b25.vap_outlet,
        "urea_out": fs.b25.liq_outlet,
    }


def _build_b28_single_stage(fs):
    """B28 (DISTL shortcut, 5 stages, RR = 1, 1 bar, D:F = 0.8) as a
    mole-balance splitter: the distillate (S30, the water/NH3 overhead)
    takes the volatiles+water up to 0.8F, with heavies overflowing only
    if the lights run short; DEF-OUT is the BOTTOMS - the urea-enriched
    20% of the moles. That identification is forced by the recorded
    results: Wurea spans 0.005-0.72 with median 0.34 (DEF grade is
    32.5 wt% urea), which no distillate of a column whose urea relative
    volatility is ~1e-3 could reach, while a 0.2F bottoms cut does so by
    plain mole balance. Duties of the shortcut column are not reproduced
    (inherent to the surrogate; excluded from validation)."""
    fs.b28 = Separator(
        property_package=fs.props,
        outlet_list=["dist", "bot"],
        split_basis=SplittingType.componentFlow,
    )
    st = fs.b28.mixed_state[0]
    f_tot = st.flow_mol
    f_light = sum(f_tot * st.mole_frac_comp[c] for c in ("NH3", "CO2", "H2O"))
    f_heavy = sum(f_tot * st.mole_frac_comp[c] for c in ("UREA", "CARB"))
    d_light = smooth_min(f_light, B28_D_F * f_tot, eps=1e-6)

    sf = fs.b28.split_fraction
    fs.b28.eq_light_a = Constraint(expr=sf[0, "dist", "NH3"]
                                   == sf[0, "dist", "CO2"])
    fs.b28.eq_light_b = Constraint(expr=sf[0, "dist", "NH3"]
                                   == sf[0, "dist", "H2O"])
    fs.b28.eq_light_bal = Constraint(
        expr=sf[0, "dist", "NH3"] * f_light == d_light)
    fs.b28.eq_heavy_a = Constraint(expr=sf[0, "dist", "UREA"]
                                   == sf[0, "dist", "CARB"])
    fs.b28.eq_heavy_bal = Constraint(
        expr=sf[0, "dist", "UREA"] * (f_heavy + 1e-8)
        == B28_D_F * f_tot - d_light)
    return {
        "feed_in": fs.b28.inlet,
        "def_out": fs.b28.bot,
        "s30_out": fs.b28.dist,
    }


COLUMN_IMPLS = {
    "single_stage": {
        "b7": _build_b7_single_stage,
        "b25": _build_b25_single_stage,
        "b28": _build_b28_single_stage,
    },
    # future: "tray" (TrayColumn) and "reactive" (B7 with the .bkp's
    # REACT1 vapor equilibrium on the stages) - register builders here
    # honouring the same port contract.
}


# ---------------------------------------------------------------------------
# Flowsheet builder
# ---------------------------------------------------------------------------

def build_model(fnh3_kgph=10.0, fco2_kgph=15.0, column_impl="single_stage"):
    cols = COLUMN_IMPLS[column_impl]

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.props = GenericParameterBlock(**get_prop(("Vap", "Liq")))
    m.fs.rxns = GenericReactionParameterBlock(**get_rxn(m.fs.props))
    m.fs.carb_rxns = GenericReactionParameterBlock(
        **get_carb_stoich_rxn(m.fs.props))
    fs = m.fs

    # ---------------- feeds ----------------
    fs.nh3_feed = Feed(property_package=fs.props)
    fs.co2_feed = Feed(property_package=fs.props)
    fs.steam_feed = Feed(property_package=fs.props)

    # ---------------- HP synthesis loop ----------------
    # Aspen's B5 mixes at the (1 bar) S41 pressure and NH3P1 immediately
    # repressurises to 20 bar. The adiabatic 1-bar mix sits right on the
    # NH3 boiling point and defeats every initializer, so the mix is done
    # at the 20 bar pump discharge instead - same downstream state, only
    # the (unvalidated) NH3P1 pump work differs.
    fs.b5 = Mixer(property_package=fs.props, inlet_list=["nh3", "s41"],
                  momentum_mixing_type=MomentumMixingType.none)
    fs.b5.mixed_state[0].pressure.fix(P_MP)
    _pump(fs, "nh3p1", P_MP)
    fs.nh3sp1 = Separator(property_package=fs.props,
                          outlet_list=["s17", "s19"],
                          split_basis=SplittingType.totalFlow)
    fs.nh3sp1.split_fraction[0, "s17"].fix(FRAC_S17)
    _pump(fs, "nh3p2", P_PUMP_HP)
    _heater(fs, "b6", T_SYN)  # Heater without pressure change: deltaP = 0
    _pump(fs, "b13", P_PUMP_HP)
    fs.m01 = Mixer(property_package=fs.props,
                   inlet_list=["co2", "s6", "s31", "rec3"],
                   momentum_mixing_type=MomentumMixingType.none)
    fs.m01.mixed_state[0].pressure.fix(P_SYN)
    _heater(fs, "b27", T_SYN, p_out=P_PUMP_HP)
    fs.r01 = CSTR(property_package=fs.props, reaction_package=fs.rxns,
                  has_heat_of_reaction=False, has_heat_transfer=True,
                  has_pressure_change=True)
    fs.r01.volume.fix(R01_VOLUME)
    fs.r01.control_volume.properties_out[0].temperature.fix(T_SYN)
    fs.r01.control_volume.properties_out[0].pressure.fix(P_SYN)

    # ---------------- MP train (20 bar) ----------------
    _valve(fs, "b2", P_MP)
    _flash(fs, "f01", t_out=T_MP_FLASH)
    _decomp_reactor(fs, "a02", T_MP_FLASH, P_MP)
    _flash(fs, "b1", t_out=T_MP_FLASH)
    # .bkp records PRES = 141 kgf/cm2 on A01/A013 (sic - they are fed from
    # the 20 bar / 1 bar levels; a probable copy-paste from R01). In Aspen
    # the value is inert because RGibbs there is declared single-phase
    # liquid (VAPOR = NO), where pressure barely enters the Gibbs
    # minimisation. Reproducing it here would force each unit's two-phase
    # outlet flash to sit at a spurious 138-bar phase boundary, so both
    # units run at their train's pressure level instead.
    _gibbs_replacement(fs, "a01", P_MP)
    fs.b11 = Mixer(property_package=fs.props, inlet_list=["s23", "s24"],
                   momentum_mixing_type=MomentumMixingType.none)
    fs.b11.mixed_state[0].pressure.fix(P_MP)
    fs.b15 = Mixer(property_package=fs.props, inlet_list=["s13", "s34"],
                   momentum_mixing_type=MomentumMixingType.none)
    fs.b15.mixed_state[0].pressure.fix(P_MP)
    _heater(fs, "b10h", T_HX_HOT_OUT)  # HeatX B10 hot side
    _flash(fs, "b12", t_out=T_B12)
    fs.b9 = Separator(property_package=fs.props,
                      outlet_list=["rec1", "purge1"],
                      split_basis=SplittingType.totalFlow)
    fs.b9.split_fraction[0, "rec1"].fix(FRAC_REC1)

    b7_ports = cols["b7"](fs)

    fs.b19 = Separator(property_package=fs.props,
                       outlet_list=["rec2", "purge2"],
                       split_basis=SplittingType.totalFlow)
    fs.b19.split_fraction[0, "rec2"].fix(FRAC_REC2)
    _heater(fs, "b16h", T_HX_HOT_OUT)  # HeatX B16 hot side
    _flash(fs, "b17", t_out=T_HX_HOT_OUT, p_out=P_ATM)

    # ---------------- LP train (4 bar) ----------------
    _valve(fs, "b22", P_LP)
    _flash(fs, "f012", t_out=T_LP_FLASH)
    _decomp_reactor(fs, "a022", T_LP_FLASH, P_LP)
    fs.carbmix2 = Mixer(property_package=fs.props, inlet_list=["s11", "s2"],
                        momentum_mixing_type=MomentumMixingType.none)
    fs.carbmix2.mixed_state[0].pressure.fix(P_LP)
    _flash(fs, "b8", t_out=T_LP_FLASH)
    _gibbs_replacement(fs, "a012", P_LP)
    _heater(fs, "b3", T_B3, p_out=P_LP)
    _flash(fs, "b18", duty=0.0, p_out=P_ATM)

    # ---------------- finishing ----------------
    _valve(fs, "x3", P_ATM)
    _flash(fs, "b4", duty=0.0, p_out=P_ATM)
    _gibbs_replacement(fs, "a013", P_ATM)  # .bkp 141 kgf/cm2: see A01 note
    fs.b24 = Mixer(property_package=fs.props, inlet_list=["s54", "s55"],
                   momentum_mixing_type=MomentumMixingType.none)
    fs.b24.mixed_state[0].pressure.fix(P_ATM)
    _heater(fs, "b20h", T_HX_HOT_OUT)  # HeatX B20 hot side
    _pump(fs, "p1", P_B25)

    b25_ports = cols["b25"](fs)
    _valve(fs, "b23", P_ATM)
    fs.b14 = Mixer(property_package=fs.props,
                   inlet_list=["nh3out", "nh3out3", "nh3out2"],
                   momentum_mixing_type=MomentumMixingType.none)
    fs.b14.mixed_state[0].pressure.fix(P_ATM)
    fs.b26 = Separator(property_package=fs.props,
                       outlet_list=["rec3", "purge3"],
                       split_basis=SplittingType.totalFlow)
    fs.b26.split_fraction[0, "rec3"].fix(FRAC_REC3)
    b28_ports = cols["b28"](fs)

    # ---------------- products ----------------
    for name in ("purge1_prod", "purge2_prod", "purge3_prod",
                 "s42_prod", "def_prod", "s30_prod"):
        setattr(fs, name, Product(property_package=fs.props))

    # ---------------- arcs ----------------
    fs.a01a = Arc(source=fs.nh3_feed.outlet, destination=fs.b5.nh3)
    fs.a02a = Arc(source=fs.b5.outlet, destination=fs.nh3p1.inlet)
    fs.a03a = Arc(source=fs.nh3p1.outlet, destination=fs.nh3sp1.inlet)
    fs.a04a = Arc(source=fs.nh3sp1.s19, destination=fs.nh3p2.inlet)
    fs.a05a = Arc(source=fs.nh3p2.outlet, destination=fs.b6.inlet)
    fs.a06a = Arc(source=fs.b6.outlet, destination=fs.m01.s6)
    fs.a07a = Arc(source=fs.co2_feed.outlet, destination=fs.m01.co2)
    fs.a08a = Arc(source=fs.b13.outlet, destination=fs.m01.s31)
    fs.a09a = Arc(source=fs.m01.outlet, destination=fs.b27.inlet)
    fs.a10a = Arc(source=fs.b27.outlet, destination=fs.r01.inlet)
    fs.a11a = Arc(source=fs.r01.outlet, destination=fs.b2.inlet)
    fs.a12a = Arc(source=fs.b2.outlet, destination=fs.f01.inlet)
    # Flash2 phase routing note: the .bkp BLOCK records' M-codes are
    # graphics connection indices, not phase tags. Phase assignment below
    # follows the process logic (confirmed consistent with the RadFrac
    # PRODUCTS records, the S5 design-spec stream being 60 wt% urea
    # liquor, and B13 being a Pump): the urea-bearing LIQUID runs
    # F01 -> A02 -> B1 -> B22 -> F012 -> A022 -> B8 -> X3 -> B4 -> P1,
    # while the gas branches feed the condensation/recovery units.
    fs.a13a = Arc(source=fs.f01.liq_outlet, destination=fs.a02.inlet)
    fs.a14a = Arc(source=fs.a02.outlet, destination=fs.b1.inlet)
    fs.a15a = Arc(source=fs.b1.vap_outlet, destination=fs.a01.inlet)
    fs.a16a = Arc(source=fs.a01.outlet, destination=fs.b11.s23)
    fs.a17a = Arc(source=fs.f01.vap_outlet, destination=fs.b11.s24)
    fs.a18a = Arc(source=fs.b11.outlet, destination=fs.b15.s13)
    fs.a19a = Arc(source=b7_ports["s34_out"], destination=fs.b15.s34)
    fs.a20a = Arc(source=fs.b15.outlet, destination=fs.b10h.inlet)
    fs.a21a = Arc(source=fs.b10h.outlet, destination=fs.b12.inlet)
    fs.a22a = Arc(source=fs.b12.liq_outlet, destination=fs.b9.inlet)
    fs.a23a = Arc(source=fs.b9.rec1, destination=fs.b13.inlet)
    fs.a24a = Arc(source=fs.b9.purge1, destination=fs.purge1_prod.inlet)
    fs.a25a = Arc(source=fs.b12.vap_outlet, destination=b7_ports["s32_in"])
    fs.a26a = Arc(source=fs.nh3sp1.s17, destination=b7_ports["s17_in"])
    fs.a27a = Arc(source=b7_ports["s27_out"], destination=fs.b19.inlet)
    fs.a28a = Arc(source=fs.b19.rec2, destination=fs.b16h.inlet)
    fs.a29a = Arc(source=fs.b19.purge2, destination=fs.purge2_prod.inlet)
    fs.a30a = Arc(source=fs.b16h.outlet, destination=fs.b17.inlet)
    fs.a31a = Arc(source=fs.b17.liq_outlet, destination=fs.b5.s41)
    fs.a32a = Arc(source=fs.b17.vap_outlet, destination=fs.s42_prod.inlet)
    fs.a33a = Arc(source=fs.b1.liq_outlet, destination=fs.b22.inlet)
    fs.a34a = Arc(source=fs.b22.outlet, destination=fs.f012.inlet)
    fs.a35a = Arc(source=fs.f012.liq_outlet, destination=fs.a022.inlet)
    fs.a36a = Arc(source=fs.a022.outlet, destination=fs.carbmix2.s11)
    fs.a37a = Arc(source=fs.f012.vap_outlet, destination=fs.carbmix2.s2)
    fs.a38a = Arc(source=fs.carbmix2.outlet, destination=fs.b8.inlet)
    fs.a39a = Arc(source=fs.b8.vap_outlet, destination=fs.a012.inlet)
    fs.a40a = Arc(source=fs.a012.outlet, destination=fs.b3.inlet)
    fs.a41a = Arc(source=fs.b3.outlet, destination=fs.b18.inlet)
    fs.a42a = Arc(source=fs.b18.liq_outlet, destination=fs.b14.nh3out)
    fs.a43a = Arc(source=fs.b18.vap_outlet, destination=fs.b24.s54)
    fs.a44a = Arc(source=fs.b8.liq_outlet, destination=fs.x3.inlet)
    fs.a45a = Arc(source=fs.x3.outlet, destination=fs.b4.inlet)
    fs.a46a = Arc(source=fs.b4.vap_outlet, destination=fs.a013.inlet)
    fs.a47a = Arc(source=fs.a013.outlet, destination=fs.b24.s55)
    fs.a48a = Arc(source=fs.b24.outlet, destination=fs.b20h.inlet)
    fs.a49a = Arc(source=fs.b20h.outlet, destination=fs.b14.nh3out3)
    fs.a50a = Arc(source=fs.b4.liq_outlet, destination=fs.p1.inlet)
    fs.a51a = Arc(source=fs.p1.outlet, destination=b25_ports["s5_in"])
    fs.a52a = Arc(source=fs.steam_feed.outlet,
                  destination=b25_ports["steam_in"])
    fs.a53a = Arc(source=b25_ports["s16_out"], destination=fs.b23.inlet)
    fs.a54a = Arc(source=fs.b23.outlet, destination=fs.b14.nh3out2)
    fs.a55a = Arc(source=fs.b14.outlet, destination=fs.b26.inlet)
    fs.a56a = Arc(source=fs.b26.rec3, destination=fs.m01.rec3)
    fs.a57a = Arc(source=fs.b26.purge3, destination=fs.purge3_prod.inlet)
    fs.a58a = Arc(source=b25_ports["urea_out"],
                  destination=b28_ports["feed_in"])
    fs.a59a = Arc(source=b28_ports["def_out"], destination=fs.def_prod.inlet)
    fs.a60a = Arc(source=b28_ports["s30_out"], destination=fs.s30_prod.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # ---------------- feed specifications ----------------
    _set_state(fs.nh3_feed.outlet,
               {"NH3": fnh3_kgph * _KGPH / MW["NH3"]},
               T_NH3_FEED, P_NH3_FEED)
    _set_state(fs.co2_feed.outlet,
               {"CO2": fco2_kgph * _KGPH / MW["CO2"]},
               T_CO2_FEED, P_SYN)
    _set_state(fs.steam_feed.outlet,
               {"H2O": STEAM_KGPH * _KGPH / MW["H2O"]},
               T_STEAM, P_B25)

    # ---------------- report expressions ----------------
    def _mass_kgph(state):
        return (state.flow_mol
                * sum(state.mole_frac_comp[c] * MW[c] for c in COMPS)
                * 3600.0 * pyunits.s / pyunits.hr)

    # widen the SmoothVLE equilibrium-temperature smoothing: every flash
    # liquid outlet feeds a downstream inlet sitting exactly ON its bubble
    # point, i.e. inside the smooth_max kink, and the default eps of
    # 0.01 K leaves that kink sharp enough to stall IPOPT there
    from pyomo.environ import Block as _Block
    for blk in m.component_data_objects(_Block, descend_into=True):
        if hasattr(blk, "eps_1_Vap_Liq"):
            blk.eps_1_Vap_Liq = 1.0

    dp = fs.def_prod.properties[0]
    fs.ft_def_out = Expression(expr=_mass_kgph(dp),
                               doc="DEF-OUT total mass flow [kg/h] "
                                   "(Aspen Ft_UREA-OUT)")
    fs.w_urea = Expression(
        expr=dp.mole_frac_comp["UREA"] * MW["UREA"]
        / sum(dp.mole_frac_comp[c] * MW[c] for c in COMPS),
        doc="DEF-OUT urea mass fraction (Aspen Wurea)")
    fs.q_r01 = Expression(expr=fs.r01.heat_duty[0], doc="QR01 [W]")
    fs.q_b3 = Expression(expr=fs.b3.heat_duty[0], doc="QB3 [W]")
    fs.q_b6 = Expression(expr=fs.b6.heat_duty[0], doc="QB6 [W]")
    fs.q_b27 = Expression(expr=fs.b27.heat_duty[0], doc="QB27 [W]")
    fs.q_b10_hot = Expression(expr=fs.b10h.heat_duty[0], doc="QB10_hot [W]")

    return m


def apply_design_spec(m, target_w_urea=0.60):
    """Aspen design spec D1: free the NH3SP1 split fraction (0.1-0.9) and
    require 60 wt% urea in S5 (the P1 discharge). Activate only after the
    plain simulation has converged."""
    fs = m.fs
    sf = fs.nh3sp1.split_fraction[0, "s17"]
    sf.unfix()
    sf.setlb(0.1)
    sf.setub(0.9)
    s5 = fs.p1.control_volume.properties_out[0]
    fs.d1_spec = Constraint(
        expr=s5.mole_frac_comp["UREA"] * MW["UREA"]
        == target_w_urea * sum(s5.mole_frac_comp[c] * MW[c] for c in COMPS)
    )


# ---------------------------------------------------------------------------
# Initialization: forward passes with guessed tears, then simultaneous solve
# ---------------------------------------------------------------------------

def initialize(m, passes=9, tee=False):
    fs = m.fs
    solver = get_solver("ipopt", options=dict(_OPTARG))

    for f in (fs.nh3_feed, fs.co2_feed, fs.steam_feed):
        f.initialize(outlvl=50, optarg=_OPTARG)

    nh3_mol = value(fs.nh3_feed.outlet.flow_mol[0])
    co2_mol = value(fs.co2_feed.outlet.flow_mol[0])

    # tear guesses (all four loops), grid-scale
    _guess_state(fs.b5.s41_state[0],  # S41: NH3 vapor recycle
                 {"NH3": 0.3 * nh3_mol, "CO2": 0.05 * co2_mol,
                  "H2O": 0.02 * nh3_mol},
                 T_HX_HOT_OUT, P_ATM)
    _guess_state(fs.b13.control_volume.properties_in[0],  # REC1 gas
                 {"NH3": 0.2 * nh3_mol, "CO2": 0.2 * co2_mol,
                  "H2O": 0.05 * nh3_mol},
                 T_B12, P_MP)
    _guess_state(fs.m01.rec3_state[0],  # REC3 recovered gas
                 {"NH3": 0.4 * nh3_mol, "CO2": 0.3 * co2_mol,
                  "H2O": 0.2 * nh3_mol},
                 T_HX_HOT_OUT, P_ATM)
    _guess_state(fs.b15.s34_state[0],  # S34 column bottoms
                 {"H2O": 0.3 * nh3_mol, "NH3": 0.05 * nh3_mol,
                  "CARB": 0.05 * co2_mol},
                 T_B12, P_MP)

    def _seed_outlet(unit):
        """Seed properties_out with the inlet composition at the unit's
        own fixed T/P spec: the near-total-condensation states several
        coolers land on stall the VLE solve from an unseeded start."""
        cv = getattr(unit, "control_volume", None)
        if cv is None or not hasattr(cv, "properties_in"):
            return
        st_in = cv.properties_in[0]
        st_out = cv.properties_out[0]
        if st_in.flow_mol.value is None:
            return
        fin = {c: max(value(st_in.flow_mol * st_in.mole_frac_comp[c]), EPS)
               for c in COMPS}
        t_out = (st_out.temperature.value if st_out.temperature.fixed
                 else value(st_in.temperature))
        p_out = (st_out.pressure.value if st_out.pressure.fixed
                 else value(st_in.pressure))
        _guess_state(st_out, fin, t_out, p_out)

    def fwd(unit, *arcs, extent=None):
        for a in arcs:
            _prop(a)
        if extent is None:
            _seed_outlet(unit)
        if extent is not None:
            ext = unit.control_volume.rate_reaction_extent[0, "CARBF"]
            con = None
            for c in ("decomp_extent", "carbeq_conv"):
                if hasattr(unit, c):
                    con = getattr(unit, c)
            con.deactivate()
            e_target = extent()
            st_in = unit.control_volume.properties_in[0]
            st_out = unit.control_volume.properties_out[0]
            fin = {c: value(st_in.flow_mol * st_in.mole_frac_comp[c])
                   for c in COMPS}
            t_out = (st_out.temperature.value
                     if st_out.temperature.fixed else value(st_in.temperature))
            p_out = (st_out.pressure.value
                     if st_out.pressure.fixed else value(st_in.pressure))
            # extent continuation: near-complete decomposition leaves a
            # tiny urea-melt liquid pool the solver can transiently kill
            # from a cold start (restoration failure); walking the extent
            # up keeps every intermediate state benign. Outlet is seeded
            # (set, not fixed) with the shifted inlet at each step.
            steps = (0.5, 0.9, 1.0)
            for frac in steps:
                e = frac * e_target
                ext.fix(e)
                shift = {"NH3": -2 * e, "CO2": -e, "CARB": e}
                guess = {c: max(fin[c] + shift.get(c, 0.0), EPS)
                         for c in COMPS}
                _guess_state(st_out, guess, t_out, p_out)
                try:
                    unit.initialize(outlvl=50, optarg=_OPTARG)
                except Exception:
                    # intermediate continuation states may themselves sit
                    # on awkward phase boundaries; only the final extent
                    # must initialize
                    if frac == steps[-1]:
                        raise
            ext.unfix()
            con.activate()
        else:
            unit.initialize(outlvl=50, optarg=_OPTARG)

    # --- Aitken acceleration of the four tear streams -------------------
    # The gas loops recycle 90-95% of their streams, so plain sequential
    # substitution contracts by only ~0.05-0.1 per pass (Aspen itself
    # needed Broyden here). Aitken's delta-squared extrapolation on the
    # tear component flows every third pass removes most of that.
    tear_states = {
        "rec1": fs.b13.control_volume.properties_in[0],
        "rec3": fs.m01.rec3_state[0],
        "s41": fs.b5.s41_state[0],
        "s34": fs.b15.s34_state[0],
    }
    tear_hist = {k: [] for k in tear_states}

    def _tear_vector(st):
        return [value(st.flow_mol * st.mole_frac_comp[c]) for c in COMPS]

    def _aitken(name, st):
        hist = tear_hist[name]
        hist.append(_tear_vector(st))
        if len(hist) < 3:
            return
        x0, x1, x2 = hist[-3], hist[-2], hist[-1]
        new = []
        for a, b, c in zip(x0, x1, x2):
            d1, d2 = b - a, c - b
            denom = d2 - d1
            if abs(denom) > 1e-12 and abs(d2) > 1e-12:
                cand = c - d2 * d2 / denom
                # accept sane extrapolations, capped at 3x so downstream
                # units see bounded feed changes between passes
                if cand >= 0.0:
                    new.append(min(cand, 3.0 * max(c, 1e-6)))
                    continue
            new.append(c)
        flows = {comp: max(f, 0.0) for comp, f in zip(COMPS, new)}
        _guess_state(st, flows, value(st.temperature), value(st.pressure))
        hist.clear()

    def carb_in(unit):
        st = unit.control_volume.properties_in[0]
        return -DECOMP_CONV * value(st.flow_mol * st.mole_frac_comp["CARB"])

    def cond_in(unit):
        st = unit.control_volume.properties_in[0]
        return 0.99 * min(
            value(st.flow_mol * st.mole_frac_comp["CO2"]),
            value(st.flow_mol * st.mole_frac_comp["NH3"]) / 2.0,
        )

    for it in range(passes):
        # HP loop
        fwd(fs.b5, fs.a01a)
        fwd(fs.nh3p1, fs.a02a)
        fwd(fs.nh3sp1, fs.a03a)
        fwd(fs.nh3p2, fs.a04a)
        fwd(fs.b6, fs.a05a)
        fwd(fs.b13)  # inlet = REC1 tear guess / previous pass
        fwd(fs.m01, fs.a06a, fs.a07a, fs.a08a)
        fwd(fs.b27, fs.a09a)
        _prop(fs.a10a)
        if it == 0:
            init_cstr_ramped(fs.r01)
        else:
            # warm start from the previous pass; fall back to the volume
            # homotopy, and if even that fails carry the previous pass's
            # outlet forward (the simultaneous solve reconciles it)
            try:
                fs.r01.initialize(outlvl=50, optarg=_OPTARG)
            except Exception:
                try:
                    init_cstr_ramped(fs.r01)
                except Exception:
                    print(f"  (pass {it + 1}: R01 re-init failed, "
                          "carrying previous outlet)")
        # MP train
        fwd(fs.b2, fs.a11a)
        fwd(fs.f01, fs.a12a)
        fwd(fs.a02, fs.a13a, extent=lambda: carb_in(fs.a02))
        fwd(fs.b1, fs.a14a)
        fwd(fs.a01, fs.a15a, extent=lambda: cond_in(fs.a01))
        fwd(fs.b11, fs.a16a, fs.a17a)
        fwd(fs.b15, fs.a18a)  # s34 from tear guess / previous pass
        fwd(fs.b10h, fs.a20a)
        fwd(fs.b12, fs.a21a)
        fwd(fs.b9, fs.a22a)
        _prop(fs.a23a)  # REC1 tear update
        fwd(fs.b7_mix, fs.a25a, fs.a26a)
        # the V/F = 0.9 spec over-determines the Flash initializer (it
        # fixes the full outlet state); initialize duty-specified, then
        # restore the V/F spec and polish the block at the true spec
        # (otherwise the simultaneous solve starts at the adiabatic V/F
        # and stalls on the way to 0.9)
        vf = fs.b7.control_volume.properties_out[0].phase_frac["Vap"]
        vf.unfix()
        fs.b7.heat_duty.fix(0.0)
        fwd(fs.b7, fs.a_b7)
        fs.b7.heat_duty.unfix()
        vf.fix(B7_D_F)
        inlet_vars = [fs.b7.inlet.flow_mol[0],
                      fs.b7.inlet.temperature[0], fs.b7.inlet.pressure[0]]
        inlet_vars += [fs.b7.inlet.mole_frac_comp[0, c] for c in COMPS]
        flags = [v.fixed for v in inlet_vars]
        for v in inlet_vars:
            v.fix()
        try:
            solver.solve(fs.b7)
        except Exception:
            pass  # the simultaneous solve gets another chance
        for v, f in zip(inlet_vars, flags):
            if not f:
                v.unfix()
        _prop(fs.a19a)  # S34 tear update
        fwd(fs.b19, fs.a27a)
        fwd(fs.b16h, fs.a28a)
        fwd(fs.b17, fs.a30a)
        _prop(fs.a31a)  # S41 tear update
        # LP train
        fwd(fs.b22, fs.a33a)
        fwd(fs.f012, fs.a34a)
        fwd(fs.a022, fs.a35a, extent=lambda: carb_in(fs.a022))
        fwd(fs.carbmix2, fs.a36a, fs.a37a)
        fwd(fs.b8, fs.a38a)
        fwd(fs.a012, fs.a39a, extent=lambda: cond_in(fs.a012))
        fwd(fs.b3, fs.a40a)
        fwd(fs.b18, fs.a41a)
        # finishing
        fwd(fs.x3, fs.a44a)
        fwd(fs.b4, fs.a45a)
        fwd(fs.a013, fs.a46a, extent=lambda: cond_in(fs.a013))
        fwd(fs.b24, fs.a43a, fs.a47a)
        fwd(fs.b20h, fs.a48a)
        fwd(fs.p1, fs.a50a)
        fwd(fs.b25_mix, fs.a51a, fs.a52a)
        fwd(fs.b25, fs.a_b25)
        fwd(fs.b23, fs.a53a)
        fwd(fs.b14, fs.a42a, fs.a49a, fs.a54a)
        fwd(fs.b26, fs.a55a)
        _prop(fs.a56a)  # REC3 tear update
        for name, st in tear_states.items():
            _aitken(name, st)
        rec3 = value(fs.m01.rec3_state[0].flow_mol)
        print(f"  tear pass {it + 1}: REC3 = {rec3:.4f} mol/s")

    # column B28 + products (feed-forward only). The B28 splitter's split
    # fractions are free variables closed by the spillover constraints;
    # fix them at values computed from the inlet for the initializer,
    # then release.
    _prop(fs.a58a)
    st = fs.b28.mixed_state[0]
    f_tot = value(st.flow_mol)
    f_light = sum(value(st.flow_mol * st.mole_frac_comp[c])
                  for c in ("NH3", "CO2", "H2O"))
    f_heavy = max(f_tot - f_light, 1e-10)
    d_light = min(f_light, B28_D_F * f_tot)
    sf_l = d_light / max(f_light, 1e-10)
    sf_h = max(B28_D_F * f_tot - d_light, 0.0) / f_heavy
    for con in (fs.b28.eq_light_a, fs.b28.eq_light_b, fs.b28.eq_light_bal,
                fs.b28.eq_heavy_a, fs.b28.eq_heavy_bal):
        con.deactivate()
    for c in ("NH3", "CO2", "H2O"):
        fs.b28.split_fraction[0, "dist", c].fix(min(sf_l, 1 - EPS))
    for c in ("UREA", "CARB"):
        fs.b28.split_fraction[0, "dist", c].fix(min(sf_h, 1 - EPS))
    fs.b28.initialize(outlvl=50, optarg=_OPTARG)
    for c in COMPS:
        fs.b28.split_fraction[0, "dist", c].unfix()
    for con in (fs.b28.eq_light_a, fs.b28.eq_light_b, fs.b28.eq_light_bal,
                fs.b28.eq_heavy_a, fs.b28.eq_heavy_bal):
        con.activate()
    for arc, prod in ((fs.a24a, fs.purge1_prod), (fs.a29a, fs.purge2_prod),
                      (fs.a57a, fs.purge3_prod), (fs.a32a, fs.s42_prod),
                      (fs.a59a, fs.def_prod), (fs.a60a, fs.s30_prod)):
        _prop(arc)
        prod.initialize(outlvl=50, optarg=_OPTARG)

    return solver


def solve(m, solver=None, tee=True):
    if solver is None:
        solver = get_solver("ipopt", options=dict(_OPTARG))
    dof = degrees_of_freedom(m)
    print(f"degrees of freedom before solve: {dof}")
    assert dof == 0, "flowsheet is not square"
    results = solver.solve(m, tee=tee)
    if str(results.solver.termination_condition) != "optimal":
        import logging
        from pyomo.util.infeasible import log_infeasible_constraints
        logging.basicConfig(level=logging.INFO)
        print("--- infeasible constraints at final point (tol 1e-4) ---")
        log_infeasible_constraints(m, log_expression=False,
                                   log_variables=False, tol=1e-4)
    return results


def _aspen_row(fnh3, fco2):
    import csv
    csv_path = (Path(__file__).resolve().parents[1]
                / "aspen" / "data" / "ureaF_results_live.csv")
    if not csv_path.exists():
        return None
    best, best_d = None, 1e9
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("run_ok") != "1":
                continue
            try:
                d = abs(float(r["Fnh3"]) - fnh3) + abs(float(r["Fco2"]) - fco2)
            except (KeyError, ValueError):
                continue
            if d < best_d:
                best, best_d = r, d
    return best if best_d < 0.5 else None


def report(m, fnh3=10.0, fco2=15.0):
    fs = m.fs
    print("\n================ ureaF IDAES results ================")
    print(f"DEF-OUT mass flow    : {value(fs.ft_def_out):10.3f} kg/h")
    print(f"DEF-OUT urea massfrac: {value(fs.w_urea):10.4f}")
    for label, expr in [("QR01", fs.q_r01), ("QB3", fs.q_b3),
                        ("QB6", fs.q_b6), ("QB27", fs.q_b27),
                        ("QB10_hot", fs.q_b10_hot)]:
        w = value(expr)
        print(f"{label:10s}: {w / 1e3:10.3f} kW = {w * W_TO_MMKCAL_H:+.6f} MMkcal/h")

    print("\nproduct streams (mass kg/h; urea kg/h):")
    for label, prod in [("DEF-OUT", fs.def_prod), ("S30", fs.s30_prod),
                        ("PURGE1", fs.purge1_prod), ("PURGE2", fs.purge2_prod),
                        ("PURGE3", fs.purge3_prod), ("S42", fs.s42_prod)]:
        st = prod.properties[0]
        mtot = value(st.flow_mol * sum(st.mole_frac_comp[c] * MW[c]
                                       for c in COMPS)) * 3600.0
        murea = value(st.flow_mol * st.mole_frac_comp["UREA"]) \
            * MW["UREA"] * 3600.0
        print(f"  {label:8s}: {mtot:9.3f}  {murea:9.4f}")

    row = _aspen_row(fnh3, fco2)
    if row is None:
        print("\n(no matching Aspen row for validation)")
        return
    print(f"\n===== vs Aspen (Fnh3={float(row['Fnh3']):.3f}, "
          f"Fco2={float(row['Fco2']):.3f}) =====")

    def line(label, idaes_val, aspen_val, unit):
        dev = ((idaes_val - aspen_val) / abs(aspen_val)
               if aspen_val else float("nan"))
        print(f"{label:22s}{idaes_val:14.4f}{float(aspen_val):14.4f}"
              f"{dev:+9.1%} {unit}")

    line("Ft_UREA-OUT", value(fs.ft_def_out), float(row["Ft_UREA-OUT"]), "kg/h")
    line("Wurea", value(fs.w_urea), float(row["Wurea"]), "-")
    line("QR01", value(fs.q_r01) * W_TO_MMKCAL_H, float(row["QR01"]),
         "MMkcal/h")
    line("QB3", value(fs.q_b3) * W_TO_MMKCAL_H, float(row["QB3"]),
         "MMkcal/h")
    line("QB6", value(fs.q_b6) * W_TO_MMKCAL_H, float(row["QB6"]),
         "MMkcal/h")
    line("QB27", value(fs.q_b27) * W_TO_MMKCAL_H, float(row["QB27"]),
         "MMkcal/h")


if __name__ == "__main__":
    fnh3, fco2 = 10.0, 15.0
    if len(sys.argv) == 3:
        fnh3, fco2 = float(sys.argv[1]), float(sys.argv[2])
    m = build_model(fnh3_kgph=fnh3, fco2_kgph=fco2)
    print("model built; initializing...")
    solver = initialize(m)
    results = solve(m, solver)
    report(m, fnh3=fnh3, fco2=fco2)

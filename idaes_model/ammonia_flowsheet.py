"""
IDAES replica of the ammoniaF (nitrate-hydrogenation) Aspen Plus flowsheet.

Topology, operating specifications, and reactor sizing were extracted
directly from ``aspen/ammoniaF.bkp`` (plain-text Aspen backup); kinetics come
from the manuscript SI (``personal/SI-1.pdf``, Table S7). Aspen block ->
IDAES unit mapping:

    MCOMPR1 (4-stage isentropic, eta_s=0.72, intercooled to 50 degC,
             discharge 27 bar)          -> cmp1..cmp4 + ic1..ic3
    B6      (Sep: brine pre-concentrator,
             S16 water capped 1400 kg/h) -> b6 (Separator + flow constraint)
    MIXER1  (27 bar)                     -> mixer1
    H1      (heater, 60 degC, 27 bar)    -> h1
    B1      (RPlug, isothermal at inlet T,
             15 tubes x 10 m x 1 m diam, liquid phase,
             reactions AMMONIA1)         -> b1 (PFR, area = 15*pi/4 m2)
    C1      (cooler, 10 degC, 27 bar)    -> c1
    B2      (Flash2, 10 degC, 27 bar)    -> b2
    SPLIT1  (FSplit, 90% recycle / 10% purge) -> split1
    V1      (valve to 1 bar)             -> v1 (isenthalpic Heater, Q=0)

Streams: H2FEED (pure H2, 25 degC, 1 bar), NO3-IN (nitrate brine, 25 degC,
27 bar), S13 (hydrous ammonia product), PURGE1, H2O-OUT1.

Three property-package variants (see properties/enrtl_ammonia.py) mirror the
Aspen phase specifications; Translator blocks bridge them:

    vapor-only   : H2 compressor train
    liquid-only  : reaction train (eNRTL, matches RPLUG NPHASE=1 PHASE=L)
    two-phase    : separation train (Ideal + Henry VLE)

Run:  python ammonia_flowsheet.py         (defaults: Ft=8500 kg/h, Fh2=10 kg/h)
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
)
from pyomo.network import Arc

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.mixer import MomentumMixingType
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
    Compressor,
    Feed,
    Flash,
    Heater,
    Mixer,
    PFR,
    Product,
    Separator,
    Translator,
)
from idaes.models.unit_models.separator import SplittingType

# swap this import (same get_prop/get_rxn signatures) to change thermo basis
from properties.enrtl_ammonia import get_prop
from reactions.ammonia_reactions import get_rxn

# ---------------------------------------------------------------------------
# Specifications extracted from aspen/ammoniaF.bkp
# ---------------------------------------------------------------------------
P_LOOP = 27e5  # Pa   (27 bar loop pressure)
P_H2_FEED = 1e5  # Pa (H2FEED: 25 degC, 1 bar)
P_PRODUCT = 1e5  # Pa (V1 P-OUT = 1 bar)
T_FEED = 298.15  # K  (both feeds at 25 degC)
T_REACTOR = 333.15  # K (H1 spec / RPLUG inlet-temp isothermal spec, 60 degC)
T_FLASH = 283.15  # K  (C1 / B2 spec, 10 degC)
T_INTERCOOL = 323.15  # K (MCOMPR1 stage coolers, 50 degC)
ETA_ISENTROPIC = 0.72  # MCOMPR1 SEFF per stage
N_STAGES = 4
STAGE_RATIO = (P_LOOP / P_H2_FEED) ** (1.0 / N_STAGES)
RECYCLE_FRAC = 0.9  # SPLIT1: 90% of flash vapor recycled
S16_H2O_KGPH = 1400.0  # B6 water cap on the concentrate stream
REACTOR_LENGTH = 10.0  # m
REACTOR_AREA = 15 * 3.141592653589793 / 4.0  # m2 (15 tubes x 1 m diameter)
PFR_ELEMENTS = 20

MW = {  # kg/mol, must match the property package
    "H2O": 18.0153e-3, "NH3": 17.0305e-3, "H2": 2.01588e-3, "N2": 28.0134e-3,
    "Na+": 22.98977e-3, "Cl-": 35.453e-3, "NO3-": 62.0049e-3,
    "NO2-": 46.0055e-3, "SO4-2": 96.0626e-3, "OH-": 17.0073e-3,
}

# NO3-IN composition at the validation point (aspen/data CSVs, kg/h)
BRINE_KGPH = {
    "H2O": 7621.015, "Cl-": 453.9, "Na+": 330.565,
    "NO3-": 78.625, "SO4-2": 15.895,
}

EPS = 1e-10  # trace mole fraction for species absent from a stream

LIQ_COMPS = ["H2O", "NH3", "H2", "N2",
             "Na+", "Cl-", "NO3-", "NO2-", "SO4-2", "OH-"]
VAP_COMPS = ["H2O", "NH3", "H2", "N2"]
ION_COMPS = ["Na+", "Cl-", "NO3-", "NO2-", "SO4-2", "OH-"]

W_TO_MMKCAL_H = 1.0 / 1.163e6  # 1 MMkcal/h = 1.163 MW


# ---------------------------------------------------------------------------
# Translator helpers
# ---------------------------------------------------------------------------

def _translator_constraints(tr, in_comps, out_comps):
    """Equate T, P and total component molar flows across a Translator.

    Components present only on the outlet side are pinned to a trace mole
    fraction (the eNRTL/Ideal formulations need strictly positive fractions).
    The outlet state is built with a sum(mole_frac)=1 constraint
    (outlet_state_defined=False), which closes the flow balance.
    """
    t0 = tr.flowsheet().time.first()
    blk_in = tr.properties_in[t0]
    blk_out = tr.properties_out[t0]

    shared = [c for c in out_comps if c in in_comps]
    only_out = [c for c in out_comps if c not in in_comps]

    @tr.Constraint(shared)
    def eq_flow_comp(b, j):
        return (b.properties_out[t0].flow_mol
                * b.properties_out[t0].mole_frac_comp[j]
                == b.properties_in[t0].flow_mol
                * b.properties_in[t0].mole_frac_comp[j])

    if only_out:
        @tr.Constraint(only_out)
        def eq_trace_comp(b, j):
            return b.properties_out[t0].mole_frac_comp[j] == EPS

    tr.eq_temperature = Constraint(
        expr=blk_out.temperature == blk_in.temperature)
    tr.eq_pressure = Constraint(expr=blk_out.pressure == blk_in.pressure)


def _set_state(port_or_state, comp_mol_flows, T, P, comps):
    """Fix an FTPx state (Feed port) from component molar flows [mol/s]."""
    total = sum(max(f, 0.0) for f in comp_mol_flows.values())
    port_or_state.flow_mol[0].fix(total)
    for c in comps:
        frac = max(comp_mol_flows.get(c, 0.0), EPS * total) / total
        port_or_state.mole_frac_comp[0, c].fix(frac)
    port_or_state.temperature[0].fix(T)
    port_or_state.pressure[0].fix(P)


def _clamp_state(state):
    """Clamp numerically-negative trace mole fractions to EPS after
    propagate_state (complete conversion leaves ~-1e-17 residuals that the
    FTPx initializer rejects)."""
    for j in state.mole_frac_comp:
        v = state.mole_frac_comp[j].value
        if v is not None and v < EPS:
            state.mole_frac_comp[j].set_value(EPS)


def _prop(arc):
    """propagate_state + clamp tiny negative mole fractions at the
    destination port (IPOPT's bound relaxation can leave trace components
    at ~-1e-9, which downstream FTPx initializers reject)."""
    propagate_state(arc)
    port = arc.destination
    if hasattr(port, "mole_frac_comp"):
        for idx in port.mole_frac_comp:
            v = port.mole_frac_comp[idx].value
            if v is not None and v < EPS:
                port.mole_frac_comp[idx].set_value(EPS)


def _guess_state(state, comp_mol_flows, T, P, comps):
    """Set (not fix) initial values on an FTPx state block."""
    total = sum(max(f, 0.0) for f in comp_mol_flows.values())
    state.flow_mol.set_value(total)
    for c in comps:
        frac = max(comp_mol_flows.get(c, 0.0), EPS * total) / total
        state.mole_frac_comp[c].set_value(frac)
    state.temperature.set_value(T)
    state.pressure.set_value(P)


# ---------------------------------------------------------------------------
# Flowsheet builder
# ---------------------------------------------------------------------------

def build_model(fh2_kgph=10.0, brine_kgph=None):
    brine_kgph = dict(brine_kgph or BRINE_KGPH)

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # property/reaction packages
    m.fs.props_vap = GenericParameterBlock(**get_prop(("Vap",)))
    m.fs.props_liq = GenericParameterBlock(**get_prop(("Liq",)))
    m.fs.props_vle = GenericParameterBlock(**get_prop(("Liq", "Vap")))
    m.fs.rxns = GenericReactionParameterBlock(**get_rxn(m.fs.props_liq))

    # ---------------- H2 compressor train (vapor package) ----------------
    m.fs.h2_feed = Feed(property_package=m.fs.props_vap)
    for k in range(1, N_STAGES + 1):
        setattr(m.fs, f"cmp{k}", Compressor(property_package=m.fs.props_vap))
    for k in range(1, N_STAGES):  # coolers after stages 1-3 only
        setattr(m.fs, f"ic{k}", Heater(property_package=m.fs.props_vap))

    m.fs.t_h2 = Translator(
        inlet_property_package=m.fs.props_vap,
        outlet_property_package=m.fs.props_liq,
        outlet_state_defined=False,
    )
    _translator_constraints(m.fs.t_h2, VAP_COMPS, LIQ_COMPS)

    # ---------------- reaction train (liquid eNRTL package) ----------------
    m.fs.no3_feed = Feed(property_package=m.fs.props_liq)
    m.fs.b6 = Separator(
        property_package=m.fs.props_liq,
        outlet_list=["s16", "h2o_out"],
        split_basis=SplittingType.componentFlow,
    )
    m.fs.h2o_out = Product(property_package=m.fs.props_liq)

    m.fs.mixer1 = Mixer(
        property_package=m.fs.props_liq,
        inlet_list=["h2", "recycle", "brine"],
        momentum_mixing_type=MomentumMixingType.none,
    )
    m.fs.h1 = Heater(property_package=m.fs.props_liq)
    m.fs.b1 = PFR(
        property_package=m.fs.props_liq,
        reaction_package=m.fs.rxns,
        has_equilibrium_reactions=False,
        has_heat_of_reaction=False,  # formation enthalpies carry reaction heat
        has_heat_transfer=True,
        has_pressure_change=False,
        transformation_method="dae.finite_difference",
        transformation_scheme="BACKWARD",
        finite_elements=PFR_ELEMENTS,
    )

    m.fs.t_rx = Translator(
        inlet_property_package=m.fs.props_liq,
        outlet_property_package=m.fs.props_vle,
        outlet_state_defined=False,
    )
    _translator_constraints(m.fs.t_rx, LIQ_COMPS, LIQ_COMPS)

    # ---------------- separation train (two-phase package) ----------------
    m.fs.c1 = Heater(property_package=m.fs.props_vle)
    m.fs.b2 = Flash(property_package=m.fs.props_vle)
    m.fs.split1 = Separator(
        property_package=m.fs.props_vle,
        outlet_list=["recycle", "purge"],
        split_basis=SplittingType.totalFlow,
    )
    m.fs.purge = Product(property_package=m.fs.props_vle)
    m.fs.v1 = Heater(property_package=m.fs.props_vle, has_pressure_change=True)
    m.fs.s13 = Product(property_package=m.fs.props_vle)

    m.fs.t_rec = Translator(
        inlet_property_package=m.fs.props_vle,
        outlet_property_package=m.fs.props_liq,
        outlet_state_defined=False,
    )
    _translator_constraints(m.fs.t_rec, LIQ_COMPS, LIQ_COMPS)

    # ---------------- arcs ----------------
    m.fs.a01 = Arc(source=m.fs.h2_feed.outlet, destination=m.fs.cmp1.inlet)
    m.fs.a02 = Arc(source=m.fs.cmp1.outlet, destination=m.fs.ic1.inlet)
    m.fs.a03 = Arc(source=m.fs.ic1.outlet, destination=m.fs.cmp2.inlet)
    m.fs.a04 = Arc(source=m.fs.cmp2.outlet, destination=m.fs.ic2.inlet)
    m.fs.a05 = Arc(source=m.fs.ic2.outlet, destination=m.fs.cmp3.inlet)
    m.fs.a06 = Arc(source=m.fs.cmp3.outlet, destination=m.fs.ic3.inlet)
    m.fs.a07 = Arc(source=m.fs.ic3.outlet, destination=m.fs.cmp4.inlet)
    m.fs.a08 = Arc(source=m.fs.cmp4.outlet, destination=m.fs.t_h2.inlet)
    m.fs.a09 = Arc(source=m.fs.t_h2.outlet, destination=m.fs.mixer1.h2)
    m.fs.a10 = Arc(source=m.fs.no3_feed.outlet, destination=m.fs.b6.inlet)
    m.fs.a11 = Arc(source=m.fs.b6.s16, destination=m.fs.mixer1.brine)
    m.fs.a12 = Arc(source=m.fs.b6.h2o_out, destination=m.fs.h2o_out.inlet)
    m.fs.a13 = Arc(source=m.fs.mixer1.outlet, destination=m.fs.h1.inlet)
    m.fs.a14 = Arc(source=m.fs.h1.outlet, destination=m.fs.b1.inlet)
    m.fs.a15 = Arc(source=m.fs.b1.outlet, destination=m.fs.t_rx.inlet)
    m.fs.a16 = Arc(source=m.fs.t_rx.outlet, destination=m.fs.c1.inlet)
    m.fs.a17 = Arc(source=m.fs.c1.outlet, destination=m.fs.b2.inlet)
    m.fs.a18 = Arc(source=m.fs.b2.vap_outlet, destination=m.fs.split1.inlet)
    m.fs.a19 = Arc(source=m.fs.split1.recycle, destination=m.fs.t_rec.inlet)
    m.fs.a20 = Arc(source=m.fs.t_rec.outlet, destination=m.fs.mixer1.recycle)
    m.fs.a21 = Arc(source=m.fs.split1.purge, destination=m.fs.purge.inlet)
    m.fs.a22 = Arc(source=m.fs.b2.liq_outlet, destination=m.fs.v1.inlet)
    m.fs.a23 = Arc(source=m.fs.v1.outlet, destination=m.fs.s13.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # ---------------- specifications ----------------
    # H2FEED: pure H2, 25 degC, 1 bar
    h2_mol = fh2_kgph / 3600.0 / MW["H2"]
    _set_state(m.fs.h2_feed.outlet,
               {"H2": h2_mol}, T_FEED, P_H2_FEED, VAP_COMPS)

    # compressor stages
    for k in range(1, N_STAGES + 1):
        cmp = getattr(m.fs, f"cmp{k}")
        cmp.ratioP.fix(STAGE_RATIO)
        cmp.efficiency_isentropic.fix(ETA_ISENTROPIC)
    for k in range(1, N_STAGES):
        ic = getattr(m.fs, f"ic{k}")
        ic.control_volume.properties_out[0].temperature.fix(T_INTERCOOL)

    # NO3-IN brine feed
    brine_mol = {c: kg / 3600.0 / MW[c] for c, kg in brine_kgph.items()}
    _set_state(m.fs.no3_feed.outlet, brine_mol, T_FEED, P_LOOP, LIQ_COMPS)

    # B6: everything except excess water to the concentrate s16
    for c in LIQ_COMPS:
        if c != "H2O":
            m.fs.b6.split_fraction[0, "s16", c].fix(1.0 - 1e-9)
    # water split left free; cap s16 water mass flow at 1400 kg/h
    s16_h2o_mol = S16_H2O_KGPH / 3600.0 / MW["H2O"]
    m.fs.b6.s16_h2o_cap = Constraint(
        expr=m.fs.b6.s16_state[0].flow_mol
        * m.fs.b6.s16_state[0].mole_frac_comp["H2O"]
        == s16_h2o_mol * pyunits.mol / pyunits.s
    )

    # MIXER1 outlet pressure (Aspen PARAM PRES = 27 bar); with
    # MomentumMixingType.none this is the single pressure spec for the loop
    # (equality mixing would be redundant around the recycle).
    m.fs.mixer1.mixed_state[0].pressure.fix(P_LOOP)

    # H1: heat mixed feed to reactor temperature
    m.fs.h1.control_volume.properties_out[0].temperature.fix(T_REACTOR)

    # B1: geometry + isothermal profile at the inlet temperature
    m.fs.b1.control_volume.area.fix(REACTOR_AREA)
    m.fs.b1.control_volume.length.fix(REACTOR_LENGTH)
    for x in m.fs.b1.control_volume.length_domain:
        if x != m.fs.b1.control_volume.length_domain.first():
            m.fs.b1.control_volume.properties[0, x].temperature.fix(T_REACTOR)

    # C1 + B2: cool and flash at 10 degC, 27 bar
    m.fs.c1.control_volume.properties_out[0].temperature.fix(T_FLASH)
    m.fs.b2.heat_duty.fix(0.0)
    m.fs.b2.deltaP.fix(0.0)

    # SPLIT1: 90% recycle
    m.fs.split1.split_fraction[0, "recycle"].fix(RECYCLE_FRAC)

    # V1: isenthalpic let-down to 1 bar
    m.fs.v1.heat_duty.fix(0.0)
    m.fs.v1.control_volume.properties_out[0].pressure.fix(P_PRODUCT)

    # ---------------- report expressions ----------------
    cv = m.fs.b1.control_volume
    dx = REACTOR_LENGTH / PFR_ELEMENTS
    xs = [x for x in cv.length_domain if x != cv.length_domain.first()]
    m.fs.Qr1 = Expression(
        expr=sum(cv.heat[0, x] for x in xs) * dx,
        doc="Total reactor heat duty [W] (isothermal profile)")
    m.fs.Qh1 = Expression(expr=m.fs.h1.heat_duty[0])
    m.fs.Qc1 = Expression(expr=m.fs.c1.heat_duty[0])
    m.fs.W_comp = Expression(
        expr=sum(getattr(m.fs, f"cmp{k}").work_mechanical[0]
                 for k in range(1, N_STAGES + 1)),
        doc="Total compression work [W]")
    # Aspen's Qcomp output is MCOMPR1 QNET, i.e. the net (intercooler) heat
    # duty of the multistage compressor, NOT the shaft work.
    m.fs.Qcomp = Expression(
        expr=sum(getattr(m.fs, f"ic{k}").heat_duty[0]
                 for k in range(1, N_STAGES)),
        doc="MCOMPR1 net heat duty [W] (intercoolers, matches Aspen QNET)")

    s13 = m.fs.s13.properties[0]
    m.fs.nh3_product_kgph = Expression(
        expr=s13.flow_mol * s13.mole_frac_comp["NH3"] * MW["NH3"]
        * 3600.0 * pyunits.s / pyunits.hr * pyunits.kg / pyunits.mol)
    m.fs.h2o_product_kgph = Expression(
        expr=s13.flow_mol * s13.mole_frac_comp["H2O"] * MW["H2O"]
        * 3600.0 * pyunits.s / pyunits.hr * pyunits.kg / pyunits.mol)
    m.fs.s13_mass_kgph = Expression(
        expr=s13.flow_mol * sum(s13.mole_frac_comp[c] * MW[c]
                                for c in LIQ_COMPS)
        * 3600.0 * pyunits.s / pyunits.hr * pyunits.kg / pyunits.mol,
        doc="S13 total mass flow (Aspen Mt)")

    return m


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize(m, recycle_guess=None):
    solver = get_solver("ipopt", options={"tol": 1e-6, "max_iter": 300})

    # H2 train
    m.fs.h2_feed.initialize()
    prev = m.fs.h2_feed
    seq = ["cmp1", "ic1", "cmp2", "ic2", "cmp3", "ic3", "cmp4"]
    arcs = ["a01", "a02", "a03", "a04", "a05", "a06", "a07"]
    for unit_name, arc_name in zip(seq, arcs):
        propagate_state(getattr(m.fs, arc_name))
        getattr(m.fs, unit_name).initialize()
    _prop(m.fs.a08)
    _clamp_state(m.fs.t_h2.properties_out[0])
    m.fs.t_h2.initialize()

    # brine feed and pre-concentrator
    m.fs.no3_feed.initialize()
    _prop(m.fs.a10)
    # good starting point for the free water split fraction
    h2o_in = value(m.fs.no3_feed.outlet.flow_mol[0]
                   * m.fs.no3_feed.outlet.mole_frac_comp[0, "H2O"])
    s16_h2o_mol = S16_H2O_KGPH / 3600.0 / MW["H2O"]
    m.fs.b6.split_fraction[0, "s16", "H2O"].set_value(s16_h2o_mol / h2o_in)
    m.fs.b6.initialize()
    _prop(m.fs.a12)
    m.fs.h2o_out.initialize()

    # recycle guess (vapor from the flash, mostly H2)
    h2_mol = value(m.fs.h2_feed.outlet.flow_mol[0])
    if recycle_guess is None:
        recycle_guess = {"H2": 0.4 * h2_mol, "N2": 0.02 * h2_mol,
                         "H2O": 1e-4 * h2_mol, "NH3": 1e-4 * h2_mol}
    _guess_state(m.fs.t_rec.properties_out[0], recycle_guess,
                 T_FLASH, P_LOOP, LIQ_COMPS)

    # iterate around the recycle loop
    for it in range(3):
        _prop(m.fs.a09)
        _prop(m.fs.a11)
        _prop(m.fs.a20)
        m.fs.mixer1.initialize()
        _prop(m.fs.a13)
        m.fs.h1.initialize()
        _prop(m.fs.a14)
        m.fs.b1.initialize()
        _prop(m.fs.a15)
        _clamp_state(m.fs.t_rx.properties_out[0])
        m.fs.t_rx.initialize()
        _prop(m.fs.a16)
        m.fs.c1.initialize()
        _prop(m.fs.a17)
        m.fs.b2.initialize()
        _prop(m.fs.a18)
        m.fs.split1.initialize()
        _prop(m.fs.a19)
        _clamp_state(m.fs.t_rec.properties_out[0])
        m.fs.t_rec.initialize()
        print(f"  recycle pass {it + 1}: recycle flow = "
              f"{value(m.fs.t_rec.properties_out[0].flow_mol):.4f} mol/s")

    _prop(m.fs.a21)
    m.fs.purge.initialize()
    _prop(m.fs.a22)
    m.fs.v1.initialize()
    _prop(m.fs.a23)
    m.fs.s13.initialize()

    return solver


def solve(m, solver=None):
    if solver is None:
        solver = get_solver("ipopt", options={"tol": 1e-6, "max_iter": 500})
    dof = degrees_of_freedom(m)
    print(f"degrees of freedom before solve: {dof}")
    assert dof == 0, "flowsheet is not square"
    results = solver.solve(m, tee=True)
    return results


def _aspen_row(ft, fh2):
    """Look up the matching row of aspen/data/ammoniaF_results_live.csv."""
    import csv
    csv_path = (Path(__file__).resolve().parents[1]
                / "aspen" / "data" / "ammoniaF_results_live.csv")
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            try:
                if (abs(float(r["Ft"]) - ft) < 1e-6
                        and abs(float(r["Fh2"]) - fh2) < 1e-6):
                    return r
            except (KeyError, ValueError):
                continue
    return None


def report(m, ft=8500.0, fh2=10.0):
    print("\n================ ammoniaF IDAES results ================")
    print(f"NH3 in S13 product : {value(m.fs.nh3_product_kgph):10.3f} kg/h")
    print(f"H2O in S13 product : {value(m.fs.h2o_product_kgph):10.3f} kg/h")
    print(f"S13 total mass (Mt): {value(m.fs.s13_mass_kgph):10.3f} kg/h")
    for name, expr in [("Qh1 (H1 duty)", m.fs.Qh1), ("Qc1 (C1 duty)", m.fs.Qc1),
                       ("Qr1 (B1 duty)", m.fs.Qr1),
                       ("Qcomp (IC duty)", m.fs.Qcomp),
                       ("W_comp (work)", m.fs.W_comp)]:
        w = value(expr)
        print(f"{name:20s}: {w / 1e3:10.2f} kW  = {w * W_TO_MMKCAL_H:+.6f} MMkcal/h")
    s13 = m.fs.s13.properties[0]
    print(f"S13 temperature    : {value(s13.temperature):.2f} K")
    purge = m.fs.purge.properties[0]
    print(f"Purge flow         : {value(purge.flow_mol):.4f} mol/s")

    row = _aspen_row(ft, fh2)
    if row is None:
        print("\n(no matching Aspen row found for validation)")
        return
    print(f"\n===== validation vs Aspen (Ft={ft:g}, Fh2={fh2:g}) =====")
    print(f"{'quantity':22s}{'IDAES':>14s}{'Aspen':>14s}{'rel.dev':>10s}")

    def line(label, idaes_val, aspen_val, unit):
        dev = (idaes_val - aspen_val) / abs(aspen_val) if aspen_val else float("nan")
        print(f"{label:22s}{idaes_val:14.4f}{aspen_val:14.4f}{dev:+9.1%} {unit}")

    line("NH3 out", value(m.fs.nh3_product_kgph),
         float(row["NH3_out_kgph"]), "kg/h")
    line("S13 mass flow (Mt)", value(m.fs.s13_mass_kgph),
         float(row["Mt"]), "kg/h")
    line("NH3 mass frac (Wnh3)",
         value(m.fs.nh3_product_kgph) / value(m.fs.s13_mass_kgph),
         float(row["Wnh3"]), "-")
    line("Qh1", value(m.fs.Qh1) * W_TO_MMKCAL_H,
         float(row["Qh1"]), "MMkcal/h")
    line("Qc1", value(m.fs.Qc1) * W_TO_MMKCAL_H,
         float(row["Qc1"]), "MMkcal/h")
    line("Qr1", value(m.fs.Qr1) * W_TO_MMKCAL_H,
         float(row["Qr1"]), "MMkcal/h")
    line("Qcomp (QNET)", value(m.fs.Qcomp) * W_TO_MMKCAL_H,
         float(row["Qcomp"]), "MMkcal/h")


if __name__ == "__main__":
    ft, fh2 = 8500.0, 10.0
    if len(sys.argv) == 3:
        ft, fh2 = float(sys.argv[1]), float(sys.argv[2])
    m = build_model(fh2_kgph=fh2)
    print("model built; initializing...")
    solver = initialize(m)
    results = solve(m, solver)
    report(m, ft=ft, fh2=fh2)

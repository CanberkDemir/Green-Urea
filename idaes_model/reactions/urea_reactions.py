"""
Reaction packages for the ureaF (Bosch-Meiser) flowsheet.

Two GenericReactionParameterBlock configurations, both replicating
``aspen/ureaF.bkp`` records verbatim:

``get_rxn(prop)`` - the POWERLAW set ``RXNTOT`` used only by the synthesis
reactor R01 (REAC-DATA / STOIC records):

    R1 CARBF : 2NH3 + CO2 -> CARB        r = k1 T e^(-Ea1/RT) C_NH3,v^2 C_CO2,v
    R2 CARBD : CARB -> 2NH3 + CO2        r = k2 T e^(-Ea2/RT) C_CARB,l
    R3 UREAF : CARB -> UREA + H2O        r = k3 T e^(-Ea3/RT) C_CARB,l
    R4 UREAD : UREA + H2O -> CARB        r = k4 T e^(-Ea4/RT) C_UREA,l C_H2O,l

    k1 = k2 = 1000, Ea1 = Ea2 =  8000 cal/mol   (PRE-EXP / ACT-ENERGY)
    k3 = k4 =  750, Ea3 = Ea4 = 10000 cal/mol

  R1/R2 carry PHASE = V in the .bkp (vapor-phase concentrations); R3/R4
  specify no phase (Aspen default = liquid for an RPlug with a liquid phase
  present) and run on liquid concentrations. DOCUMENTED DEVIATION: the
  property package declares carbamate liquid-only (see ideal_urea.py), so
  R2's concentration basis is the LIQUID carbamate concentration instead of
  Aspen's (tiny) vapor one - the R1/R2 pseudo-equilibrium becomes the
  heterogeneous form C_NH3,v^2 C_CO2,v = C_CARB,l (K_c = 1 in Aspen's
  kmol/m3 units) instead of an all-vapor one. Because k1 = k2 and k3 = k4
  with equal activation energies, each pair pins such an equilibrium; the
  .bkp comment block on R01 says exactly this ("make Par #1 large enough to
  keep Rxn #1 at equilibrium"). The finite rates then set how far the 65%
  CO2->carbamate / 55% carbamate->urea single-pass targets (SI-1 /
  Pustjens) are approached in the reactor volume.

``get_carb_stoich_rxn(prop)`` + ``add_carbeq_conversion(reactor)`` - the
replacement for the three RGibbs blocks (A01 / A012 / A013). Those blocks
minimise Gibbs energy at fixed T with UREA (and the unused N2/O2) declared
inert, i.e. the ONLY chemistry they can perform is 2NH3 + CO2 ->
CARB(condensed). The relevant equilibrium is

    x_CARB,liq = K_p(T) * p_NH3^2 * p_CO2,      p_j = y_j * P

    K_p(298.15) = 3.2e-12 Pa^-3
    dH_rxn      = -146.2 kJ/mol

with K(T) derived from the same formation data Aspen uses ([BKP]
DGFORM/DHFORM for CARB, NIST for NH3/CO2), moved to the condensed-
carbamate basis consistent with the property package (divide the gas-basis
K by carbamate's [BKP] PLXANT saturation pressure; subtract its 88.0
kJ/mol vaporisation enthalpy). Sanity anchor: the literature dissociation
pressure of solid ammonium carbamate (0.117 bar at 298 K) gives
K_p = 4.2e-12 Pa^-3 and dH = -159 kJ/mol - Aspen's pseudo-component data
is evidently built to reproduce carbamate dissociation.

At all three units' conditions (72.4 degC, P >= 4 bar) that equilibrium
lies within ~0.3% of COMPLETE consumption of the limiting gas reagent
(worked example at A012: the interior crossing sits at 99.7% CO2
conversion), which makes an equality-constrained free extent numerically
hopeless - IPOPT keeps overshooting into the vapor-death corner (an
EquilibriumReactor with the log-form constraint fails the same way). The
replacement therefore fixes the extent at 99% of the limiting reagent via
a smooth min - equivalent to the Gibbs answer within the thermodynamics'
own accuracy, and robust. ``carbeq_expression`` (the raw equilibrium
relation) is kept for diagnostics and for anyone revisiting the free-
extent formulation.

Implementation notes (same conventions as ammonia_reactions.py):
* Aspen's T-EXP = 1 prefactor (rate ~ k * T * e^(-Ea/RT) * C) has no IDAES
  arrhenius equivalent, so T = 473.15 K (R01's isothermal CONST-TEMP spec)
  is folded into the pre-exponential constants. Exact at the reactor
  temperature; the reaction package is used nowhere else.
* Aspen power-law concentrations are kmol/m3; IDAES molarity is mol/m3.
  Pre-exponentials are rescaled by 1000^(1 - sum(orders)) so the rate in
  mol/m3/s is identical.
* Rates are applied to the total control-volume holdup rather than to the
  per-phase holdups Aspen's two-phase RPlug uses. With the plant's grid
  feeds (5-50 kg/h) the 98 m3 reactor is hugely oversized, so both reaction
  pairs sit at their equilibrium points and the volume-basis difference
  does not move the outlet state (verified in test_urea_kinetics.py).
* No declared heats of reaction: the property package carries ideal-gas
  formation enthalpies for every species (including [BKP] DHFORM for
  UREA/CARB), so reaction heat emerges from the energy balance. Build units
  with has_heat_of_reaction=False.
"""

from pyomo.environ import Constraint, exp, units as pyunits

from idaes.core.util.math import smooth_min
from idaes.models.properties.modular_properties.base.generic_reaction import (
    ConcentrationForm,
)
from idaes.models.properties.modular_properties.reactions.rate_constant import (
    arrhenius,
)
from idaes.models.properties.modular_properties.reactions.rate_forms import (
    power_law_rate,
)

_CAL = 4.184  # J per thermochemical calorie
T_REACTOR = 473.15  # K, R01 CONST-TEMP spec; folds Aspen's T-EXP=1 into k0

# All four pre-exponentials are scaled down together. The .bkp's comment
# block states the magnitudes are arbitrary above the "large enough to keep
# the reaction at equilibrium" threshold, and at grid-scale feeds the 98 m3
# reactor is equilibrium-limited with or without the scale (Damkohler
# ~1e3-1e7; test_urea_kinetics.py asserts the equilibrium closure). Without
# it, the near-cancelling forward/reverse rate terms reach 1e4-1e6 mol/s
# against material balances of ~0.1 mol/s - more cancellation than double
# precision can deliver, and IPOPT stalls. Ratios (k1/k2, k3/k4), and hence
# both pinned equilibria, are unchanged.
RATE_SCALE = 1e-5

_BASE_UNITS = {
    "time": pyunits.s,
    "length": pyunits.m,
    "mass": pyunits.kg,
    "amount": pyunits.mol,
    "temperature": pyunits.K,
}

# name: (Aspen PRE-EXP, Ea [cal/mol], {(phase, comp): stoich},
#        {(phase, comp): order})
_RXNTOT = {
    "CARBF": (
        1000.0, 8000.0,
        {("Vap", "NH3"): -2, ("Vap", "CO2"): -1, ("Liq", "CARB"): 1},
        {("Vap", "NH3"): 2.0, ("Vap", "CO2"): 1.0},
    ),
    "CARBD": (
        1000.0, 8000.0,
        {("Liq", "CARB"): -1, ("Vap", "NH3"): 2, ("Vap", "CO2"): 1},
        {("Liq", "CARB"): 1.0},  # liquid basis - see docstring deviation
    ),
    "UREAF": (
        750.0, 10000.0,
        {("Liq", "CARB"): -1, ("Liq", "UREA"): 1, ("Liq", "H2O"): 1},
        {("Liq", "CARB"): 1.0},
    ),
    "UREAD": (
        750.0, 10000.0,
        {("Liq", "UREA"): -1, ("Liq", "H2O"): -1, ("Liq", "CARB"): 1},
        {("Liq", "UREA"): 1.0, ("Liq", "H2O"): 1.0},
    ),
}


def get_rxn(property_package):
    """RXNTOT kinetic network (R01 only) as a reaction-package config."""
    rate_reactions = {}
    for name, (k0, ea_cal, stoich, orders) in _RXNTOT.items():
        n_tot = sum(orders.values())
        # kmol/m3 -> mol/m3 basis change + T-EXP=1 folded at T_REACTOR
        k_eff = k0 * T_REACTOR * 1000.0 ** (1.0 - n_tot) * RATE_SCALE
        k_units = (pyunits.mol / pyunits.m**3) ** (1.0 - n_tot) / pyunits.s
        full_orders = {key: 0.0 for key in stoich}
        full_orders.update(orders)
        rate_reactions[name] = {
            "stoichiometry": dict(stoich),
            "rate_constant": arrhenius,
            "rate_form": power_law_rate,
            "concentration_form": ConcentrationForm.molarity,
            "parameter_data": {
                "arrhenius_const": (k_eff, k_units),
                "energy_activation": (ea_cal * _CAL, pyunits.J / pyunits.mol),
                "reaction_order": full_orders,
            },
        }

    return {
        "property_package": property_package,
        "base_units": dict(_BASE_UNITS),
        "rate_reactions": rate_reactions,
    }


K_P_298 = 3.2e-12  # Pa^-3, condensed-carbamate formation constant at 298.15 K
DH_RXN_COND = -146.2e3  # J/mol, 2NH3(g) + CO2(g) -> CARB(l)
_T_REF = 298.15  # K
_R_GAS = 8.314462618  # J/mol/K


def get_carb_stoich_rxn(property_package):
    """Stoichiometry-only package for the RGibbs-replacement units.

    One reaction (CARBF, 2NH3(v) + CO2(v) -> CARB(l)) with no rate form:
    a StoichiometricReactor exposes its extent as the degree of freedom,
    which ``add_carbeq_constraint`` closes.
    """
    return {
        "property_package": property_package,
        "base_units": dict(_BASE_UNITS),
        "rate_reactions": {
            "CARBF": {
                "stoichiometry": {
                    ("Vap", "NH3"): -2,
                    ("Vap", "CO2"): -1,
                    ("Liq", "CARB"): 1,
                },
            },
        },
    }


def carbeq_expression(state):
    """x_CARB,liq - K_p(T) p_NH3^2 p_CO2 == 0 on an FTPx state block."""
    T = state.temperature
    P = state.pressure
    kp = (
        K_P_298
        * exp(-(DH_RXN_COND / _R_GAS) * (1 / T - 1 / (_T_REF * pyunits.K)))
        * pyunits.Pa**-3
    )
    p_nh3 = state.mole_frac_phase_comp["Vap", "NH3"] * P
    p_co2 = state.mole_frac_phase_comp["Vap", "CO2"] * P
    return state.mole_frac_phase_comp["Liq", "CARB"] == kp * p_nh3**2 * p_co2


CARBEQ_CONV = 0.99  # fraction of the limiting gas reagent condensed


def add_carbeq_conversion(reactor, conv=CARBEQ_CONV):
    """Fix a StoichiometricReactor's extent at ``conv`` of the limiting
    gas reagent (RGibbs replacement - see module docstring for why this
    is equivalent to the equilibrium at the A01/A012/A013 conditions).

    extent = conv * smooth_min(F_CO2_in, F_NH3_in / 2)
    """
    t0 = reactor.flowsheet().time.first()
    cv = reactor.control_volume
    st_in = cv.properties_in[t0]
    ext = cv.rate_reaction_extent[t0, "CARBF"]
    f_co2 = st_in.flow_mol * st_in.mole_frac_comp["CO2"]
    f_nh3 = st_in.flow_mol * st_in.mole_frac_comp["NH3"]
    # eps in mol/s; feeds are O(0.01-1) mol/s
    reactor.carbeq_conv = Constraint(
        expr=ext == conv * smooth_min(f_co2, f_nh3 / 2, eps=1e-6)
    )

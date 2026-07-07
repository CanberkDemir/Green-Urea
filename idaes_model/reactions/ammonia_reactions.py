"""
Reaction package for the nitrate-hydrogenation (ammoniaF) RPLUG reactor.

Replicates the reaction set ``AMMONIA1`` EXACTLY as implemented in
``aspen/ammoniaF.bkp`` (REAC-DATA / STOIC records, verbatim):

    MAIN1: NO3- +   H2 ->  NO2- +  H2O    r = k1 T e^(-Ea1/RT) C_NO3
    MAIN2: 2NO2- + 7H2 -> 2NH3  + 4H2O    r = k2 T e^(-Ea2/RT) C_NO2
    SIDE1: 2NO2- + 4H2 ->  N2   + 4H2O    r = k3 T e^(-Ea3/RT) C_NO2
    SIDE2: 2NO3- + 6H2 ->  N2   + 6H2O    r = k4 T e^(-Ea4/RT) C_NO3

    k1 = 3.0e3, Ea1 =  9560 cal/mol       (PRE-EXP / ACT-ENERGY records)
    k2 = 3.5e5, Ea2 =  9000 cal/mol
    k3 = 8.2e3, Ea3 = 11970 cal/mol
    k4 = 2.0e2, Ea4 =  8370 cal/mol

Every rate is strictly first order in the consumed nitrogen ion (the .bkp
puts EXPONENT = 1 on the ion only; H2 carries no exponent), on a molarity
basis, liquid phase.

IMPORTANT - documented discrepancy with the manuscript SI (Table S7):
the SI describes the side reactions as r3: NO3- -> N2 (k=8.2e3) and
r4: NO2- -> N2 (k=2.0e2), which would give ~93% NH3 selectivity at 60 degC.
The Aspen file swaps the acting ions (SIDE1 with k=8.2e3 consumes NO2-,
SIDE2 with k=2.0e2 consumes NO3-) and uses per-extent coefficients of -2
on the ions, giving ~55% ultimate NH3 selectivity - which is what the
recorded Aspen results reproduce (NH3_out = 12.75 kg/h at Ft=8500,
Fh2=10, vs. the ~20 kg/h the SI network would give). This package
implements the file, i.e. what Aspen actually ran.

Implementation notes:
* Stoichiometry is normalised per mole of consumed ion (coefficients / 2
  for MAIN2/SIDE1/SIDE2) and the pre-exponential factors are doubled to
  keep d[ion]/dt identical to Aspen's extent formulation.
* Aspen applies a T-EXP = 1 prefactor (rate ~ k * T * exp(-Ea/RT) * C).
  IDAES's arrhenius form has no T^n term, so T = 333.15 K (the reactor's
  isothermal operating temperature) is folded into the pre-exponential
  constants. This is exact at operating temperature; it only (weakly)
  affects rate magnitudes - not selectivity, which is a k-ratio - if the
  reactor temperature is changed.
* Aspen's MAIN2/SIDE1/SIDE2 records are not charge-balanced (no OH-
  production; Aspen's solution chemistry re-speciates on the apparent
  basis). They are replicated verbatim here - mass is conserved, charge
  bookkeeping drifts exactly as in the Aspen apparent-component records.
* Heats of reaction are not declared: the property package carries
  standard-state formation enthalpies for every species, so reaction heat
  emerges from the energy balance (use has_heat_of_reaction=False).
"""

from pyomo.environ import units as pyunits

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
T_OPERATING = 333.15  # K, folds Aspen's T-EXP=1 prefactor into k0

# name: (Aspen PRE-EXP [1/s], Ea [cal/mol], consumed ion,
#        ion moles consumed per Aspen extent, per-ion stoichiometry)
_NETWORK = {
    "MAIN1": (
        3.0e3, 9560.0, "NO3-", 1,
        {
            ("Liq", "NO3-"): -1,
            ("Liq", "H2"): -1,
            ("Liq", "NO2-"): 1,
            ("Liq", "H2O"): 1,
        },
    ),
    "MAIN2": (
        3.5e5, 9000.0, "NO2-", 2,
        {
            ("Liq", "NO2-"): -1,
            ("Liq", "H2"): -3.5,
            ("Liq", "NH3"): 1,
            ("Liq", "H2O"): 2,
        },
    ),
    "SIDE1": (
        8.2e3, 11970.0, "NO2-", 2,
        {
            ("Liq", "NO2-"): -1,
            ("Liq", "H2"): -2,
            ("Liq", "N2"): 0.5,
            ("Liq", "H2O"): 2,
        },
    ),
    "SIDE2": (
        2.0e2, 8370.0, "NO3-", 2,
        {
            ("Liq", "NO3-"): -1,
            ("Liq", "H2"): -3,
            ("Liq", "N2"): 0.5,
            ("Liq", "H2O"): 3,
        },
    ),
}


def get_rxn(property_package):
    """Return a GenericReactionParameterBlock configuration dict."""
    rate_reactions = {}
    for name, (k0, ea_cal, ion, ion_per_extent, stoich) in _NETWORK.items():
        # per-ion normalisation: rate constant scaled by the ion moles
        # consumed per Aspen extent, T-EXP=1 folded in at T_OPERATING
        k_eff = k0 * ion_per_extent * T_OPERATING
        orders = {("Liq", j): 0.0 for p, j in stoich}
        orders[("Liq", ion)] = 1.0
        rate_reactions[name] = {
            "stoichiometry": dict(stoich),
            "rate_constant": arrhenius,
            "rate_form": power_law_rate,
            "concentration_form": ConcentrationForm.molarity,
            "parameter_data": {
                "arrhenius_const": (k_eff, pyunits.s**-1),
                "energy_activation": (ea_cal * _CAL, pyunits.J / pyunits.mol),
                "reaction_order": orders,
            },
        }

    return {
        "property_package": property_package,
        "base_units": {
            "time": pyunits.s,
            "length": pyunits.m,
            "mass": pyunits.kg,
            "amount": pyunits.mol,
            "temperature": pyunits.K,
        },
        "rate_reactions": rate_reactions,
    }

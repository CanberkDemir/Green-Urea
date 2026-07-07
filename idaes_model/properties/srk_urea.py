"""
Property package for the ureaF (Bosch-Meiser urea synthesis) flowsheet.

Replicates the thermodynamic setup of ``aspen/ureaF.bkp``. Aspen's global
method there is SR-POLAR (Schwartzentruber-Renon), but the .bkp populates
*no* polar parameters (the PARMNB record is empty, as are RKTKIJ / BWRK* /
CLSK*), so the method degenerates to plain RKS with quadratic mixing and the
binary kij values that ARE embedded in the file (RKUKA0 records). This
package therefore uses IDAES's cubic SRK for both phases with those kij
values - a close structural match, not an approximation layered on top of
missing data.

Everything marked [BKP] below is copied verbatim from ``aspen/ureaF.bkp``:

* UREA / CARB critical constants (TC / PC / OMEGA records - note Aspen's own
  placeholder-ish Tc = 1000 K for both pseudo-heavies; replicated verbatim
  because the recorded Aspen results were produced with them),
* UREA / CARB ideal-gas formation enthalpy and Gibbs energy (DHFORM/DGFORM),
* UREA / CARB ideal-gas cp (CPIGDP, DIPPR-107) - refitted here to the RPP4
  cubic polynomial over 250-800 K (max error 1.5% UREA / 5.9% CARB, at the
  range edges),
* the SRK binary interaction matrix (RKUKA0). The single small
  temperature-dependent term (RKUKA1 urea/water = -3.04e-4 1/K) is dropped;
  IDAES's cubic takes constant kappa.

NH3 / CO2 / H2O use standard Reid-Prausnitz-Poling 4th ed. ("RPP") and NIST
values, matching what Aspen resolves from its APV140 databank for these
common species (those values are bank-referenced, not embedded).

The Aspen component list also declares N2, but no feed stream contains it
and it appears nowhere in the recorded results; it is omitted here to avoid
carrying a dead trace component through every VLE flash.

Volatility note: Aspen's PLXANT (extended Antoine) records for UREA/CARB are
NOT used by SR-POLAR (fugacities come from the EoS) and are likewise unused
here - urea/carbamate volatility comes from the cubic EoS with the [BKP]
critical constants, exactly as in the source model.

Same swappable interface as ``enrtl_ammonia``: ``get_prop(phases)`` returns
a GenericParameterBlock config dict. To change the thermo basis later,
provide another module with the same signature.

PLATFORM CAVEAT: IDAES's cubic EoS evaluates compressibility roots through
the compiled AMPL external functions in ``cubic_roots.so`` (idaes-ext),
which has no Fedora-aarch64 binary. A locally compiled library (gcc against
the conda-forge ASL headers) loads and evaluates correctly in-process but
SEGFAULTS inside conda-forge IPOPT's ASL on the first derivative callback,
so this package currently cannot be *solved* on this machine. The flowsheet
therefore runs on ``ideal_urea.py`` (same interface); swap the import back
once a working cubic_roots.so exists.
"""

from copy import deepcopy

from pyomo.environ import units as pyunits

from idaes.core import LiquidPhase, VaporPhase, Component
from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.eos.ceos import Cubic, CubicType
from idaes.models.properties.modular_properties.phase_equil import (
    CubicComplementarityVLE,
)
from idaes.models.properties.modular_properties.phase_equil.bubble_dew import (
    LogBubbleDew,
)
from idaes.models.properties.modular_properties.phase_equil.forms import log_fugacity
from idaes.models.properties.modular_properties.pure import NIST, RPP4


COMPONENTS = ["NH3", "CO2", "H2O", "UREA", "CARB"]

_J = pyunits.J / pyunits.mol
_JK = pyunits.J / pyunits.mol / pyunits.K


def _comp(mw, tc, pc, omega, cp_abcd, hf, sf, antoine):
    """Cubic-EoS component entry (RPP4 ideal-gas cp/enth/entr basis).

    ``antoine`` (NIST log10-bar form) is used ONLY for initialization
    guesses - VLE itself runs on cubic fugacities. Values above Tc are
    extrapolations and never enter the converged solution.
    """
    a, b, c, d = cp_abcd
    ant_a, ant_b, ant_c = antoine
    return {
        "type": Component,
        "enth_mol_ig_comp": RPP4,
        "entr_mol_ig_comp": RPP4,
        "pressure_sat_comp": NIST,
        "phase_equilibrium_form": {("Vap", "Liq"): log_fugacity},
        "parameter_data": {
            "mw": (mw, pyunits.kg / pyunits.mol),
            "temperature_crit": (tc, pyunits.K),
            "pressure_crit": (pc, pyunits.Pa),
            "omega": omega,
            "cp_mol_ig_comp_coeff": {
                "A": (a, _JK),
                "B": (b, pyunits.J / pyunits.mol / pyunits.K**2),
                "C": (c, pyunits.J / pyunits.mol / pyunits.K**3),
                "D": (d, pyunits.J / pyunits.mol / pyunits.K**4),
            },
            "enth_mol_form_vap_comp_ref": (hf, _J),
            "entr_mol_form_vap_comp_ref": (sf, _JK),
            "pressure_sat_comp_coeff": {
                "A": (ant_a, None),
                "B": (ant_b, pyunits.K),
                "C": (ant_c, pyunits.K),
            },
        },
    }


_COMP_DATA = {
    # mw [kg/mol], Tc [K], Pc [Pa], omega, cp RPP4 (A,B,C,D) [J/mol/K...],
    # dHf_ig [J/mol], dSf_ig [J/mol/K]
    "NH3": _comp(  # [RPP] criticals + cp; [NIST] formation + Antoine
        17.0305e-3, 405.5, 113.5e5, 0.250,
        (2.731e1, 2.383e-2, 1.707e-5, -1.185e-8),
        -45.898e3, -99.05,
        (4.86886, 1113.928, -10.409),
    ),
    "CO2": _comp(  # [RPP] criticals + cp; [NIST] formation + Antoine
        44.0095e-3, 304.2, 73.8e5, 0.224,
        (1.980e1, 7.344e-2, -5.602e-5, 1.715e-8),
        -393.51e3, 2.90,
        (6.81228, 1301.679, -3.494),
    ),
    "H2O": _comp(  # [RPP] criticals + cp; [NIST] formation + Antoine
        18.0153e-3, 647.3, 221.2e5, 0.344,
        (3.224e1, 1.924e-3, 1.055e-5, -3.596e-9),
        -241.826e3, -44.42,
        (4.6543, 1435.264, -64.848),
    ),
    "UREA": _comp(  # [BKP] TC/PC/OMEGA/DHFORM; cp fit of [BKP] CPIGDP;
        # dSf = (DHFORM - DGFORM)/298.15 with [BKP] DGFORM = -1.6028620e8
        # J/kmol, keeping Gibbs energies exactly Aspen-consistent.
        # Antoine refit of the [BKP] PLXANT curve over 300-550 K.
        60.0553e-3, 1000.0, 1.284e7, 0.220,
        (-2.21554, 0.289238, -2.133e-4, 6.06076e-8),
        -245.60e3, -286.14,
        (7.58626, 4596.306, 0.787),
    ),
    "CARB": _comp(  # ammonium carbamate NH2COONH4; [BKP] as above,
        # DGFORM = -4.0068750e8 J/kmol -> dSf = -479.02 J/mol/K
        78.0707e-3, 1000.0, 9.32e6, 0.450,
        (-130.051, 0.722429, -2.65925e-4, 1.64947e-8),
        -543.50e3, -479.02,
        (7.21407, 4596.306, 0.787),
    ),
}

# [BKP] RKUKA0 binary interaction parameters (symmetric; unlisted pairs 0).
# The lone temperature-dependent term (RKUKA1 UREA/H2O = -3.04e-4 1/K) is
# dropped: IDAES's cubic uses constant kappa, and over the plant's 295-475 K
# window it shifts kij by less than +/-0.03.
_KIJ = {
    ("NH3", "H2O"): -0.280,
    ("CO2", "H2O"): -0.050,
    ("NH3", "CARB"): -0.0176702,
    ("H2O", "CARB"): -0.0787303,
    ("UREA", "NH3"): -0.131021,
    ("UREA", "H2O"): 0.08996,
}


def _kappa():
    kappa = {}
    for i in COMPONENTS:
        for j in COMPONENTS:
            kappa[(i, j)] = _KIJ.get((i, j), _KIJ.get((j, i), 0.0))
    return kappa


def get_prop(phases=("Vap", "Liq")):
    """Return a GenericParameterBlock configuration dict.

    Args:
        phases: iterable drawn from {"Liq", "Vap"}. The urea flowsheet is
            two-phase throughout (every Aspen block runs VLE on the same
            SR-POLAR basis), so ("Vap", "Liq") is the variant used
            everywhere; single-phase variants are provided for standalone
            testing.
    """
    phases = tuple(phases)
    if not phases or any(p not in ("Liq", "Vap") for p in phases):
        raise ValueError(f"phases must be drawn from ('Liq', 'Vap'), got {phases}")
    vle = "Liq" in phases and "Vap" in phases

    components = {}
    for name in COMPONENTS:
        cfg = deepcopy(_COMP_DATA[name])
        if not vle:
            cfg.pop("phase_equilibrium_form", None)
        components[name] = cfg

    phase_cfg = {}
    if "Liq" in phases:
        phase_cfg["Liq"] = {
            "type": LiquidPhase,
            "equation_of_state": Cubic,
            "equation_of_state_options": {"type": CubicType.SRK},
        }
    if "Vap" in phases:
        phase_cfg["Vap"] = {
            "type": VaporPhase,
            "equation_of_state": Cubic,
            "equation_of_state_options": {"type": CubicType.SRK},
        }

    config = {
        "components": components,
        "phases": phase_cfg,
        "base_units": {
            "time": pyunits.s,
            "length": pyunits.m,
            "mass": pyunits.kg,
            "amount": pyunits.mol,
            "temperature": pyunits.K,
        },
        "state_definition": FTPx,
        # flows: grid feeds are 5-50 kg/h (~0.03-0.3 mol/s); recycles run a
        # few times larger. T spans the 0 degC adiabatic flashes to the
        # 200 degC synthesis loop; P spans 1 bar let-down to 200 bar pumps.
        "state_bounds": {
            "flow_mol": (0, 1, 500, pyunits.mol / pyunits.s),
            "temperature": (250, 350, 700, pyunits.K),
            "pressure": (1e4, 1e5, 3e7, pyunits.Pa),
        },
        "pressure_ref": (101325, pyunits.Pa),
        "temperature_ref": (298.15, pyunits.K),
        "parameter_data": {"SRK_kappa": _kappa()},
    }

    if vle:
        config["phases_in_equilibrium"] = [("Vap", "Liq")]
        config["phase_equilibrium_state"] = {("Vap", "Liq"): CubicComplementarityVLE}
        config["bubble_dew_method"] = LogBubbleDew

    return config

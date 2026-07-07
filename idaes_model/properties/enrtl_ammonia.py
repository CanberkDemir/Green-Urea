"""
Property package for the ammoniaF (nitrate-hydrogenation) flowsheet.

Replicates the thermodynamic setup of ``aspen/ammoniaF.bkp`` as closely as
IDAES allows. Aspen uses ELECNRTL (electrolyte NRTL, "ENRTL-RK") with the
licensed APV140 databanks; those binary-interaction values are referenced by
bank name only inside the .bkp and are not extractable, so:

* The liquid phase uses IDAES's eNRTL equation of state
  (``idaes.models.properties.modular_properties.eos.enrtl.ENRTL``), which is
  the same activity-coefficient model family Aspen uses.
* Molecule/ion-pair tau parameters use the published Chen (1982) value for
  (H2O, Na+ Cl-) and Aspen's own documented defaults (tau_m,ca = 8,
  tau_ca,m = -4) for pairs whose databank values are inaccessible.
* NOTE (IDAES limitation): IDAES's ENRTL class inherits its fugacity
  expressions from the Ideal EOS, so vapor-liquid equilibrium is computed by
  Raoult's law (H2O) / Henry's law (NH3, H2, N2) WITHOUT the eNRTL activity
  correction. The eNRTL activity coefficients are available on the liquid
  state blocks, but Aspen's gamma-corrected VLE is not reproduced exactly.

Three package variants are exposed so different flowsheet sections can match
the Aspen block phase specifications (the RPLUG runs NPHASE=1 PHASE=L):

* ``get_prop(("Liq",))``       - liquid-only, for the reaction train
                                 (B6, MIXER1, H1, B1).
* ``get_prop(("Liq", "Vap"))`` - two-phase VLE, for the separation train
                                 (C1, B2, SPLIT1, V1).
* ``get_prop(("Vap",))``       - vapor-only ideal gas, for the H2 compressor
                                 train (MCOMPR1). Includes entropy data needed
                                 by isentropic compressors.

To swap the whole thermodynamic basis later, provide another module with the
same ``get_prop(phases)`` signature and point the flowsheet at it.

Data sources are cited inline: [NIST] NIST Chemistry WebBook (Shomate /
Antoine); [Perry] Perry's Chemical Engineers' Handbook (DIPPR forms);
[NBS] NBS Tables of Chemical Thermodynamic Properties (aqueous standard-state
formation enthalpies and partial molar heat capacities, 298.15 K);
[Sander] Sander, Compilation of Henry's law constants, ACP 2015;
[Chen] Chen et al., AIChE J. 28 (1982) 588.
"""

from copy import deepcopy

from pyomo.environ import units as pyunits

from idaes.core import (
    AqueousPhase,
    VaporPhase,
    Component,
    Solvent,
    Solute,
    Anion,
    Cation,
    PhaseType as PT,
)
from idaes.models.properties.modular_properties.base.generic_property import (
    StateIndex,
)
from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.eos.enrtl import ENRTL
from idaes.models.properties.modular_properties.eos.enrtl_reference_states import (
    Symmetric,
)
from idaes.models.properties.modular_properties.eos.ideal import Ideal
from idaes.models.properties.modular_properties.phase_equil import SmoothVLE
from idaes.models.properties.modular_properties.phase_equil.bubble_dew import (
    IdealBubbleDew,
)
from idaes.models.properties.modular_properties.phase_equil.forms import fugacity
from idaes.models.properties.modular_properties.phase_equil.henry import (
    ConstantH,
    HenryType,
)
from idaes.models.properties.modular_properties.pure.ConstantProperties import (
    Constant,
)
from idaes.models.properties.modular_properties.pure.NIST import NIST
from idaes.models.properties.modular_properties.pure.Perrys import Perrys
from idaes.models.properties.modular_properties.pure.electrolyte import (
    relative_permittivity_constant,
)


# ---------------------------------------------------------------------------
# Pure/solute component data
# ---------------------------------------------------------------------------

_H2O = {
    "type": Solvent,
    "dens_mol_liq_comp": Perrys,
    "enth_mol_liq_comp": Perrys,
    "enth_mol_ig_comp": NIST,
    "entr_mol_ig_comp": NIST,
    "pressure_sat_comp": NIST,
    "relative_permittivity_liq_comp": relative_permittivity_constant,
    "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
    "parameter_data": {
        "mw": (18.0153e-3, pyunits.kg / pyunits.mol),  # [NIST]
        "pressure_crit": (220.64e5, pyunits.Pa),  # [NIST]
        "temperature_crit": (647, pyunits.K),  # [NIST]
        "relative_permittivity_liq_comp": 78.54,
        "dens_mol_liq_comp_coeff": {  # [Perry] DIPPR 105, 273-333 K
            "eqn_type": 1,
            "1": (5.459, pyunits.kmol * pyunits.m**-3),
            "2": (0.30542, None),
            "3": (647.13, pyunits.K),
            "4": (0.081, None),
        },
        "cp_mol_liq_comp_coeff": {  # [Perry] DIPPR 100, 273-533 K
            "1": (2.7637e5, pyunits.J / pyunits.kmol / pyunits.K),
            "2": (-2.0901e3, pyunits.J / pyunits.kmol / pyunits.K**2),
            "3": (8.125, pyunits.J / pyunits.kmol / pyunits.K**3),
            "4": (-1.4116e-2, pyunits.J / pyunits.kmol / pyunits.K**4),
            "5": (9.3701e-6, pyunits.J / pyunits.kmol / pyunits.K**5),
        },
        "cp_mol_ig_comp_coeff": {  # [NIST] Shomate, 500-1700 K
            "A": (30.09200, pyunits.J / pyunits.mol / pyunits.K),
            "B": (6.832514, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-1),
            "C": (6.793435, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-2),
            "D": (-2.534480, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-3),
            "E": (0.082139, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**2),
            "F": (-250.8810, pyunits.kJ / pyunits.mol),
            "G": (223.3967, pyunits.J / pyunits.mol / pyunits.K),
            "H": (-241.8264, pyunits.kJ / pyunits.mol),
        },
        "enth_mol_form_liq_comp_ref": (-285.83e3, pyunits.J / pyunits.mol),  # [NIST]
        "enth_mol_form_vap_comp_ref": (-241.8264e3, pyunits.J / pyunits.mol),  # [NIST]
        "pressure_sat_comp_coeff": {  # [NIST] Antoine (bar, K), 255.9-373 K
            "A": (4.6543, None),
            "B": (1435.264, pyunits.K),
            "C": (-64.848, pyunits.K),
        },
    },
}

# Dissolved-gas / volatile solutes. Henry's law (mole-fraction basis, Kpx)
# with constants representative of the 10-60 degC operating window [Sander].
_NH3 = {
    "type": Solute,
    "henry_component": {
        "Liq": {"method": ConstantH, "type": HenryType.Kpx, "basis": StateIndex.true},
    },
    "enth_mol_liq_comp": Constant,
    "dens_mol_liq_comp": Constant,
    "enth_mol_ig_comp": NIST,
    "entr_mol_ig_comp": NIST,
    "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
    "parameter_data": {
        "mw": (17.0305e-3, pyunits.kg / pyunits.mol),  # [NIST]
        "henry_ref": {"Liq": 6.0e4},  # [Sander] Kpx in Pa, 10-25 degC
        "cp_mol_liq_comp_coeff": (79.9, pyunits.J / pyunits.mol / pyunits.K),  # [NBS] NH3(aq)
        "enth_mol_form_liq_comp_ref": (-80.29e3, pyunits.J / pyunits.mol),  # [NBS] NH3(aq)
        "dens_mol_liq_comp_coeff": (35.0e3, pyunits.mol / pyunits.m**3),
        "cp_mol_ig_comp_coeff": {  # [NIST] Shomate, 298-1400 K
            "A": (19.99563, pyunits.J / pyunits.mol / pyunits.K),
            "B": (49.77119, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-1),
            "C": (-15.37599, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-2),
            "D": (1.921168, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-3),
            "E": (0.189174, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**2),
            "F": (-53.30667, pyunits.kJ / pyunits.mol),
            "G": (203.8591, pyunits.J / pyunits.mol / pyunits.K),
            "H": (-45.89806, pyunits.kJ / pyunits.mol),
        },
        "enth_mol_form_vap_comp_ref": (-45.898e3, pyunits.J / pyunits.mol),  # [NIST]
    },
}

_H2 = {
    "type": Solute,
    "henry_component": {
        "Liq": {"method": ConstantH, "type": HenryType.Kpx, "basis": StateIndex.true},
    },
    "enth_mol_liq_comp": Constant,
    "dens_mol_liq_comp": Constant,
    "enth_mol_ig_comp": NIST,
    "entr_mol_ig_comp": NIST,
    "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
    "parameter_data": {
        "mw": (2.01588e-3, pyunits.kg / pyunits.mol),  # [NIST]
        "henry_ref": {"Liq": 7.1e9},  # [Sander] Kpx in Pa, 298 K
        "cp_mol_liq_comp_coeff": (28.8, pyunits.J / pyunits.mol / pyunits.K),
        "enth_mol_form_liq_comp_ref": (-4.2e3, pyunits.J / pyunits.mol),  # [NBS] H2(aq)
        "dens_mol_liq_comp_coeff": (70.0e3, pyunits.mol / pyunits.m**3),
        "cp_mol_ig_comp_coeff": {  # [NIST] Shomate, 298-1000 K
            "A": (33.066178, pyunits.J / pyunits.mol / pyunits.K),
            "B": (-11.363417, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-1),
            "C": (11.432816, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-2),
            "D": (-2.772874, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-3),
            "E": (-0.158558, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**2),
            "F": (-9.980797, pyunits.kJ / pyunits.mol),
            "G": (172.707974, pyunits.J / pyunits.mol / pyunits.K),
            "H": (0.0, pyunits.kJ / pyunits.mol),
        },
        "enth_mol_form_vap_comp_ref": (0.0, pyunits.J / pyunits.mol),
    },
}

_N2 = {
    "type": Solute,
    "henry_component": {
        "Liq": {"method": ConstantH, "type": HenryType.Kpx, "basis": StateIndex.true},
    },
    "enth_mol_liq_comp": Constant,
    "dens_mol_liq_comp": Constant,
    "enth_mol_ig_comp": NIST,
    "entr_mol_ig_comp": NIST,
    "phase_equilibrium_form": {("Vap", "Liq"): fugacity},
    "parameter_data": {
        "mw": (28.0134e-3, pyunits.kg / pyunits.mol),  # [NIST]
        "henry_ref": {"Liq": 9.1e9},  # [Sander] Kpx in Pa, 298 K
        "cp_mol_liq_comp_coeff": (28.9, pyunits.J / pyunits.mol / pyunits.K),
        "enth_mol_form_liq_comp_ref": (-10.8e3, pyunits.J / pyunits.mol),  # approx N2(aq)
        "dens_mol_liq_comp_coeff": (40.0e3, pyunits.mol / pyunits.m**3),
        "cp_mol_ig_comp_coeff": {  # [NIST] Shomate, 100-500 K
            "A": (28.98641, pyunits.J / pyunits.mol / pyunits.K),
            "B": (1.853978, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-1),
            "C": (-9.647459, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-2),
            "D": (16.63537, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**-3),
            "E": (0.000117, pyunits.J * pyunits.mol**-1 * pyunits.K**-1 * pyunits.kiloK**2),
            "F": (-8.671914, pyunits.kJ / pyunits.mol),
            "G": (226.4168, pyunits.J / pyunits.mol / pyunits.K),
            "H": (0.0, pyunits.kJ / pyunits.mol),
        },
        "enth_mol_form_vap_comp_ref": (0.0, pyunits.J / pyunits.mol),
    },
}


def _ion(ion_type, charge, mw, cp, h_form):
    """Aqueous ion with constant partial-molar properties (298.15 K [NBS]).

    Molar density is a nominal 40 kmol/m3 (approx. 25 cm3/mol partial molar
    volume); the mixture density is dominated by the H2O solvent term.
    """
    return {
        "type": ion_type,
        "charge": charge,
        "enth_mol_liq_comp": Constant,
        "dens_mol_liq_comp": Constant,
        "parameter_data": {
            "mw": (mw, pyunits.kg / pyunits.mol),
            "cp_mol_liq_comp_coeff": (cp, pyunits.J / pyunits.mol / pyunits.K),
            "enth_mol_form_liq_comp_ref": (h_form, pyunits.J / pyunits.mol),
            "dens_mol_liq_comp_coeff": (40.0e3, pyunits.mol / pyunits.m**3),
        },
    }


_IONS = {
    # name: (type, charge, mw [kg/mol], cp aq [J/mol/K], dHf aq [J/mol]) [NBS]
    "Na+": _ion(Cation, +1, 22.98977e-3, 46.4, -240.12e3),
    "Cl-": _ion(Anion, -1, 35.453e-3, -136.4, -167.16e3),
    "NO3-": _ion(Anion, -1, 62.0049e-3, -86.6, -207.36e3),
    "NO2-": _ion(Anion, -1, 46.0055e-3, -97.5, -104.6e3),
    "SO4-2": _ion(Anion, -2, 96.0626e-3, -293.0, -909.27e3),
    "OH-": _ion(Anion, -1, 17.0073e-3, -148.5, -230.0e3),
}


# ---------------------------------------------------------------------------
# eNRTL binary interaction parameters
# ---------------------------------------------------------------------------
# (H2O, Na+ Cl-) from Chen (1982) as regressed in the IDAES eNRTL example;
# all other molecule/ion-pair values use Aspen's documented defaults
# (tau_m,ca = 8, tau_ca,m = -4) because the APV140 databank values are not
# extractable from the .bkp file.
_TAU = {
    ("H2O", "Na+, Cl-"): 9.0234,  # [Chen]
    ("Na+, Cl-", "H2O"): -4.5916,  # [Chen]
    ("H2O", "Na+, NO3-"): 8.0,
    ("Na+, NO3-", "H2O"): -4.0,
    ("H2O", "Na+, NO2-"): 8.0,
    ("Na+, NO2-", "H2O"): -4.0,
    ("H2O", "Na+, SO4-2"): 8.0,
    ("Na+, SO4-2", "H2O"): -4.0,
    ("H2O", "Na+, OH-"): 8.0,
    ("Na+, OH-", "H2O"): -4.0,
}


def get_prop(phases=("Liq", "Vap")):
    """Return a GenericParameterBlock configuration dict.

    Args:
        phases: iterable drawn from {"Liq", "Vap"}. "Liq" builds the
            electrolyte (eNRTL) aqueous phase with all ionic species;
            "Vap" builds an ideal-gas phase with the volatile molecular
            species (H2O, NH3, H2, N2). Passing both enables VLE.
    """
    phases = tuple(phases)
    if not phases or any(p not in ("Liq", "Vap") for p in phases):
        raise ValueError(f"phases must be drawn from ('Liq', 'Vap'), got {phases}")
    has_liq = "Liq" in phases
    has_vap = "Vap" in phases
    vle = has_liq and has_vap

    components = {}
    molecular = {"H2O": _H2O, "NH3": _NH3, "H2": _H2, "N2": _N2}
    for name, data in molecular.items():
        cfg = deepcopy(data)
        if vle and name in ("H2", "N2"):
            # In the two-phase (separation train) variant H2 and N2 are
            # treated as non-condensable, vapor-only gases. Their liquid
            # solubility is tiny (Kpx ~ 7-9 GPa) and keeping them as Henry
            # solutes makes the SmoothVLE formulation depend on a bubble
            # point that does not exist for an H2-laden liquid. Declaring
            # them vapor-only removes the bubble/dew subproblem entirely
            # (see idaes .../phase_equil/smooth_VLE.py: with vapor-only and
            # liquid-only species present, Teq collapses to T). The
            # liquid-only (reactor train) variant keeps them as dissolved
            # Henry solutes, matching Aspen's NPHASE=1 liquid reactor.
            # Type stays Solute (not Component) because the eNRTL expressions
            # classify every true species as solvent/solute/ion.
            cfg["valid_phase_types"] = PT.vaporPhase
            cfg.pop("henry_component", None)
            cfg.pop("enth_mol_liq_comp", None)
            cfg.pop("dens_mol_liq_comp", None)
            cfg.pop("phase_equilibrium_form", None)
            for key in (
                "henry_ref",
                "cp_mol_liq_comp_coeff",
                "enth_mol_form_liq_comp_ref",
                "dens_mol_liq_comp_coeff",
            ):
                cfg["parameter_data"].pop(key, None)
        if not vle:
            cfg.pop("phase_equilibrium_form", None)
        if not has_liq:
            # vapor-only variant: strip liquid/electrolyte config
            cfg.pop("henry_component", None)
            cfg.pop("enth_mol_liq_comp", None)
            cfg.pop("dens_mol_liq_comp", None)
            cfg.pop("pressure_sat_comp", None)
            cfg.pop("relative_permittivity_liq_comp", None)
            cfg["type"] = Component
            for key in (
                "henry_ref",
                "cp_mol_liq_comp_coeff",
                "enth_mol_form_liq_comp_ref",
                "dens_mol_liq_comp_coeff",
                "pressure_sat_comp_coeff",
                "relative_permittivity_liq_comp",
            ):
                cfg["parameter_data"].pop(key, None)
        components[name] = cfg

    if has_liq:
        for name, data in _IONS.items():
            components[name] = deepcopy(data)

    phase_cfg = {}
    if has_liq:
        if vle:
            # IDAES's eNRTL implementation requires every species in the
            # parameter block to be valid in the aqueous phase, so it cannot
            # coexist with the vapor-only H2/N2 needed by the separation
            # train. It also does not couple its activity coefficients into
            # the (Ideal-inherited) VLE fugacity expressions, so eNRTL would
            # be inert for the flash split anyway. The two-phase variant
            # therefore uses the Ideal EOS (Raoult for H2O, Henry for NH3);
            # the liquid-only reactor-train variant keeps full eNRTL.
            phase_cfg["Liq"] = {
                "type": AqueousPhase,
                "equation_of_state": Ideal,
            }
        else:
            phase_cfg["Liq"] = {
                "type": AqueousPhase,
                "equation_of_state": ENRTL,
                "equation_of_state_options": {"reference_state": Symmetric},
            }
    if has_vap:
        phase_cfg["Vap"] = {"type": VaporPhase, "equation_of_state": Ideal}

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
        "state_components": StateIndex.true,
        "state_bounds": {
            "flow_mol": (0, 100, 1e4, pyunits.mol / pyunits.s),
            "temperature": (273.15, 300, 550, pyunits.K),
            "pressure": (5e3, 1e5, 1e7, pyunits.Pa),
        },
        "pressure_ref": (101325, pyunits.Pa),
        "temperature_ref": (298.15, pyunits.K),
    }

    if has_liq and not vle:
        # tau parameters are only meaningful for the eNRTL liquid phase
        config["parameter_data"] = {"Liq_tau": dict(_TAU)}

    if vle:
        config["phases_in_equilibrium"] = [("Vap", "Liq")]
        config["phase_equilibrium_state"] = {("Vap", "Liq"): SmoothVLE}
        config["bubble_dew_method"] = IdealBubbleDew

    return config
